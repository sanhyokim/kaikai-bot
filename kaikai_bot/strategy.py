"""入り口・出口・安全装置の判断(仕様書 第7章、第2版)。

売買の判断はこのファイルにだけ書く。検証(backtest)・紙上運転(paper)・本番(live)は、
ここにある同じ関数を呼ぶだけにする(仕様書 12.1、15章3)。

入力と出力の形:
- `build_features(足, 1つ上の足, 足の種類)` で指標の表を作る。
- `entry_mask(指標の表, パラメータ, …)` が、各足の確定時に「買い指値を出す」(合図)かどうかを返す。
- `entry_tags(指標の表, パラメータ)` が、記録だけの情報(印A/B、パラボリックの陽/陰)を返す。
- `can_open(安全装置の状態, 時刻, …)` が、安全装置の上で新規に入ってよいかを返す。
- `fills(指値, 次の足)` が、指値が約定したかを返す。
- `exit_on_bar(持ち玉, 足, …)` が、その足で手仕舞うか(理由と値段)を返す。
- `settle(持ち玉, 理由, 値段, …)` が、コストを引いた1取引の損益を返す。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import candles, indicators as ind
from .config import TF_MINUTES

MINUTE_MS = 60_000
HOUR_MS = 60 * MINUTE_MS
DAY_MS = 24 * HOUR_MS
COOLDOWN_BARS = 10            # 7.7 / 5.4: 欠損明け10本は入らない
NY_TZ = "America/New_York"
US_ENTRY_START = 9 * 60 + 50  # 7.2: NY 9:50〜15:30 のみ新規
US_ENTRY_END = 15 * 60 + 30
US_FORCE_EXIT = 15 * 60 + 55  # 7.2: 15:55 までに強制手仕舞い

ROUND_WALL_SHARE = 0.1        # 6.6: 壁の直下 = すぐ上のキリ番まで刻み幅の1割以内

# 入り口の各条件(7.3、7.4)。否決メモ・中間報告の「1つ外した時の合図回数」に使う
CONDITIONS = ("env1", "env2", "req1", "req2", "req3", "markA", "markB")
CONDITION_NAMES = {
    "env1": "環1 終値が75本線の上",
    "env2": "環2 上の足の25本線が上向き",
    "req1": "必1 RCI短期の急角度",
    "req2": "必2 終値が5本線の上",
    "req3": "必3 キリのいい数字の壁の直下でない",
    "markA": "印A マイナス圏のゴールデンクロス",
    "markB": "印B RCI中期の底からの上向き",
}


# ---------------------------------------------------------------- キリのいい数字(7.4 条件6)

def round_step(price: float) -> float:
    """6.6 刻み幅: 10^(floor(log10(値段)) - 1)。例 67,420→1,000、2,480→100、179→10。"""
    return 10.0 ** (math.floor(math.log10(price)) - 1)


def next_round_number(price: float) -> float:
    """price 以上で最も近い刻み幅の倍数(price ちょうどが倍数なら price)。"""
    step = round_step(price)
    k = math.ceil(price / step - 1e-9)
    return k * step


def below_round_wall(price):
    """6.6 壁の直下: すぐ上のキリ番 R との差 R - p が刻み幅の1割以内。配列でも可。"""
    p = np.asarray(price, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = 10.0 ** (np.floor(np.log10(p)) - 1)
        nxt = np.ceil(p / step - 1e-9) * step
        out = (nxt - p) <= step * ROUND_WALL_SHARE * (1 + 1e-9)
    out = np.where(np.isfinite(p) & (p > 0), out, False)
    return bool(out) if out.ndim == 0 else out


# ---------------------------------------------------------------- 時間帯(7.2)

def us_time_table(close_ms: np.ndarray, tf: str, holidays: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """米国株用。(入ってよいか, この足の終値で強制手仕舞いか) を各足について返す。

    入ってよい: 足 t の確定時刻(=指値を出す時刻)が NY の平日・休日以外の 9:50 以上 15:30 未満で、
    約定を見る次の足の終わりが 15:55 以前。
    強制手仕舞い: 次の足の終わりが 15:55 を過ぎる(または日をまたぐ)なら、この足の終値で手仕舞う。
    夏時間・冬時間は時刻計算ライブラリに任せる。
    """
    width = TF_MINUTES[tf] * MINUTE_MS
    now = pd.to_datetime(close_ms, unit="ms", utc=True).tz_convert(NY_TZ)
    nxt = pd.to_datetime(close_ms + width, unit="ms", utc=True).tz_convert(NY_TZ)
    now_min = now.hour * 60 + now.minute
    nxt_min = nxt.hour * 60 + nxt.minute
    same_day = (now.date == nxt.date)
    dates = now.strftime("%Y-%m-%d")
    trading_day = (now.weekday < 5) & ~np.isin(dates, list(holidays))
    allowed = (trading_day & (now_min >= US_ENTRY_START) & (now_min < US_ENTRY_END)
               & same_day & (nxt_min <= US_FORCE_EXIT))
    force = ~same_day | (nxt_min > US_FORCE_EXIT)
    return np.asarray(allowed), np.asarray(force)


# ---------------------------------------------------------------- 指標の表

def build_features(bars: pd.DataFrame, upper: pd.DataFrame, tf: str, upper_tf: str) -> pd.DataFrame:
    """足 t の確定時点で分かる値だけで作った指標の表(6章、7.3)。"""
    f = bars[["ts_open", "open", "high", "low", "close"]].reset_index(drop=True).copy()
    width = TF_MINUTES[tf] * MINUTE_MS
    f["close_ms"] = f["ts_open"] + width
    c, h, lo = f["close"], f["high"], f["low"]
    f["sma5"] = ind.sma(c, 5)
    f["sma75"] = ind.sma(c, 75)
    sar = ind.psar(h, lo, c)
    f["psar_bull"] = ind.psar_bull(h, lo, sar)      # 第2版では条件に使わず記録だけ(6.2)
    f["psar_bear"] = ind.psar_bear(h, lo, sar)
    f["gc_neg"] = ind.macd(c)["gc_neg"]
    r9 = ind.rci(c, 9)
    r14 = ind.rci(c, 14)
    f["slope9"] = r9 - r9.shift(1)
    f["rci14_prev"] = r14.shift(1)
    f["rci14_turn_up"] = (r14 > r14.shift(1)) & (r14.shift(1) <= r14.shift(2))
    f["atr"] = ind.atr(h, lo, c, 14)
    f["upper_sma25_up"] = _upper_rising(f["close_ms"].to_numpy(), upper, upper_tf)
    f["gap_after"] = candles.gap_after(bars, tf)
    f["since_gap"] = candles.bars_since_gap(bars, tf)
    return f


def _upper_rising(close_ms: np.ndarray, upper: pd.DataFrame, upper_tf: str) -> np.ndarray:
    """1つ上の足の SMA(25) が上向きか。足 t の確定時刻までに閉じた、直前の上の足を使う。

    直前の上の足が欠けている(1本前の区間の足が無い)時は False。
    """
    out = np.zeros(len(close_ms), dtype=bool)
    if upper.empty:
        return out
    uw = TF_MINUTES[upper_tf] * MINUTE_MS
    u_close = upper["ts_open"].to_numpy() + uw
    u_up = ind.rising(ind.sma(upper["close"].reset_index(drop=True), 25)).to_numpy()
    idx = np.searchsorted(u_close, close_ms, side="right") - 1
    ok = idx >= 0
    fresh = np.zeros(len(close_ms), dtype=bool)
    fresh[ok] = u_close[idx[ok]] > close_ms[ok] - uw
    out[ok & fresh] = u_up[idx[ok & fresh]]
    return out


# ---------------------------------------------------------------- 入り口(7.3〜7.5、第2版)

def entry_conditions(f: pd.DataFrame, params: dict) -> dict[str, np.ndarray]:
    """7.3 環境フィルタと 7.4 入り口の各条件を、足ごとの真偽で返す。"""
    lb = int(params["macd_lookback"])
    gc_recent = f["gc_neg"].astype(float).rolling(lb, min_periods=1).max().fillna(0).astype(bool)
    return {
        "env1": (f["close"] > f["sma75"]).to_numpy(),
        "env2": f["upper_sma25_up"].to_numpy(dtype=bool),
        "req1": (f["slope9"] >= params["rci_slope_min"]).to_numpy(),
        "req2": (f["close"] > f["sma5"]).to_numpy(),
        "req3": ~below_round_wall(f["close"].to_numpy()),
        "markA": gc_recent.to_numpy(),
        "markB": ((f["rci14_prev"] <= params["rci14_max"]) & f["rci14_turn_up"]).to_numpy(),
    }


def entry_mask(f: pd.DataFrame, params: dict, us_allowed: np.ndarray | None = None,
               disabled: frozenset = frozenset()) -> np.ndarray:
    """各足の確定時に買い指値を出す(合図)か。環1・環2・必1〜3 すべてと、印A か印B の1つ以上。

    加えて ATR が計算できること、欠損明け10本でないこと、次の足が欠けていないこと、米国株は時間帯。
    disabled: 中間報告・否決メモの分析用に「課さない」条件の名前(CONDITIONS)。課さない条件は満たした扱い
    (印を課さないと印のどちらか1つ以上という条件も満たされる)。売買には使わない。
    持ち玉や安全装置は見ない(それは can_open と呼び出し側の順番で扱う)。
    """
    c = entry_conditions(f, params)
    n = len(f)
    ok = {k: (np.ones(n, dtype=bool) if k in disabled else v) for k, v in c.items()}
    m = ok["env1"] & ok["env2"] & ok["req1"] & ok["req2"] & ok["req3"] & (ok["markA"] | ok["markB"])
    m &= f["atr"].notna().to_numpy()
    m &= f["since_gap"].to_numpy() > COOLDOWN_BARS
    m &= ~f["gap_after"].to_numpy()          # 次の足が欠けていたら約定を見られない
    if us_allowed is not None:
        m &= us_allowed
    return m


def entry_tags(f: pd.DataFrame, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """記録だけの情報(7.4)を足ごとに返す: どの印で入ったか(A / B / 両方)と、パラボリックの陽/陰。"""
    c = entry_conditions(f, params)
    a, b = c["markA"], c["markB"]
    mark = np.where(a & b, "両方", np.where(a, "A", np.where(b, "B", "なし")))
    bull, bear = f["psar_bull"].to_numpy(dtype=bool), f["psar_bear"].to_numpy(dtype=bool)
    psar = np.where(bull, "陽", np.where(bear, "陰", "どちらでもない"))
    return mark, psar


def fills(limit_price: float, next_low: float) -> bool:
    """7.5: 次の足の安値が指値以下(等号含む)なら指値で約定。"""
    return next_low <= limit_price


# ---------------------------------------------------------------- 持ち玉と出口(7.6)

@dataclass
class Position:
    entry_price: float
    entry_open_ms: int        # 約定した足の開始時刻
    signal_close_ms: int      # 指値を出した時刻(足 t の確定時刻)
    stop: float
    take: float
    atr: float
    max_bars: int
    bars_held: int = 0
    mark: str = ""            # 記録だけ: どの印で入ったか
    psar: str = ""            # 記録だけ: 入った時のパラボリック


def open_position(limit_price: float, atr: float, params: dict, max_bars: int,
                  entry_open_ms: int, signal_close_ms: int, mark: str = "", psar: str = "") -> Position:
    """入値と ATR(足 t の値で固定)から損切り・利確の値段を決める。"""
    return Position(entry_price=limit_price, entry_open_ms=entry_open_ms, signal_close_ms=signal_close_ms,
                    stop=limit_price - atr * params["stop_atr"],
                    take=limit_price + atr * params["take_atr"],
                    atr=atr, max_bars=max_bars, mark=mark, psar=psar)


def exit_on_bar(pos: Position, high: float, low: float, close: float, slope9: float,
                exit_mode: str, is_entry_bar: bool, force_exit: bool, slippage_stop: float):
    """足が閉じた時に呼ぶ。手仕舞うなら (理由, 約定値段, 滑りを除いた値段) を返す。手仕舞わなければ None。

    呼ぶ前に pos.bars_held を1増やしておく(約定した足が1本目)。
    順番: 損切り → 利確 → 出口方式B → 時間切れ → 欠損の直前・米国株の引け前の強制手仕舞い。
    同じ足で損切りと利確の両方に届いたら損切り(保守的)。
    約定した足では利確を見ない(その足の高値が約定より前だったか分からないため。保守的)。
    """
    if low <= pos.stop:
        return "stop", pos.stop * (1 - slippage_stop), pos.stop
    if not is_entry_bar and high >= pos.take:
        return "take", pos.take, pos.take
    if exit_mode == "B" and slope9 < 0:
        return "rci_exit", close, close
    if pos.bars_held >= pos.max_bars:
        return "time", close, close
    if force_exit:
        return "forced", close, close
    return None


def settle(pos: Position, reason: str, exit_price: float, raw_exit_price: float, exit_close_ms: int,
           costs: dict, hip3: bool, capital: float) -> dict:
    """1取引の損益とコスト内訳(5.6)。入り口は maker、利確は maker、その他は taker。"""
    mult = costs["hip3_fee_multiplier"] if hip3 else 1.0
    qty = capital / pos.entry_price
    entry_fee = capital * costs["maker_fee"] * mult
    exit_rate = costs["maker_fee"] if reason == "take" else costs["taker_fee"]
    exit_fee = qty * exit_price * exit_rate * mult
    slippage = qty * (raw_exit_price - exit_price)
    hours = (exit_close_ms - pos.entry_open_ms) / HOUR_MS
    funding = capital * costs["funding_per_hour"] * hours
    gross = qty * (exit_price - pos.entry_price)
    net = gross - entry_fee - exit_fee - funding
    return {
        "signal_close_ms": pos.signal_close_ms,
        "entry_open_ms": pos.entry_open_ms,
        "exit_close_ms": exit_close_ms,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "stop": pos.stop,
        "take": pos.take,
        "reason": reason,
        "bars_held": pos.bars_held,
        "mark": pos.mark,
        "psar": pos.psar,
        "fees": entry_fee + exit_fee,
        "slippage": slippage,
        "funding": funding,
        "gross_no_cost": qty * (raw_exit_price - pos.entry_price),
        "net": net,
    }


# ---------------------------------------------------------------- 安全装置(7.7)

@dataclass
class Safety:
    daily_loss_limit: float
    max_consecutive_losses: int
    capital: float
    day_pnl: dict = field(default_factory=dict)   # UTC の日 → その日に確定した損益
    losing_streak: int = 0
    blocked_day: int | None = None                # この UTC の日の終わりまで新規停止


def can_open(s: Safety, now_ms: int) -> bool:
    """1日の損失上限と連敗停止。now_ms は判断する時刻(足の確定時刻)。"""
    day = now_ms // DAY_MS
    if s.blocked_day is not None:
        if day <= s.blocked_day:
            return False
        s.blocked_day = None
        s.losing_streak = 0
    return s.day_pnl.get(day, 0.0) > -s.daily_loss_limit * s.capital


def record(s: Safety, trade: dict) -> None:
    day = trade["exit_close_ms"] // DAY_MS
    s.day_pnl[day] = s.day_pnl.get(day, 0.0) + trade["net"]
    if trade["net"] < 0:
        s.losing_streak += 1
        if s.losing_streak >= s.max_consecutive_losses:
            s.blocked_day = day
    else:
        s.losing_streak = 0
