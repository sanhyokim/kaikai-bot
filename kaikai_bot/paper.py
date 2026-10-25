"""紙上運転(仕様書 第9章)。注文は出さず、検証と同じ判断・同じ約定規則・同じコストで記録だけする。

売買の判断は strategy.py の関数を呼ぶだけ。足を1本ずつ確定順に流す点も検証と同じ。
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import candles, collect, config, exchange, strategy

MINUTE_MS = 60_000
DAY_MS = 86_400_000
WINDOW_DAYS = 8          # 指標の計算に使う直近の日数(1時間足の上の4時間足 SMA25+10本 ≒ 6日に余裕を足した)
FETCH_BACK_MS = 120 * MINUTE_MS
STATUS_PATH = config.LOGS_DIR / "paper_status.json"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, tf TEXT, coin TEXT, exit_mode TEXT, params TEXT, kind TEXT, started_ms INTEGER
);
CREATE TABLE IF NOT EXISTS state (
    run_id TEXT PRIMARY KEY, last_close_ms INTEGER, pending TEXT, position TEXT, safety TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    run_id TEXT, signal_close_ms INTEGER, entry_open_ms INTEGER, entry_price REAL,
    exit_close_ms INTEGER, exit_price REAL, reason TEXT, bars_held INTEGER,
    fees REAL, slippage REAL, funding REAL, gross_no_cost REAL, net REAL,
    PRIMARY KEY (run_id, entry_open_ms)
);
"""


def connect(path: Path = config.PAPER_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def run_id(tf: str, coin: str, exit_mode: str) -> str:
    return f"{tf}_{coin}_{exit_mode}"


# ---------------------------------------------------------------- 対象の設定(9.1)

def parse_params(text: str) -> dict:
    """判定表の「選んだパラメータ」の文字列を戻す。"""
    parts = [p.strip() for p in text.split("/")]
    return {"rci_slope_min": float(parts[0].removeprefix("傾き")),
            "rci14_max": float(parts[1].removeprefix("RCI14≤")),
            "macd_lookback": int(parts[2].removeprefix("GC").removesuffix("本")),
            "stop_atr": float(parts[3].removeprefix("損切り").removesuffix("ATR")),
            "take_atr": float(parts[4].removeprefix("利確").removesuffix("ATR"))}


def select_targets(judgement_csv: Path | list[Path]) -> list[dict]:
    """合格した設定すべて。無ければ暫定設定を1つ(仕組みの確認用。昇格の対象にしない)。

    暫定は、後半 PF が数字で出ている行の中で最も高いもの(9.1 は「不合格の中で」だが、
    不合格が無い時のために全行から選ぶ)。どれも数字が無ければ判定表の先頭行(1分足・BTC・出口A)。
    """
    paths = judgement_csv if isinstance(judgement_csv, list) else [judgement_csv]
    rows = []
    for path in paths:
        with open(path, encoding="utf-8-sig") as f:
            rows += list(csv.DictReader(f))
    passed = [r for r in rows if r["判定"] == "合格"]
    if passed:
        return [_target(r, "合格") for r in passed]

    def pf(r):
        v = r["PF_後半"]
        return float("inf") if v == "∞" else float(v) if v not in ("-", "") else None
    scored = [r for r in rows if pf(r) is not None]
    pool = [r for r in scored if r["判定"] == "不合格"] or scored
    best = max(pool, key=pf) if pool else rows[0]
    return [_target(best, "暫定")]


def _target(r: dict, kind: str) -> dict:
    return {"tf": r["足"], "coin": r["銘柄"], "exit": r["出口方式"],
            "params": parse_params(r["選んだパラメータ"]), "kind": kind}


def latest_judgements(reports: Path = config.REPORTS_DIR) -> list[Path]:
    """第1段階と第2段階の、それぞれ最新の判定表(1設定だけの実行の表は使わない)。"""
    out = []
    for pattern in ("judgement_????-??-??.csv", "judgement_????-??-??_stage2.csv"):
        files = sorted(reports.glob(pattern))
        if files:
            out.append(files[-1])
    return out


# ---------------------------------------------------------------- 1設定の状態

class Runner:
    """1設定の紙上運転。状態は paper.db に保存し、落ちても読み戻して続ける(9.4)。"""

    def __init__(self, con: sqlite3.Connection, target: dict, cfg: dict, costs: dict, now_ms: int):
        self.con, self.cfg, self.costs = con, cfg, costs
        self.tf, self.coin, self.exit = target["tf"], target["coin"], target["exit"]
        self.params, self.kind = target["params"], target["kind"]
        self.id = run_id(self.tf, self.coin, self.exit)
        self.hip3 = config.coin_info(self.coin)["hip3"]
        self.capital = cfg["backtest"]["capital"]
        self.max_bars = cfg["fixed"]["max_bars"][self.tf]
        con.execute("INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?,?,?)",
                    (self.id, self.tf, self.coin, self.exit, json.dumps(self.params), self.kind, now_ms))
        con.commit()
        row = con.execute("SELECT last_close_ms, pending, position, safety FROM state WHERE run_id=?",
                          (self.id,)).fetchone()
        fx = cfg["fixed"]
        self.safety = strategy.Safety(fx["daily_loss_limit"], fx["max_consecutive_losses"], self.capital)
        self.pending: dict | None = None
        self.pos: strategy.Position | None = None
        self.last_close_ms: int | None = None
        self.last_bar: dict | None = None
        if row:
            self.last_close_ms = row[0]
            self.pending = json.loads(row[1]) if row[1] else None
            self.pos = strategy.Position(**json.loads(row[2])) if row[2] else None
            s = json.loads(row[3])
            self.safety.day_pnl = {int(k): v for k, v in s["day_pnl"].items()}
            self.safety.losing_streak = s["losing_streak"]
            self.safety.blocked_day = s["blocked_day"]
            self.last_bar = s.get("last_bar")

    @property
    def busy(self) -> bool:
        return self.pending is not None or self.pos is not None

    def save(self) -> None:
        s = {"day_pnl": self.safety.day_pnl, "losing_streak": self.safety.losing_streak,
             "blocked_day": self.safety.blocked_day, "last_bar": self.last_bar}
        self.con.execute("INSERT OR REPLACE INTO state VALUES (?,?,?,?,?)",
                         (self.id, self.last_close_ms, json.dumps(self.pending) if self.pending else None,
                          json.dumps(asdict(self.pos)) if self.pos else None, json.dumps(s)))
        self.con.commit()

    def _close(self, reason: str, price: float, raw: float, close_ms: int) -> None:
        t = strategy.settle(self.pos, reason, price, raw, close_ms, self.costs, self.hip3, self.capital)
        self.con.execute("INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (self.id, t["signal_close_ms"], t["entry_open_ms"], t["entry_price"], t["exit_close_ms"],
                          t["exit_price"], t["reason"], t["bars_held"], t["fees"], t["slippage"], t["funding"],
                          t["gross_no_cost"], t["net"]))
        strategy.record(self.safety, t)
        self.pos = None

    def step(self, f: pd.DataFrame, j: int, mask: np.ndarray, force: np.ndarray,
             others_busy: bool, shared_position: bool) -> None:
        """確定した足 j を1本処理する。検証の simulate と同じ順番。"""
        ts_open = int(f["ts_open"].iat[j])
        close_ms = int(f["close_ms"].iat[j])
        high, low, close = float(f["high"].iat[j]), float(f["low"].iat[j]), float(f["close"].iat[j])
        contiguous = self.last_close_ms is not None and ts_open == self.last_close_ms
        # 欠損をまたいだ: 持ち玉は欠損の直前の足の終値で手仕舞い、出していた指値は取り消す
        if not contiguous and self.last_close_ms is not None:
            if self.pos is not None and self.last_bar is not None:
                self._close("forced", self.last_bar["close"], self.last_bar["close"], self.last_bar["close_ms"])
            self.pending = None
        filled_now = False
        if self.pending is not None:
            if strategy.fills(self.pending["limit"], low):
                self.pos = strategy.open_position(self.pending["limit"], self.pending["atr"], self.params,
                                                  self.max_bars, ts_open, self.pending["signal_close_ms"])
                filled_now = True
            self.pending = None
        if self.pos is not None:
            self.pos.bars_held += 1
            res = strategy.exit_on_bar(self.pos, high, low, close, float(f["slope9"].iat[j]), self.exit,
                                       filled_now, bool(force[j]), self.costs["slippage_stop"])
            if res is not None:
                self._close(res[0], res[1], res[2], close_ms)
        if (self.pos is None and mask[j] and not (shared_position and others_busy)
                and strategy.can_open(self.safety, close_ms)):
            self.pending = {"limit": close, "atr": float(f["atr"].iat[j]), "signal_close_ms": close_ms}
        self.last_close_ms = close_ms
        self.last_bar = {"close": close, "close_ms": close_ms}


# ---------------------------------------------------------------- 成績と昇格候補(9.3)

PAPER_MIN_DAYS = 28
PAPER_MIN_TRADES = 50
PAPER_MIN_PF = 1.20
PAPER_MIN_PF_RATIO = 0.70
PAPER_MAX_DD = 0.10


def run_results(con: sqlite3.Connection, now_ms: int, judgement_csv: Path | list[Path] | None = None,
                capital: float = 1000.0) -> list[dict]:
    """設定ごとの紙上の成績と、9.3 の5基準による昇格候補の印。暫定設定は候補にしない。"""
    from . import backtest as bt
    second_pf = {}
    paths = [] if judgement_csv is None else judgement_csv if isinstance(judgement_csv, list) else [judgement_csv]
    for path in paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                v = r["PF_後半"]
                second_pf[run_id(r["足"], r["銘柄"], r["出口方式"])] = (
                    float("inf") if v == "∞" else float(v) if v not in ("-", "") else None)
    out = []
    for rid, kind, started in con.execute("SELECT run_id, kind, started_ms FROM runs ORDER BY run_id"):
        nets = [x[0] for x in con.execute("SELECT net FROM trades WHERE run_id=? ORDER BY exit_close_ms", (rid,))]
        pf = bt.profit_factor(nets)
        dd = bt.max_drawdown(nets, capital) if nets else 0.0
        days = (now_ms - started) / DAY_MS
        ref = second_pf.get(rid)
        checks = {
            "期間28日以上": days >= PAPER_MIN_DAYS,
            "取引50回以上": len(nets) >= PAPER_MIN_TRADES,
            "PF1.20以上": pf is not None and pf >= PAPER_MIN_PF,
            "検証の後半PFの70%以上": pf is not None and ref is not None and pf >= PAPER_MIN_PF_RATIO * ref,
            "最大の落ち込み10%以内": dd <= PAPER_MAX_DD,
        }
        out.append({"run_id": rid, "kind": kind, "days": days, "trades": len(nets), "pf": pf,
                    "net": sum(nets), "max_dd": dd, "checks": checks,
                    "candidate": kind != "暫定" and all(checks.values())})
    return out


# ---------------------------------------------------------------- 1回分の処理

def features_for(df1m: pd.DataFrame, tf: str, hip3: bool, holidays: set[str]):
    upper_tf = config.UPPER_TF[tf]
    bars = candles.resample(df1m, tf)
    upper = candles.resample(df1m, upper_tf)
    f = strategy.build_features(bars, upper, tf, upper_tf)
    us_allowed, force = None, np.zeros(len(f), dtype=bool)
    if hip3:
        us_allowed, force = strategy.us_time_table(f["close_ms"].to_numpy(), tf, holidays)
    return f, us_allowed, force


def process(runners: list[Runner], market: sqlite3.Connection, now_ms: int, holidays: set[str],
            shared_position: bool) -> None:
    """保存庫の1分足から、各設定の新しく確定した足を時刻順に処理する。

    欠損の直前の手仕舞いは、紙上では「欠損が分かった時点で、直前の足の終値で」行う(検証と同じ値段)。
    """
    cache: dict[tuple, tuple] = {}
    events = []
    for r in runners:
        key = (r.coin, r.tf)
        if key not in cache:
            df = candles.load_1m(market, r.coin, start_ms=now_ms - WINDOW_DAYS * DAY_MS)
            cache[key] = features_for(df, r.tf, r.hip3, holidays) if not df.empty else None
        if cache[key] is None:
            continue
        f, us_allowed, force = cache[key]
        mask = strategy.entry_mask(f, r.params, r.cfg["fixed"]["round_number_margin"], us_allowed)
        close_ms = f["close_ms"].to_numpy()
        start = 0 if r.last_close_ms is None else int(np.searchsorted(close_ms, r.last_close_ms, side="right"))
        if r.last_close_ms is None:
            start = len(f) - 1           # 初回は最新の確定足から始める(過去を紙上で取引しない)
        for j in range(max(start, 0), len(f)):
            events.append((int(close_ms[j]), r.id, r, f, j, mask, force))
    events.sort(key=lambda e: (e[0], e[1]))
    for _, _, r, f, j, mask, force in events:
        others = any(o.busy for o in runners if o is not r)
        r.step(f, j, mask, force, others, shared_position)
    for r in runners:
        r.save()


def write_status(path: Path, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(kw)
    path.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")


def loop(targets: list[dict], shared_position: bool = True) -> None:
    cfg, costs, holidays = config.strategy(), config.costs(), config.us_holidays()
    market = collect.connect()
    paper = connect()
    runners = [Runner(paper, t, cfg, costs, collect.now_ms()) for t in targets]
    coins = sorted({r.coin for r in runners})
    print("紙上運転を開始:", ", ".join(f"{r.id}({r.kind})" for r in runners))
    while True:
        now = collect.now_ms()
        ok = True
        for coin in coins:
            try:
                last = collect.latest_ts(market, coin)
                start = now - FETCH_BACK_MS if last is None else min(last - FETCH_BACK_MS, now - FETCH_BACK_MS)
                rows = [x for x in exchange.candles_1m(coin, max(start, 0), now) if x["ts_open"] + MINUTE_MS <= now]
                collect.upsert(market, coin, rows, now)
            except exchange.ExchangeError as e:   # 3回再試行しても失敗: 警告を出し、次の足で再開(9.4)
                ok = False
                collect.alert(f"紙上運転: {coin} の取得に失敗: {e}")
        process(runners, market, now, holidays, shared_position)
        write_status(STATUS_PATH, last_loop_ms=now, last_ok=ok,
                     runs=[{"id": r.id, "kind": r.kind, "busy": r.busy} for r in runners])
        time.sleep(max(5.0, 60 - (time.time() % 60) + 3))   # 毎分、足が閉じた3秒後


def watch(argv: list[str]) -> None:
    """落ちたら自動で起動し直す小さな見張り(9.4)。"""
    while True:
        code = subprocess.call([sys.executable, "-m", "kaikai_bot.paper", *argv])
        collect.alert(f"紙上運転が終了(終了コード {code})。10秒後に再起動")
        time.sleep(10)


def main() -> None:
    ap = argparse.ArgumentParser(description="紙上運転")
    ap.add_argument("--watch", action="store_true", help="落ちたら自動で再起動する")
    a, rest = ap.parse_known_args()
    if a.watch:
        watch(rest)
        return
    files = latest_judgements()
    if not files:
        raise SystemExit("判定表がありません。先に python -m kaikai_bot.backtest --stage 1 を実行してください")
    loop(select_targets(files))


if __name__ == "__main__":
    main()
