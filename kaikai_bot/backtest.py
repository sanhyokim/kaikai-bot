"""検証エンジン・パラメータ探索・判定表・否決メモ(仕様書 第8章)。

売買の判断はすべて strategy.py の関数を呼ぶ。このファイルは「順番に足を流す」「成績を数える」だけ。
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import candles, collect, config, strategy

DAY_MS = 86_400_000
FIRST_SHARE = 0.6               # 8.2: 最初の60%が前半
MIN_SECOND_HALF_DAYS = 90       # 8.2: 後半90日以上
EXIT_MODES = ("A", "B")
NEIGHBORS = {"1m": ["5m"], "5m": ["1m", "15m"], "15m": ["5m", "1h"], "1h": ["15m"]}

# 8.5 合格基準(先に固定。後から緩めない)
MIN_TRADES = 300
MIN_PF = 1.30
MAX_PF_DROP = 0.5
MAX_DRAWDOWN = 0.10
MAX_MONTH_SHARE = 0.40
MIN_AVG_NET = 1.0
MIN_NEIGHBOR_PF = 0.9

REASON_NAMES = {"stop": "損切り", "take": "利確", "time": "時間切れ",
                "rci_exit": "出口方式B(RCI短期の下向き)", "forced": "強制手仕舞い(欠損の直前・米国株の引け前)"}


# ---------------------------------------------------------------- データの準備

@dataclass
class Prepared:
    coin: str
    tf: str
    hip3: bool
    features: pd.DataFrame
    us_allowed: np.ndarray | None
    force_exit: np.ndarray
    start_ms: int
    split_ms: int
    end_ms: int
    excluded_days: int


def prepare(df1m: pd.DataFrame, coin: str, tf: str, hip3: bool, holidays: set[str]) -> Prepared | None:
    """1分足 → 除外日を落とす → 足と1つ上の足 → 指標の表(1回だけ計算して使い回す)。"""
    if df1m.empty:
        return None
    ex = candles.excluded_days(df1m)
    clean = candles.drop_days(df1m, ex)
    if clean.empty:
        return None
    upper_tf = config.UPPER_TF[tf]
    bars = candles.resample(clean, tf)
    upper = candles.resample(clean, upper_tf)
    if bars.empty:
        return None
    f = strategy.build_features(bars, upper, tf, upper_tf)
    force = f["gap_after"].to_numpy().copy()
    us_allowed = None
    if hip3:
        us_allowed, us_force = strategy.us_time_table(f["close_ms"].to_numpy(), tf, holidays)
        force |= us_force
    start = int(clean["ts_open"].iloc[0])
    end = int(clean["ts_open"].iloc[-1]) + 60_000
    split = start + int((end - start) * FIRST_SHARE)
    return Prepared(coin, tf, hip3, f, us_allowed, force, start, split, end, len(ex))


# ---------------------------------------------------------------- 1回の検証

def simulate(p: Prepared, mask: np.ndarray, params: dict, exit_mode: str, start_ms: int, end_ms: int,
             costs: dict, fixed: dict, capital: float) -> list[dict]:
    """足を時間順に流す。持ち玉は1つまで。指値を出す時刻が [start_ms, end_ms) の取引だけ数える。"""
    f = p.features
    close_ms = f["close_ms"].to_numpy()
    o_ts = f["ts_open"].to_numpy()
    high = f["high"].to_numpy()
    low = f["low"].to_numpy()
    close = f["close"].to_numpy()
    slope9 = f["slope9"].to_numpy()
    atr = f["atr"].to_numpy()
    max_bars = fixed["max_bars"][p.tf]
    safety = strategy.Safety(fixed["daily_loss_limit"], fixed["max_consecutive_losses"], capital)
    lo_i, hi_i = np.searchsorted(close_ms, [start_ms, end_ms])
    cands = np.flatnonzero(mask[lo_i:hi_i]) + lo_i
    trades: list[dict] = []
    free_from = 0
    n = len(close)
    for t in cands:
        if t < free_from or t + 1 >= n:
            continue
        if not strategy.can_open(safety, int(close_ms[t])):
            continue
        fill = t + 1
        if not strategy.fills(close[t], low[fill]):
            free_from = t + 1
            continue
        pos = strategy.open_position(close[t], atr[t], params, max_bars, int(o_ts[fill]), int(close_ms[t]))
        done = None
        j = fill
        while j < n:
            pos.bars_held += 1
            res = strategy.exit_on_bar(pos, high[j], low[j], close[j], slope9[j], exit_mode,
                                       j == fill, bool(p.force_exit[j]), costs["slippage_stop"])
            if res is not None:
                done = strategy.settle(pos, res[0], res[1], res[2], int(close_ms[j]), costs, p.hip3, capital)
                break
            j += 1
        if done is None:      # データの終わりで持ち玉が残った取引は数えない
            break
        trades.append(done)
        strategy.record(safety, done)
        free_from = j         # 手仕舞った足の確定時から次の判断ができる
    return trades


# ---------------------------------------------------------------- 成績

def profit_factor(nets) -> float | None:
    nets = np.asarray(nets, dtype=float)
    if len(nets) == 0:
        return None
    gain = nets[nets > 0].sum()
    loss = -nets[nets < 0].sum()
    if loss == 0:
        return math.inf if gain > 0 else None
    return gain / loss


def max_drawdown(nets, capital: float) -> float:
    eq = capital + np.cumsum(np.asarray(nets, dtype=float))
    eq = np.concatenate([[capital], eq])
    peak = np.maximum.accumulate(eq)
    return float(((peak - eq) / peak).max())


def month_share(trades: list[dict]) -> float | None:
    """最も儲かった1か月の利益 ÷ 総利益。総利益が0以下なら計算できない(None)。"""
    if not trades:
        return None
    total = sum(t["net"] for t in trades)
    if total <= 0:
        return None
    by_month: dict[str, float] = {}
    for t in trades:
        m = datetime.fromtimestamp(t["exit_close_ms"] / 1000, tz=timezone.utc).strftime("%Y-%m")
        by_month[m] = by_month.get(m, 0.0) + t["net"]
    return max(by_month.values()) / total


def stats(trades: list[dict], capital: float) -> dict:
    nets = [t["net"] for t in trades]
    return {
        "trades": len(trades),
        "win_rate": (sum(1 for x in nets if x > 0) / len(nets)) if nets else None,
        "pf": profit_factor(nets),
        "avg_net": (sum(nets) / len(nets)) if nets else None,
        "max_dd": max_drawdown(nets, capital) if nets else 0.0,
        "month_share": month_share(trades),
        "pf_no_cost": profit_factor([t["gross_no_cost"] for t in trades]),
    }


def hour_table(trades: list[dict]) -> dict[int, tuple[int, float | None]]:
    """入った時刻(UTC の時)ごとの (回数, PF)。"""
    out = {}
    for h in range(24):
        sub = [t["net"] for t in trades if (t["entry_open_ms"] // 3_600_000) % 24 == h]
        out[h] = (len(sub), profit_factor(sub))
    return out


# ---------------------------------------------------------------- 探索

def param_grid(search: dict) -> list[dict]:
    keys = ["rci_slope_min", "rci14_max", "macd_lookback", "stop_atr", "take_atr"]
    return [dict(zip(keys, vals)) for vals in itertools.product(*(search[k] for k in keys))]


def _better(a: tuple, b: tuple | None) -> bool:
    """(PF, 取引回数) で比べる。PF が高い方、同じなら取引回数が多い方。それも同じなら先に見た方。"""
    if b is None:
        return True
    return (a[0], a[1]) > (b[0], b[1])


def search_params(p: Prepared, exit_mode: str, cfg: dict, costs: dict) -> dict:
    """前半データだけで全組み合わせを流し、PF が最も高い1組を選ぶ(7.8)。"""
    fixed, capital = cfg["fixed"], cfg["backtest"]["capital"]
    best, best_key = None, None
    masks: dict[tuple, np.ndarray] = {}
    for prm in param_grid(cfg["search"]):
        mk = (prm["rci_slope_min"], prm["rci14_max"], prm["macd_lookback"])
        if mk not in masks:
            masks[mk] = strategy.entry_mask(p.features, prm, fixed["round_number_margin"], p.us_allowed)
        tr = simulate(p, masks[mk], prm, exit_mode, p.start_ms, p.split_ms, costs, fixed, capital)
        pf = profit_factor([t["net"] for t in tr])
        if pf is None:
            continue
        key = (pf, len(tr))
        if _better(key, best_key):
            best, best_key = prm, key
    return dict(best) if best is not None else dict(cfg["initial"])


# ---------------------------------------------------------------- 1設定の検証と判定

def evaluate(p: Prepared, exit_mode: str, cfg: dict, costs: dict) -> dict:
    fixed, capital = cfg["fixed"], cfg["backtest"]["capital"]
    prm = search_params(p, exit_mode, cfg, costs)
    mask = strategy.entry_mask(p.features, prm, fixed["round_number_margin"], p.us_allowed)
    first = simulate(p, mask, prm, exit_mode, p.start_ms, p.split_ms, costs, fixed, capital)
    second = simulate(p, mask, prm, exit_mode, p.split_ms, p.end_ms, costs, fixed, capital)
    return {
        "tf": p.tf, "coin": p.coin, "exit": exit_mode, "params": prm,
        "start_ms": p.start_ms, "split_ms": p.split_ms, "end_ms": p.end_ms,
        "second_days": (p.end_ms - p.split_ms) / DAY_MS,
        "excluded_days": p.excluded_days,
        "first": stats(first, capital), "second": stats(second, capital),
        "second_trades": second, "hours": hour_table(second) if not p.hip3 else None,
        "_prepared": p, "_mask_params": prm,
    }


def judge(rows: list[dict]) -> None:
    """8.5 の7基準で判定する。基準7は同じ銘柄・同じ出口方式の隣の足を見るので、全行がそろってから。"""
    index = {(r["tf"], r["coin"], r["exit"]): r for r in rows}
    for r in rows:
        s1, s2 = r["first"], r["second"]
        fails = []
        if r["second_days"] < MIN_SECOND_HALF_DAYS:
            r["verdict"], r["fails"] = "データ待ち", [f"後半が{r['second_days']:.1f}日(90日以上が必要)"]
            continue
        if s2["trades"] < MIN_TRADES:
            r["verdict"], r["fails"] = "判定不能", [f"基準1 取引回数 {s2['trades']}回(300回以上が必要)"]
            continue
        pf2 = s2["pf"] if s2["pf"] is not None else 0.0
        pf1 = s1["pf"] if s1["pf"] is not None else 0.0
        if pf2 < MIN_PF:
            fails.append(f"基準2 PF {pf2:.2f}(1.30以上が必要)")
        if pf1 - pf2 > MAX_PF_DROP:
            fails.append(f"基準3 前半との差 {pf1 - pf2:.2f}(0.5以内が必要)")
        if s2["max_dd"] > MAX_DRAWDOWN:
            fails.append(f"基準4 最大の落ち込み {s2['max_dd'] * 100:.1f}%(10%以内が必要)")
        ms = s2["month_share"]
        if ms is None or ms > MAX_MONTH_SHARE:
            fails.append("基準5 月別利益の偏り " + ("総利益が0以下で計算できない" if ms is None
                                              else f"{ms * 100:.1f}%(40%以下が必要)"))
        if s2["avg_net"] is None or s2["avg_net"] < MIN_AVG_NET:
            fails.append(f"基準6 平均純利益 {s2['avg_net'] or 0:.2f}ドル(1.0ドル以上が必要)")
        r["fails"] = fails
    for r in rows:
        if "verdict" in r:
            continue
        for ntf in NEIGHBORS[r["tf"]]:
            nb = index.get((ntf, r["coin"], r["exit"]))
            if nb is None or nb.get("verdict") in ("データ待ち", "判定不能"):
                continue      # 隣が判定不能(データ待ちも同じ扱い)なら基準7は適用しない
            npf = nb["second"]["pf"] if nb["second"]["pf"] is not None else 0.0
            if npf < MIN_NEIGHBOR_PF:
                r["fails"].append(f"基準7 隣の足 {ntf} の後半 PF {npf:.2f}(0.9以上が必要)")
        r["verdict"] = "合格" if not r["fails"] else "不合格"


# ---------------------------------------------------------------- 出力

def _d(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _f(x, nd=2, pct=False) -> str:
    if x is None:
        return "-"
    if x == math.inf:
        return "∞"
    return f"{x * 100:.1f}%" if pct else f"{x:.{nd}f}"


def _params_text(p: dict) -> str:
    return (f"傾き{p['rci_slope_min']} / RCI14≤{p['rci14_max']} / GC{p['macd_lookback']}本 / "
            f"損切り{p['stop_atr']}ATR / 利確{p['take_atr']}ATR")


CSV_HEADER = ["足", "銘柄", "出口方式", "選んだパラメータ", "前半期間", "後半期間",
              "取引回数_前半", "取引回数_後半", "勝率_前半", "勝率_後半", "PF_前半", "PF_後半",
              "平均純利益_後半", "最大の落ち込み_後半", "月別利益の最大比率_後半", "除外日数", "判定", "否決理由"]


def table_rows(rows: list[dict]) -> list[list[str]]:
    out = []
    for r in rows:
        s1, s2 = r["first"], r["second"]
        out.append([r["tf"], r["coin"], r["exit"], _params_text(r["params"]),
                    f"{_d(r['start_ms'])}〜{_d(r['split_ms'])}", f"{_d(r['split_ms'])}〜{_d(r['end_ms'])}",
                    str(s1["trades"]), str(s2["trades"]),
                    _f(s1["win_rate"], pct=True), _f(s2["win_rate"], pct=True),
                    _f(s1["pf"]), _f(s2["pf"]), _f(s2["avg_net"]), _f(s2["max_dd"], pct=True),
                    _f(s2["month_share"], pct=True), str(r["excluded_days"]), r["verdict"],
                    " / ".join(r["fails"]) if r["verdict"] != "合格" else ""])
    return out


def write_reports(rows: list[dict], stage: int | None, out_dir: Path, date: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md, cs = out_dir / f"judgement_{date}.md", out_dir / f"judgement_{date}.csv"
    body = table_rows(rows)
    with open(cs, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(body)
    counts = {k: sum(1 for r in rows if r["verdict"] == k) for k in ("合格", "不合格", "判定不能", "データ待ち")}
    lines = [f"# 判定表 {date}" + (f"(第{stage}段階)" if stage else ""), "",
             "要約: " + " / ".join(f"{k} {v}" for k, v in counts.items()), "",
             "| " + " | ".join(CSV_HEADER) + " |", "|" + "---|" * len(CSV_HEADER)]
    lines += ["| " + " | ".join(c.replace("|", "/") for c in row) + " |" for row in body]
    crypto = [r for r in rows if r["hours"] is not None]
    if crypto:
        lines += ["", "## 時間帯別成績(仮想通貨、後半、入った時刻 UTC)", "",
                  "PF(回数)。取引が無い時間帯は「-」。", "",
                  "| 足 | 銘柄 | 出口 | " + " | ".join(f"{h}時" for h in range(24)) + " |",
                  "|" + "---|" * 27]
        for r in crypto:
            cells = [f"{_f(pf)}({n})" if n else "-" for n, pf in (r["hours"][h] for h in range(24))]
            lines.append(f"| {r['tf']} | {r['coin']} | {r['exit']} | " + " | ".join(cells) + " |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md, cs


def write_memo(r: dict, cfg: dict, costs: dict, out_dir: Path) -> Path:
    """8.6 否決メモ(不合格・判定不能の設定ごとに1ファイル)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    p: Prepared = r["_prepared"]
    fixed, capital = cfg["fixed"], cfg["backtest"]["capital"]
    tr = r["second_trades"]
    s2 = r["second"]
    lines = [f"# 否決メモ: {r['tf']} / {r['coin']} / 出口方式{r['exit']}", "",
             f"判定: {r['verdict']}", "",
             "## 1. 設定と選んだパラメータ", "",
             f"- {_params_text(r['params'])}",
             f"- 前半 {_d(r['start_ms'])}〜{_d(r['split_ms'])}、後半 {_d(r['split_ms'])}〜{_d(r['end_ms'])}(後半 {r['second_days']:.1f}日)",
             "", "## 2. 引っかかった基準と数字", ""]
    lines += [f"- {x}" for x in r["fails"]]
    lines += ["", "## 3. 負けの内訳(後半)", "", "| 出口 | 回数 | 合計損益(ドル) |", "|---|---|---|"]
    for reason, name in REASON_NAMES.items():
        sub = [t["net"] for t in tr if t["reason"] == reason]
        lines.append(f"| {name} | {len(sub)} | {sum(sub):.2f} |")
    fees = sum(t["fees"] for t in tr)
    slip = sum(t["slippage"] for t in tr)
    fund = sum(t["funding"] for t in tr)
    lines += ["", "## 4. コスト内訳(後半)", "",
              f"- 手数料合計: {fees:.2f} ドル", f"- 滑り合計: {slip:.2f} ドル", f"- 資金調達料合計: {fund:.2f} ドル",
              f"- コストを0にした時の PF: {_f(s2['pf_no_cost'])}(実際の PF {_f(s2['pf'])})",
              "", "## 5. 入り口条件を1つ外した時の取引回数(後半)", "",
              f"- 6条件すべて: {s2['trades']}回"]
    for name in strategy.CONDITIONS:
        mask = strategy.entry_mask(p.features, r["params"], fixed["round_number_margin"], p.us_allowed,
                                   disabled=frozenset([name]))
        n = len(simulate(p, mask, r["params"], r["exit"], p.split_ms, p.end_ms, costs, fixed, capital))
        lines.append(f"- {strategy.CONDITION_NAMES[name]} を外す: {n}回")
    lines += ["", "## 6. 所見", "", _remark(r)]
    path = out_dir / f"reject_{r['tf']}_{r['coin'].replace(':', '_')}_{r['exit']}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _remark(r: dict) -> str:
    s2 = r["second"]
    if r["verdict"] == "判定不能":
        return f"後半の取引が{s2['trades']}回で、判定に必要な300回に届かない。"
    pf, pf0 = s2["pf"], s2["pf_no_cost"]
    if pf is not None and pf0 is not None and pf < 1.0 <= pf0:
        return "コストを引く前は勝っており、コスト負け。"
    if pf0 is not None and pf0 < 1.0:
        return "コストを引く前から負けており、ロジック負け。"
    return "コスト前後とも PF は1.0以上だが、合格基準のどれかに届かない。"


# ---------------------------------------------------------------- 実行

def run(stage: int | None = None, coin: str | None = None, tf: str | None = None, exit_mode: str | None = None,
        db_path: Path = config.MARKET_DB, out_dir: Path = config.REPORTS_DIR, memo_dir: Path = config.MEMOS_DIR,
        date: str | None = None) -> list[dict]:
    cfg, costs, holidays = config.strategy(), config.costs(), config.us_holidays()
    coins = [config.coin_info(coin)] if coin else config.coins(stage or 1)
    tfs = [tf] if tf else config.TIMEFRAMES
    modes = [exit_mode] if exit_mode else list(EXIT_MODES)
    con = collect.connect(db_path)
    rows = []
    for info in coins:
        df1m = candles.load_1m(con, info["coin"])
        for t in tfs:
            p = prepare(df1m, info["coin"], t, info["hip3"], holidays)
            if p is None:
                print(f"{t} {info['coin']}: データなし")
                continue
            for m in modes:
                print(f"検証中 {t} {info['coin']} 出口{m}")
                rows.append(evaluate(p, m, cfg, costs))
    con.close()
    judge(rows)
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    md, cs = write_reports(rows, stage, out_dir, date)
    for r in rows:
        if r["verdict"] in ("不合格", "判定不能"):
            write_memo(r, cfg, costs, memo_dir)
    print(f"判定表: {md}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="検証(バックテスト)")
    ap.add_argument("--stage", type=int, choices=[1, 2])
    ap.add_argument("--coin")
    ap.add_argument("--tf", choices=config.TIMEFRAMES)
    ap.add_argument("--exit", choices=EXIT_MODES)
    a = ap.parse_args()
    run(stage=a.stage, coin=a.coin, tf=a.tf, exit_mode=a.exit)


if __name__ == "__main__":
    main()
