"""検証: 中間報告・予備判定・正式判定・判定表・否決メモ(仕様書 第2版 第8章)。

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
MIN_SECOND_HALF_DAYS = 90       # 8.2: 正式判定は後半90日以上
EXIT_MODES = ("A", "B")
MODES = ("frequency", "preliminary", "formal")
MODE_NAMES = {"frequency": "中間報告", "preliminary": "予備判定", "formal": "正式判定"}
NEIGHBORS = {"1m": ["5m"], "5m": ["1m", "15m"], "15m": ["5m", "1h"], "1h": ["15m"]}

# 8.5 合格基準(第2版。先に固定し、後から緩めない)。基準1・2は足ごと
MIN_TRADES = {"1m": 300, "5m": 200, "15m": 120, "1h": 60}
MIN_PF = {"1m": 1.30, "5m": 1.35, "15m": 1.40, "1h": 1.50}
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
    full_days: int = 0


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
    full_days = int(clean["ts_open"].floordiv(DAY_MS).nunique())
    return Prepared(coin, tf, hip3, f, us_allowed, force, start, split, end, len(ex), full_days)


# ---------------------------------------------------------------- 1回の検証

def simulate(p: Prepared, mask: np.ndarray, params: dict, exit_mode: str, start_ms: int, end_ms: int,
             costs: dict, fixed: dict, capital: float, tags: tuple | None = None) -> list[dict]:
    """足を時間順に流す。持ち玉は1つまで。指値を出す時刻が [start_ms, end_ms) の取引だけ数える。

    tags: strategy.entry_tags の結果(記録だけの情報)。省略すると記録は空。
    """
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
        mark, psar = (str(tags[0][t]), str(tags[1][t])) if tags is not None else ("", "")
        pos = strategy.open_position(close[t], atr[t], params, max_bars, int(o_ts[fill]), int(close_ms[t]),
                                     mark, psar)
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


def run_one(p: Prepared, params: dict, exit_mode: str, start_ms: int, end_ms: int, cfg: dict, costs: dict,
            disabled: frozenset = frozenset()) -> list[dict]:
    mask = strategy.entry_mask(p.features, params, p.us_allowed, disabled)
    return simulate(p, mask, params, exit_mode, start_ms, end_ms, costs, cfg["fixed"],
                    cfg["backtest"]["capital"], strategy.entry_tags(p.features, params))


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

    def part(key, value):
        sub = [t["net"] for t in trades if t.get(key) == value]
        return len(sub), profit_factor(sub)
    return {
        "trades": len(trades),
        "win_rate": (sum(1 for x in nets if x > 0) / len(nets)) if nets else None,
        "pf": profit_factor(nets),
        "avg_net": (sum(nets) / len(nets)) if nets else None,
        "max_dd": max_drawdown(nets, capital) if nets else 0.0,
        "month_share": month_share(trades),
        "pf_no_cost": profit_factor([t["gross_no_cost"] for t in trades]),
        "mark": {k: part("mark", k) for k in ("A", "B", "両方")},
        "psar": {k: part("psar", k) for k in ("陽", "陰")},
    }


def hour_table(trades: list[dict]) -> dict[int, tuple[int, float | None]]:
    """入った時刻(UTC の時)ごとの (回数, PF)。"""
    out = {}
    for h in range(24):
        sub = [t["net"] for t in trades if (t["entry_open_ms"] // 3_600_000) % 24 == h]
        out[h] = (len(sub), profit_factor(sub))
    return out


# ---------------------------------------------------------------- 探索(7.8)

def param_grid(search: dict) -> list[dict]:
    keys = ["rci_slope_min", "rci14_max", "macd_lookback", "stop_atr", "take_atr"]
    return [dict(zip(keys, vals)) for vals in itertools.product(*(search[k] for k in keys))]


def search_params(p: Prepared, exit_mode: str, cfg: dict, costs: dict) -> tuple[dict, bool]:
    """前半データだけで全組を流し、(選んだ組, 探索で選べたか) を返す。

    候補は前半の取引回数がその足の最低回数(8.5)に届く組だけ。その中で PF が最も高い組、
    同じなら取引回数が多い組、それも同じなら探索表の先の組。候補が無ければ初期値。
    """
    fixed, capital = cfg["fixed"], cfg["backtest"]["capital"]
    best, best_key = None, None
    masks: dict[tuple, np.ndarray] = {}
    for prm in param_grid(cfg["search"]):
        mk = (prm["rci_slope_min"], prm["rci14_max"], prm["macd_lookback"])
        if mk not in masks:
            masks[mk] = strategy.entry_mask(p.features, prm, p.us_allowed)
        tr = simulate(p, masks[mk], prm, exit_mode, p.start_ms, p.split_ms, costs, fixed, capital)
        if len(tr) < MIN_TRADES[p.tf]:
            continue
        pf = profit_factor([t["net"] for t in tr])
        if pf is None:
            continue
        key = (pf, len(tr))
        if best_key is None or key > best_key:
            best, best_key = prm, key
    if best is None:
        return dict(cfg["initial"]), False
    return dict(best), True


# ---------------------------------------------------------------- 1設定の検証

def evaluate(p: Prepared, exit_mode: str, cfg: dict, costs: dict, mode: str) -> dict:
    capital = cfg["backtest"]["capital"]
    row = {"tf": p.tf, "coin": p.coin, "exit": exit_mode, "mode": mode,
           "start_ms": p.start_ms, "split_ms": p.split_ms, "end_ms": p.end_ms,
           "full_days": p.full_days, "second_days": (p.end_ms - p.split_ms) / DAY_MS,
           "excluded_days": p.excluded_days, "_prepared": p}
    if mode == "formal":
        prm, searched = search_params(p, exit_mode, cfg, costs)
        first = run_one(p, prm, exit_mode, p.start_ms, p.split_ms, cfg, costs)
        second = run_one(p, prm, exit_mode, p.split_ms, p.end_ms, cfg, costs)
        row.update(params=prm, searched=searched, first=stats(first, capital), eval=stats(second, capital),
                   eval_trades=second)
    else:
        prm = dict(cfg["initial"])
        trades = run_one(p, prm, exit_mode, p.start_ms, p.end_ms, cfg, costs)
        row.update(params=prm, searched=False, first=None, eval=stats(trades, capital), eval_trades=trades)
    row["hours"] = hour_table(row["eval_trades"]) if not p.hip3 else None
    return row


def frequency_row(p: Prepared, exit_mode: str, cfg: dict, costs: dict) -> dict:
    """8.2 中間報告: 1日あたりの合図回数・約定回数と、各条件を1つ外した時の合図回数(初期値)。"""
    prm = dict(cfg["initial"])
    days = max(p.full_days, 1)
    signals = int(strategy.entry_mask(p.features, prm, p.us_allowed).sum())
    trades = run_one(p, prm, exit_mode, p.start_ms, p.end_ms, cfg, costs)
    removed = {c: int(strategy.entry_mask(p.features, prm, p.us_allowed, frozenset([c])).sum())
               for c in strategy.CONDITIONS}
    need = MIN_TRADES[p.tf] / MIN_SECOND_HALF_DAYS
    return {"tf": p.tf, "coin": p.coin, "exit": exit_mode, "full_days": p.full_days,
            "excluded_days": p.excluded_days, "signals": signals, "signals_per_day": signals / days,
            "fills": len(trades), "fills_per_day": len(trades) / days, "need_per_day": need,
            "removed": removed}


# ---------------------------------------------------------------- 判定(8.5)

def judge(rows: list[dict]) -> None:
    """予備判定と正式判定。基準7は同じ銘柄・同じ出口方式の隣の足を見るので、全行がそろってから。"""
    index = {(r["tf"], r["coin"], r["exit"]): r for r in rows}
    for r in rows:
        formal = r.get("mode", "formal") == "formal"
        s = r["eval"]
        tf = r["tf"]
        fails = []
        if formal and r["second_days"] < MIN_SECOND_HALF_DAYS:
            r["verdict"], r["fails"] = "データ待ち", [f"後半が{r['second_days']:.1f}日(90日以上が必要)"]
            continue
        if s["trades"] < MIN_TRADES[tf]:
            r["verdict"] = "判定不能"
            r["fails"] = [f"基準1 取引回数 {s['trades']}回({MIN_TRADES[tf]}回以上が必要)"]
            continue
        pf = s["pf"] if s["pf"] is not None else 0.0
        if pf < MIN_PF[tf]:
            fails.append(f"基準2 PF {pf:.2f}({MIN_PF[tf]:.2f}以上が必要)")
        if formal:
            pf1 = r["first"]["pf"] if r["first"]["pf"] is not None else 0.0
            if pf1 - pf > MAX_PF_DROP:
                fails.append(f"基準3 前半との差 {pf1 - pf:.2f}(0.5以内が必要)")
        if s["max_dd"] > MAX_DRAWDOWN:
            fails.append(f"基準4 最大の落ち込み {s['max_dd'] * 100:.1f}%(10%以内が必要)")
        ms = s["month_share"]
        if ms is None or ms > MAX_MONTH_SHARE:
            fails.append("基準5 月別利益の偏り " + ("総利益が0以下" if ms is None else f"{ms * 100:.1f}%(40%以下が必要)"))
        if s["avg_net"] is None or s["avg_net"] < MIN_AVG_NET:
            fails.append(f"基準6 平均純利益 {s['avg_net'] or 0:.2f}ドル(1.0ドル以上が必要)")
        r["fails"] = fails
    for r in rows:
        if "verdict" in r:
            continue
        for ntf in NEIGHBORS[r["tf"]]:
            nb = index.get((ntf, r["coin"], r["exit"]))
            if nb is None or nb.get("verdict") in ("データ待ち", "判定不能"):
                continue
            npf = nb["eval"]["pf"] if nb["eval"]["pf"] is not None else 0.0
            if npf < MIN_NEIGHBOR_PF:
                r["fails"].append(f"基準7 隣の足 {ntf} の PF {npf:.2f}(0.9以上が必要)")
        r["verdict"] = "合格" if not r["fails"] else "不合格"
    for r in rows:
        if r.get("mode") == "preliminary" and r["verdict"] in ("合格", "不合格"):
            r["verdict"] = "予備" + r["verdict"]


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


def _part_text(d: dict) -> str:
    return " / ".join(f"{k} {n}回 PF{_f(pf)}" for k, (n, pf) in d.items())


CSV_HEADER = ["足", "銘柄", "出口方式", "選んだパラメータ", "前半期間", "後半期間",
              "取引回数_前半", "取引回数_後半", "勝率_前半", "勝率_後半", "PF_前半", "PF_後半",
              "平均純利益_後半", "最大の落ち込み_後半", "月別利益の最大比率_後半",
              "印の内訳_後半", "パラボ陽の割合_後半", "パラボ別PF_後半", "除外日数", "判定", "否決理由"]


def table_rows(rows: list[dict]) -> list[list[str]]:
    """判定表の行。予備判定は「後半」の列に全データの数字を入れ、前半の列は空。"""
    out = []
    for r in rows:
        s1, s2 = r["first"], r["eval"]
        formal = r["mode"] == "formal"
        prm = _params_text(r["params"]) if r["searched"] else f"初期値({_params_text(r['params'])})"
        bull_share = s2["psar"]["陽"][0] / s2["trades"] if s2["trades"] else None
        out.append([r["tf"], r["coin"], r["exit"], prm,
                    f"{_d(r['start_ms'])}〜{_d(r['split_ms'])}" if formal else "",
                    f"{_d(r['split_ms'])}〜{_d(r['end_ms'])}" if formal else f"{_d(r['start_ms'])}〜{_d(r['end_ms'])}",
                    str(s1["trades"]) if s1 else "", str(s2["trades"]),
                    _f(s1["win_rate"], pct=True) if s1 else "", _f(s2["win_rate"], pct=True),
                    _f(s1["pf"]) if s1 else "", _f(s2["pf"]), _f(s2["avg_net"]), _f(s2["max_dd"], pct=True),
                    _f(s2["month_share"], pct=True), _part_text(s2["mark"]),
                    _f(bull_share, pct=True), _part_text(s2["psar"]),
                    str(r["excluded_days"]), r["verdict"],
                    " / ".join(r["fails"]) if r["verdict"] not in ("合格", "予備合格") else ""])
    return out


def report_name(date: str, mode: str, stage: int | None, single: str | None = None) -> str:
    """8.7: judgement_<日付>_<mode>。第2段階と1設定だけの実行は、第1段階の表を上書きしないよう後ろに付ける。"""
    name = f"judgement_{date}_{mode}"
    if single:
        return f"{name}_{single}"
    return name if (stage or 1) == 1 else f"{name}_stage{stage}"


def write_reports(rows: list[dict], mode: str, stage: int | None, out_dir: Path, date: str,
                  single: str | None = None) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = report_name(date, mode, stage, single)
    md, cs = out_dir / f"{name}.md", out_dir / f"{name}.csv"
    if mode == "frequency":
        header, body = frequency_table(rows)
        summary = f"データ: 丸1日そろった日 {max((r['full_days'] for r in rows), default=0)}日。判定はしない。"
    else:
        header, body = CSV_HEADER, table_rows(rows)
        kinds = (("予備合格", "予備不合格", "判定不能") if mode == "preliminary"
                 else ("合格", "不合格", "判定不能", "データ待ち"))
        summary = "要約: " + " / ".join(f"{k} {sum(1 for r in rows if r['verdict'] == k)}" for k in kinds)
    with open(cs, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)
    title = f"# {MODE_NAMES[mode]} {date}" + (f"(第{stage}段階)" if stage else "")
    lines = [title, "", summary, "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(c.replace("|", "/") for c in row) + " |" for row in body]
    crypto = [r for r in rows if r.get("hours") is not None]
    if crypto and mode != "frequency":
        lines += ["", "## 時間帯別成績(仮想通貨、入った時刻 UTC)", "",
                  "PF(回数)。取引が無い時間帯は「-」。正式判定は後半、予備判定は全データ。", "",
                  "| 足 | 銘柄 | 出口 | " + " | ".join(f"{h}時" for h in range(24)) + " |",
                  "|" + "---|" * 27]
        for r in crypto:
            cells = [f"{_f(pf)}({n})" if n else "-" for n, pf in (r["hours"][h] for h in range(24))]
            lines.append(f"| {r['tf']} | {r['coin']} | {r['exit']} | " + " | ".join(cells) + " |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md, cs


def frequency_table(rows: list[dict]) -> tuple[list[str], list[list[str]]]:
    header = ["足", "銘柄", "出口方式", "丸1日の日数", "合図回数", "合図/日", "約定回数", "約定/日",
              "必要な約定/日(最低回数÷90日)"] + [f"{strategy.CONDITION_NAMES[c]}を外した合図" for c in strategy.CONDITIONS]
    body = [[r["tf"], r["coin"], r["exit"], str(r["full_days"]), str(r["signals"]), f"{r['signals_per_day']:.2f}",
             str(r["fills"]), f"{r['fills_per_day']:.2f}", f"{r['need_per_day']:.2f}"]
            + [str(r["removed"][c]) for c in strategy.CONDITIONS] for r in rows]
    return header, body


def write_memo(r: dict, cfg: dict, costs: dict, out_dir: Path) -> Path:
    """8.6 否決メモ(正式判定の不合格・判定不能の設定ごとに1ファイル)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    p: Prepared = r["_prepared"]
    tr = r["eval_trades"]
    s2 = r["eval"]
    lines = [f"# 否決メモ: {r['tf']} / {r['coin']} / 出口方式{r['exit']}", "",
             f"判定: {r['verdict']}", "",
             "## 1. 設定と選んだパラメータ", "",
             f"- {_params_text(r['params'])}" + ("" if r["searched"] else "(候補なしのため初期値)"),
             f"- 前半 {_d(r['start_ms'])}〜{_d(r['split_ms'])}、後半 {_d(r['split_ms'])}〜{_d(r['end_ms'])}(後半 {r['second_days']:.1f}日)",
             "", "## 2. 引っかかった基準と数字", ""]
    lines += [f"- {x}" for x in r["fails"]]
    lines += ["", "## 3. 負けの内訳(後半)", "", "| 出口 | 回数 | 合計損益(ドル) |", "|---|---|---|"]
    for reason, name in REASON_NAMES.items():
        sub = [t["net"] for t in tr if t["reason"] == reason]
        lines.append(f"| {name} | {len(sub)} | {sum(sub):.2f} |")
    lines += ["", "## 4. コスト内訳(後半)", "",
              f"- 手数料合計: {sum(t['fees'] for t in tr):.2f} ドル",
              f"- 滑り合計: {sum(t['slippage'] for t in tr):.2f} ドル",
              f"- 資金調達料合計: {sum(t['funding'] for t in tr):.2f} ドル",
              f"- コストを0にした時の PF: {_f(s2['pf_no_cost'])}(実際の PF {_f(s2['pf'])})",
              "", "## 5. 入り口の条件を1つ外した時の合図回数(後半)", ""]
    close_ms = p.features["close_ms"].to_numpy()
    in_second = (close_ms >= p.split_ms) & (close_ms < p.end_ms)
    base = strategy.entry_mask(p.features, r["params"], p.us_allowed)
    lines.append(f"- すべて課した時: {int((base & in_second).sum())}回")
    for name in strategy.CONDITIONS:
        m = strategy.entry_mask(p.features, r["params"], p.us_allowed, frozenset([name]))
        lines.append(f"- {strategy.CONDITION_NAMES[name]} を外す: {int((m & in_second).sum())}回")
    lines += ["", "## 6. 所見", "", _remark(r)]
    path = out_dir / f"reject_{r['tf']}_{r['coin'].replace(':', '_')}_{r['exit']}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _remark(r: dict) -> str:
    s2 = r["eval"]
    if r["verdict"] == "判定不能":
        return f"取引回数不足(後半 {s2['trades']}回、必要 {MIN_TRADES[r['tf']]}回)。"
    pf, pf0 = s2["pf"], s2["pf_no_cost"]
    if pf is not None and pf0 is not None and pf < 1.0 <= pf0:
        return "コスト負け(コスト前は PF 1.0以上)。"
    if pf0 is not None and pf0 < 1.0:
        return "ロジック負け(コスト前から PF 1.0未満)。"
    return "その他の基準。"


# ---------------------------------------------------------------- 実行

def run(mode: str = "formal", stage: int | None = None, coin: str | None = None, tf: str | None = None,
        exit_mode: str | None = None, db_path: Path = config.MARKET_DB, out_dir: Path = config.REPORTS_DIR,
        memo_dir: Path = config.MEMOS_DIR, date: str | None = None) -> list[dict]:
    cfg, costs, holidays = config.strategy(), config.costs(), config.us_holidays()
    if coin:
        coins = [config.coin_info(coin)]
        stage = coins[0]["stage"]
    else:
        coins = config.coins(stage or 1)
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
                print(f"{MODE_NAMES[mode]} {t} {info['coin']} 出口{m}")
                rows.append(frequency_row(p, m, cfg, costs) if mode == "frequency"
                            else evaluate(p, m, cfg, costs, mode))
    con.close()
    if mode != "frequency":
        judge(rows)
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    single = None
    if coin or tf or exit_mode:
        single = "_".join(x.replace(":", "_") for x in (tf or "all", coin or "all", exit_mode or "all"))
    md, _ = write_reports(rows, mode, stage, out_dir, date, single)
    if mode == "formal":
        for r in rows:
            if r["verdict"] in ("不合格", "判定不能"):
                write_memo(r, cfg, costs, memo_dir)
    print(f"判定表: {md}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="検証(中間報告・予備判定・正式判定)")
    ap.add_argument("--stage", type=int, choices=[1, 2])
    ap.add_argument("--mode", choices=MODES, default="formal")
    ap.add_argument("--coin")
    ap.add_argument("--tf", choices=config.TIMEFRAMES)
    ap.add_argument("--exit", choices=EXIT_MODES)
    a = ap.parse_args()
    run(mode=a.mode, stage=a.stage, coin=a.coin, tf=a.tf, exit_mode=a.exit)


if __name__ == "__main__":
    main()
