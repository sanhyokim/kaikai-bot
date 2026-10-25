"""検証エンジンの試験(仕様書 13.1 再現性、13.2 結合試験、8.5 判定)。"""
import math

import numpy as np
import pandas as pd
import pytest

from kaikai_bot import backtest as bt, collect, config

M = 60_000
H = 60 * M
T0 = 1_767_225_600_000
COSTS = {"taker_fee": 0.00045, "maker_fee": 0.00015, "hip3_fee_multiplier": 2.0,
         "slippage_stop": 0.0002, "slippage_entry": 0.0, "funding_per_hour": 0.00001}
FIXED = {"max_bars": {"1m": 15, "5m": 12, "15m": 8, "1h": 6}, "round_number_margin": 0.005,
         "daily_loss_limit": 0.03, "max_consecutive_losses": 5}
PRM = {"rci_slope_min": 20, "rci14_max": -50, "macd_lookback": 3, "stop_atr": 1.0, "take_atr": 2.0}


def artificial(n=100):
    """値段 100 で横ばいの1分足 100本に、利確・損切りの値動きを1回ずつ入れる。ATR は 1.0 に固定。"""
    ts = T0 + np.arange(n) * M
    f = pd.DataFrame({"ts_open": ts, "close_ms": ts + M, "open": 100.0, "high": 100.2, "low": 99.8,
                      "close": 100.0, "slope9": 1.0, "atr": 1.0})
    f.loc[12, "high"] = 102.5       # 1回目: 11本目で約定 → 12本目で利確 102
    f.loc[32, "low"] = 98.5         # 2回目: 31本目で約定 → 32本目で損切り 99
    mask = np.zeros(n, dtype=bool)
    mask[[10, 30, 50]] = True        # 3回目: 51本目で約定 → 15本持って時間切れ
    p = bt.Prepared("BTC", "1m", False, f, None, np.zeros(n, dtype=bool), T0, T0 + 60 * M, T0 + n * M, 0)
    return p, mask


def test_three_trade_types_with_hand_calc():
    p, mask = artificial()
    trades = bt.simulate(p, mask, PRM, "A", p.start_ms, p.end_ms, COSTS, FIXED, 1000.0)
    assert [t["reason"] for t in trades] == ["take", "stop", "time"]
    qty = 1000.0 / 100.0
    entry_fee = 1000 * 0.00015
    # 1回目 利確: 11本目の始まり〜12本目の終わり = 2時間分ではなく2分
    take = trades[0]
    exit_fee = qty * 102.0 * 0.00015
    funding = 1000 * 0.00001 * (2 * M / H)
    assert take["exit_price"] == 102.0
    assert take["net"] == pytest.approx(qty * 2.0 - entry_fee - exit_fee - funding)
    # 2回目 損切り: 99 から滑り 0.02% 下で約定、taker
    stop = trades[1]
    px = 99.0 * (1 - 0.0002)
    assert stop["exit_price"] == pytest.approx(px)
    assert stop["slippage"] == pytest.approx(qty * (99.0 - px))
    assert stop["net"] == pytest.approx(qty * (px - 100.0) - entry_fee - qty * px * 0.00045
                                        - 1000 * 0.00001 * (2 * M / H))
    assert stop["gross_no_cost"] == pytest.approx(qty * -1.0)
    # 3回目 時間切れ: 51本目から15本目(65本目)の終値 100、taker
    tm = trades[2]
    assert tm["bars_held"] == 15
    assert tm["exit_close_ms"] == T0 + 66 * M
    assert tm["net"] == pytest.approx(-entry_fee - qty * 100.0 * 0.00045 - 1000 * 0.00001 * (15 * M / H))


def test_unfilled_order_is_cancelled():
    p, mask = artificial()
    p.features.loc[11, "low"] = 100.01        # 次の足が指値に届かない
    trades = bt.simulate(p, mask, PRM, "A", p.start_ms, p.end_ms, COSTS, FIXED, 1000.0)
    assert [t["reason"] for t in trades] == ["stop", "time"]


def test_no_new_entry_while_holding():
    p, mask = artificial()
    mask[55] = True                           # 3回目の保有中の合図は無視される
    trades = bt.simulate(p, mask, PRM, "A", p.start_ms, p.end_ms, COSTS, FIXED, 1000.0)
    assert len(trades) == 3


def test_half_split_by_signal_time():
    p, mask = artificial()
    split = T0 + 45 * M                       # 3回目の合図(50本目の確定 = 51分)は後半
    first = bt.simulate(p, mask, PRM, "A", p.start_ms, split, COSTS, FIXED, 1000.0)
    second = bt.simulate(p, mask, PRM, "A", split, p.end_ms, COSTS, FIXED, 1000.0)
    assert len(first) == 2 and len(second) == 1


def test_metrics():
    assert bt.profit_factor([3.0, -1.0, -2.0]) == pytest.approx(1.0)
    assert bt.profit_factor([1.0]) == math.inf
    assert bt.profit_factor([]) is None
    assert bt.max_drawdown([100.0, -110.0, 50.0], 1000.0) == pytest.approx(110.0 / 1100.0)
    trades = [{"net": 30.0, "exit_close_ms": T0}, {"net": 10.0, "exit_close_ms": T0 + 40 * 86_400_000}]
    assert bt.month_share(trades) == pytest.approx(0.75)


def _row(tf, verdict_inputs):
    days, trades, pf1, pf2, dd, ms, avg = verdict_inputs
    return {"tf": tf, "coin": "BTC", "exit": "A", "second_days": days,
            "first": {"pf": pf1}, "second": {"trades": trades, "pf": pf2, "max_dd": dd,
                                              "month_share": ms, "avg_net": avg}}


def test_judge_rules():
    good = (100, 400, 1.6, 1.5, 0.05, 0.3, 1.5)
    rows = [_row("1m", good), _row("5m", (100, 400, 1.6, 0.85, 0.05, 0.3, 1.5)),
            _row("15m", (100, 299, 1.6, 2.0, 0.05, 0.3, 1.5)), _row("1h", (80, 400, 1.6, 1.5, 0.05, 0.3, 1.5))]
    bt.judge(rows)
    v = {r["tf"]: r for r in rows}
    assert v["1h"]["verdict"] == "データ待ち"
    assert v["15m"]["verdict"] == "判定不能"
    assert v["5m"]["verdict"] == "不合格"
    assert v["1m"]["verdict"] == "不合格"            # 隣の 5分の後半 PF 0.85 < 0.9
    assert any("基準7" in x for x in v["1m"]["fails"])


def test_judge_pass_and_each_criterion():
    rows = [_row("1m", (100, 400, 1.6, 1.5, 0.05, 0.3, 1.5))]
    bt.judge(rows)
    assert rows[0]["verdict"] == "合格"            # 隣(5分)が無いので基準7は適用しない
    for bad, key in [((100, 400, 2.1, 1.5, 0.05, 0.3, 1.5), "基準3"), ((100, 400, 1.6, 1.29, 0.05, 0.3, 1.5), "基準2"),
                     ((100, 400, 1.6, 1.5, 0.11, 0.3, 1.5), "基準4"), ((100, 400, 1.6, 1.5, 0.05, 0.41, 1.5), "基準5"),
                     ((100, 400, 1.6, 1.5, 0.05, 0.3, 0.99), "基準6")]:
        rows = [_row("1m", bad)]
        bt.judge(rows)
        assert rows[0]["verdict"] == "不合格" and any(key in x for x in rows[0]["fails"])


def _random_db(path, days=3, seed=5):
    rng = np.random.default_rng(seed)
    n = days * 1440
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0015, n)))
    con = collect.connect(path)
    rows = [{"ts_open": T0 + i * M, "open": close[i - 1] if i else close[0], "high": close[i] * 1.001,
             "low": close[i] * 0.999, "close": close[i], "volume": 1.0, "trades": 1} for i in range(n)]
    collect.upsert(con, "BTC", rows, 0)
    con.close()


def test_reproducible(tmp_path, monkeypatch):
    """同じデータ・同じ設定で2回流して、小数点以下6桁まで同じ結果(8.7)。"""
    _random_db(tmp_path / "m.db")
    small = {"rci_slope_min": [10, 20], "rci14_max": [-40], "macd_lookback": [3, 5],
             "stop_atr": [1.0], "take_atr": [2.0]}
    real = config.strategy

    def cfg():
        c = real()
        c["search"] = small
        return c
    monkeypatch.setattr(config, "strategy", cfg)
    outs = []
    for k in range(2):
        rows = bt.run(coin="BTC", tf="1m", db_path=tmp_path / "m.db", out_dir=tmp_path / f"r{k}",
                      memo_dir=tmp_path / f"m{k}", date="test")
        outs.append([(r["params"], r["second"]["trades"], round(r["second"]["pf"] or 0, 6),
                      [round(t["net"], 6) for t in r["second_trades"]]) for r in rows])
        assert (tmp_path / f"r{k}" / "judgement_test.md").exists()
    assert outs[0] == outs[1]
    assert (tmp_path / "r0" / "judgement_test.csv").read_bytes() == (tmp_path / "r1" / "judgement_test.csv").read_bytes()
