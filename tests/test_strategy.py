"""strategy.py の単体試験(仕様書 13.1: キリのいい数字・時間帯・約定判定・安全装置)。"""
import numpy as np
import pandas as pd
import pytest

from kaikai_bot import strategy as st

M = 60_000
H = 60 * M
T0 = 1_767_225_600_000
COSTS = {"taker_fee": 0.00045, "maker_fee": 0.00015, "hip3_fee_multiplier": 2.0,
         "slippage_stop": 0.0002, "slippage_entry": 0.0, "funding_per_hour": 0.00001}


# ---------------------------------------------------------------- キリのいい数字(6.6 の6例)

@pytest.mark.parametrize("price, step", [(67_420.0, 1000.0), (2_480.0, 100.0), (179.0, 10.0), (185.0, 10.0),
                                         (24_800.0, 1000.0)])
def test_round_step(price, step):
    assert st.round_step(price) == pytest.approx(step)


@pytest.mark.parametrize("price, wall, nxt", [
    (67_950.0, True, 68_000.0),    # BTC: 差 50 ≤ 100
    (67_420.0, False, 68_000.0),   # BTC: 差 580 > 100
    (179.2, True, 180.0),          # SOL: 差 0.8 ≤ 1
    (185.3, False, 190.0),         # NVDA: 差 4.7 > 1
    (68_000.0, True, 68_000.0),    # ちょうど倍数なら R = p
    (2_490.0, True, 2_500.0),      # ETH: 差 10 ≤ 10(等号も直下)
])
def test_round_wall(price, wall, nxt):
    assert st.next_round_number(price) == pytest.approx(nxt)
    assert st.below_round_wall(price) is wall


def test_round_wall_array():
    assert list(st.below_round_wall(np.array([67_950.0, 67_420.0, np.nan]))) == [True, False, False]


# ---------------------------------------------------------------- 時間帯(米国株)

def _utc(s):
    return int(pd.Timestamp(s, tz="UTC").value // 1_000_000)


def test_us_window_summer_time():
    # 2026-07-15(水)は夏時間: NY 9:50 = UTC 13:50、15:30 = UTC 19:30
    ts = np.array([_utc("2026-07-15 13:49"), _utc("2026-07-15 13:50"), _utc("2026-07-15 19:29"),
                   _utc("2026-07-15 19:30")])
    allowed, _ = st.us_time_table(ts, "1m", set())
    assert list(allowed) == [False, True, True, False]


def test_us_window_winter_time():
    # 2026-12-02(水)は冬時間: NY 9:50 = UTC 14:50
    ts = np.array([_utc("2026-12-02 13:50"), _utc("2026-12-02 14:49"), _utc("2026-12-02 14:50"),
                   _utc("2026-12-02 20:29"), _utc("2026-12-02 20:30")])
    allowed, _ = st.us_time_table(ts, "1m", set())
    assert list(allowed) == [False, False, True, True, False]


def test_us_holiday_and_weekend_closed():
    hol = {"2026-11-26"}
    ts = np.array([_utc("2026-11-26 16:00"),     # 感謝祭(冬時間 NY 11:00)
                   _utc("2026-11-28 16:00"),     # 土曜
                   _utc("2026-11-25 16:00")])    # 平日
    allowed, _ = st.us_time_table(ts, "1m", hol)
    assert list(allowed) == [False, False, True]


def test_us_force_exit_1555():
    # 夏時間 NY 15:54 確定の1分足は持ってよい、15:55 確定の足の終値で手仕舞い
    ts = np.array([_utc("2026-07-15 19:54"), _utc("2026-07-15 19:55")])
    _, force = st.us_time_table(ts, "1m", set())
    assert list(force) == [False, True]
    # 1時間足: NY 15:00 確定の足で手仕舞い(次の足は 16:00 確定で 15:55 を過ぎる)。15:00 確定では新規に入らない
    ts = np.array([_utc("2026-07-15 18:00"), _utc("2026-07-15 19:00")])
    allowed, force = st.us_time_table(ts, "1h", set())
    assert list(force) == [False, True]
    assert list(allowed) == [True, False]


# ---------------------------------------------------------------- 約定判定と出口

def test_fill_rule():
    assert st.fills(100.0, 99.9)
    assert st.fills(100.0, 100.0)       # 等号でも約定
    assert not st.fills(100.0, 100.01)


def _pos(max_bars=15):
    p = st.open_position(100.0, 2.0, {"stop_atr": 1.0, "take_atr": 2.0}, max_bars, 0, 0)
    assert (p.stop, p.take) == (98.0, 104.0)
    return p


def test_stop_and_take_same_bar_stop_first():
    p = _pos()
    p.bars_held = 2
    res = st.exit_on_bar(p, high=105.0, low=97.0, close=101.0, slope9=5.0, exit_mode="A",
                         is_entry_bar=False, force_exit=False, slippage_stop=0.0002)
    assert res[0] == "stop"
    assert res[1] == pytest.approx(98.0 * (1 - 0.0002))


def test_take_and_time_and_rci_exit():
    p = _pos(max_bars=3)
    p.bars_held = 2
    assert st.exit_on_bar(p, 104.0, 99.0, 103.0, 5.0, "A", False, False, 0.0002)[:2] == ("take", 104.0)
    assert st.exit_on_bar(p, 103.0, 99.0, 102.0, -1.0, "A", False, False, 0.0002) is None
    assert st.exit_on_bar(p, 103.0, 99.0, 102.0, -1.0, "B", False, False, 0.0002)[:2] == ("rci_exit", 102.0)
    p.bars_held = 3
    assert st.exit_on_bar(p, 103.0, 99.0, 102.0, 1.0, "A", False, False, 0.0002)[:2] == ("time", 102.0)


def test_entry_bar_does_not_take_profit():
    p = _pos()
    p.bars_held = 1
    assert st.exit_on_bar(p, 105.0, 99.0, 101.0, 1.0, "A", True, False, 0.0002) is None
    assert st.exit_on_bar(p, 105.0, 97.0, 101.0, 1.0, "A", True, False, 0.0002)[0] == "stop"


def test_forced_exit_before_gap():
    p = _pos()
    p.bars_held = 1
    assert st.exit_on_bar(p, 101.0, 99.0, 100.5, 1.0, "A", False, True, 0.0002)[:2] == ("forced", 100.5)


def test_settle_costs():
    p = _pos()
    p.entry_open_ms = 0
    t = st.settle(p, "take", 104.0, 104.0, 2 * H, COSTS, hip3=False, capital=1000.0)
    qty = 1000.0 / 100.0
    fees = 1000 * 0.00015 + qty * 104.0 * 0.00015
    assert t["fees"] == pytest.approx(fees)
    assert t["funding"] == pytest.approx(1000 * 0.00001 * 2)
    assert t["net"] == pytest.approx(qty * 4.0 - fees - 0.02)
    h = st.settle(p, "take", 104.0, 104.0, 2 * H, COSTS, hip3=True, capital=1000.0)
    assert h["fees"] == pytest.approx(fees * 2)


# ---------------------------------------------------------------- 安全装置

def _trade(net, day, minute=0):
    return {"net": net, "exit_close_ms": day * 86_400_000 + minute * M}


def test_daily_loss_limit():
    s = st.Safety(0.03, 5, 1000.0)
    st.record(s, _trade(-20.0, 0, 10))
    assert st.can_open(s, 20 * M)
    st.record(s, _trade(-10.0, 0, 30))         # 合計 -30 = 3% に達した
    assert not st.can_open(s, 40 * M)
    assert st.can_open(s, 86_400_000)           # 次の UTC の日は入ってよい


def test_losing_streak_blocks_until_next_utc_day():
    s = st.Safety(0.5, 5, 1000.0)
    for i in range(4):
        st.record(s, _trade(-1.0, 0, i))
    assert st.can_open(s, 10 * M)
    st.record(s, _trade(-1.0, 0, 5))            # 5連敗
    assert not st.can_open(s, 10 * M)
    assert not st.can_open(s, 86_400_000 - 1)
    assert st.can_open(s, 86_400_000)           # UTC 0時で解除
    assert s.losing_streak == 0


def test_win_resets_streak():
    s = st.Safety(0.5, 5, 1000.0)
    for i in range(4):
        st.record(s, _trade(-1.0, 0, i))
    st.record(s, _trade(2.0, 0, 5))
    assert s.losing_streak == 0


def _feat(n=40, **over):
    """入り口の条件がすべてそろった作り物の指標の表。over で列を上書きする。"""
    f = pd.DataFrame({
        "close": np.full(n, 103.0), "sma5": np.full(n, 100.0), "sma75": np.full(n, 100.0),
        "upper_sma25_up": True, "psar_bull": False, "psar_bear": True, "gc_neg": True,
        "slope9": 30.0, "rci14_prev": -60.0, "rci14_turn_up": True,
        "atr": 1.0, "gap_after": False, "since_gap": np.arange(11, 11 + n),
    })
    for k, v in over.items():
        f[k] = v
    return f


PRM2 = {"rci_slope_min": 20, "rci14_max": -50, "macd_lookback": 3}


def test_env_filter():
    assert st.entry_mask(_feat(), PRM2).all()
    assert not st.entry_mask(_feat(close=99.0, sma5=98.0), PRM2).any()        # 終値が75本線の下
    assert not st.entry_mask(_feat(upper_sma25_up=False), PRM2).any()         # 上の足の25本線が上向きでない


def test_upper_bar_missing_or_down():
    """上の足が欠けている時・下向きの時は環2を満たさない。"""
    M5 = 5 * M
    up = pd.DataFrame({"ts_open": T0 + np.arange(40) * M5, "close": 100.0 + np.arange(40)})
    close_ms = np.array([T0 + 40 * M5, T0 + 40 * M5 + M5 * 3])   # 直後 / 上の足が3本欠けた先
    got = st._upper_rising(close_ms, up, "5m")
    assert list(got) == [True, False]
    down = up.assign(close=200.0 - np.arange(40))
    assert not st._upper_rising(close_ms[:1], down, "5m").any()


def test_entry_four_patterns():
    """必須3つ+印Aのみ / 印Bのみ / 両方 / どちらも無し。"""
    only_a = _feat(rci14_turn_up=False)
    only_b = _feat(gc_neg=False)
    both = _feat()
    none = _feat(gc_neg=False, rci14_turn_up=False)
    assert st.entry_mask(only_a, PRM2).all()
    assert st.entry_mask(only_b, PRM2).all()
    assert st.entry_mask(both, PRM2).all()
    assert not st.entry_mask(none, PRM2).any()
    assert set(st.entry_tags(only_a, PRM2)[0]) == {"A"}
    assert set(st.entry_tags(only_b, PRM2)[0]) == {"B"}
    assert set(st.entry_tags(both, PRM2)[0]) == {"両方"}
    assert set(st.entry_tags(both, PRM2)[1]) == {"陰"}                     # パラボリック陰でも入る(記録だけ)


def test_required_conditions():
    assert not st.entry_mask(_feat(slope9=19.9), PRM2).any()                # 必1
    assert not st.entry_mask(_feat(close=99.5, sma75=90.0), PRM2).any()     # 必2(5本線 100 の下)
    assert not st.entry_mask(_feat(close=109.95, sma5=100.0), PRM2).any()   # 必3(110 の壁まで 0.05 ≤ 1)


def test_mark_b_uses_previous_bar_depth():
    """印B: 1本前の RCI(14) が −50 以下で、そこから上向きに転じた。"""
    assert st.entry_mask(_feat(gc_neg=False, rci14_prev=-50.0), PRM2).all()
    assert not st.entry_mask(_feat(gc_neg=False, rci14_prev=-49.0), PRM2).any()


def test_disabled_condition_counts():
    f = _feat(slope9=0.0)
    assert not st.entry_mask(f, PRM2).any()
    assert st.entry_mask(f, PRM2, disabled=frozenset(["req1"])).all()


def test_cooldown_after_gap_in_mask():
    """欠損明け10本は入り口の条件を満たしても入らない。"""
    n = 40
    f = _feat(n, since_gap=np.r_[np.arange(1, 21), np.arange(1, 21)])
    f.loc[19, "gap_after"] = True
    m = st.entry_mask(f, PRM2)
    assert not m[:10].any() and m[10:19].all()
    assert not m[19]                              # 次の足が欠けている
    assert not m[20:30].any() and m[30:].all()
