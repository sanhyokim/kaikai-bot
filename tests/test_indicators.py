"""指標の単体試験(仕様書 6.6)。既知の値と小数点以下4桁で一致することを確かめる。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kaikai_bot import indicators as ind

DATA = Path(__file__).parent / "data"
DECIMALS = 4


@pytest.fixture(scope="module")
def known():
    return pd.read_csv(DATA / "known_values.csv")


def assert_same(actual, expected):
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    assert np.array_equal(np.isnan(a), np.isnan(e)), "計算不能の足の位置が違う"
    ok = ~np.isnan(e)
    np.testing.assert_array_equal(np.round(a[ok], DECIMALS), np.round(e[ok], DECIMALS))


@pytest.mark.parametrize("n", [5, 25, 75])
def test_sma_matches_known(known, n):
    assert_same(ind.sma(known["close"], n), known[f"sma{n}"])


def test_sma_needs_n_bars():
    s = ind.sma(pd.Series([1.0, 2.0, 3.0, 4.0]), 5)
    assert s.isna().all()


def test_sma_hand_calc():
    s = ind.sma(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 5)
    assert s.iloc[4] == 3.0 and s.iloc[5] == 4.0


def test_rising_needs_10_bars_back():
    x = pd.Series(np.arange(12, dtype=float))
    r = ind.rising(x)
    assert not r.iloc[:10].any()
    assert r.iloc[10] and r.iloc[11]
    assert not ind.rising(pd.Series([5.0] * 12)).any()   # 同じ値は上向きではない


def test_macd_matches_known(known):
    m = ind.macd(known["close"])
    assert_same(m["macd"], known["macd"])
    assert_same(m["signal"], known["signal"])


def test_macd_gc_negative_in_known(known):
    m = ind.macd(known["close"])
    line, sig = known["macd"], known["signal"]
    expected = (line.shift(1) <= sig.shift(1)) & (line > sig) & (line < 0)
    assert (m["gc_neg"] == expected).all()
    assert m["gc_neg"].any(), "試験用の値動きにマイナス圏のゴールデンクロスが1回はあること"


def test_atr_matches_known(known):
    assert_same(ind.atr(known["high"], known["low"], known["close"]), known["atr14"])


def test_atr_hand_calc():
    # 真の値幅が毎回 2 になる値動きなら ATR も 2
    close = pd.Series([10.0 + i for i in range(20)])
    high, low = close + 1, close - 1
    a = ind.atr(high, low, close, 14)
    assert a.iloc[:14].isna().all()
    assert a.iloc[14] == pytest.approx(2.0)
    assert a.iloc[19] == pytest.approx(2.0)


def test_psar_matches_known(known):
    assert_same(ind.psar(known["high"], known["low"], known["close"]), known["sar"])


def test_psar_flips_bull_bear_bull():
    """陽→陰→陽の切り替わりが、既知の表(上げ15本・下げ15本・上げ15本)と同じ足で起きる。"""
    w = pd.read_csv(DATA / "psar_wave.csv")
    sar = ind.psar(w["high"], w["low"], w["close"])
    assert_same(sar, w["sar"])
    bull = ind.psar_bull(w["high"], w["low"], sar)
    bear = ind.psar_bear(w["high"], w["low"], sar)
    assert bull.iloc[1:17].all() and not bear.iloc[1:17].any()
    assert bear.iloc[17:32].all() and not bull.iloc[17:32].any()
    assert bull.iloc[32:].all()
    turns_to_bull = [i for i in range(1, len(w)) if bull.iloc[i] and not bull.iloc[i - 1] and bear.iloc[i - 1]]
    turns_to_bear = [i for i in range(1, len(w)) if bear.iloc[i] and bull.iloc[i - 1]]
    assert turns_to_bear == [17]
    assert turns_to_bull == [32]


def test_rci_hand_calc_with_ties():
    # 手計算: 順位差の2乗の合計は 4.5、RCI = (1 - 6*4.5/(9*80))*100 = 96.25
    closes = pd.Series([10, 12, 11, 13, 13, 15, 14, 16, 17], dtype=float)
    r = ind.rci(closes, 9)
    assert r.iloc[:8].isna().all()
    assert round(r.iloc[8], DECIMALS) == 96.25


def test_rci_all_same_price():
    # 全部同じ値段なら値段の順位は全部 5。順位差の2乗の合計は 60 → (1 - 360/720)*100 = 50
    r = ind.rci(pd.Series([7.0] * 9), 9)
    assert round(r.iloc[8], DECIMALS) == 50.0


def test_rci_extremes():
    up = ind.rci(pd.Series(np.arange(20, dtype=float)), 14)
    down = ind.rci(pd.Series(np.arange(20, 0, -1, dtype=float)), 14)
    assert up.iloc[13:].round(DECIMALS).eq(100.0).all()
    assert down.iloc[13:].round(DECIMALS).eq(-100.0).all()


@pytest.mark.parametrize("n", [9, 14])
def test_rci_matches_rank_by_rank(known, n):
    """1本ずつ平均順位を付けて式どおり計算したものと一致(同値を含む値動き)。"""
    close = known["close"].round(0)   # 丸めて同値をたくさん作る
    got = ind.rci(close, n)
    for t in range(n - 1, len(close)):
        win = close.iloc[t - n + 1:t + 1].reset_index(drop=True)
        price_rank = win.rank(method="average")
        time_rank = pd.Series(range(1, n + 1), dtype=float)
        d2 = ((time_rank - price_rank) ** 2).sum()
        expected = (1 - 6 * d2 / (n * (n * n - 1))) * 100
        assert round(got.iloc[t], DECIMALS) == round(expected, DECIMALS)
