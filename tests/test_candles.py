"""上位足の生成と欠損の試験(仕様書 5.3、5.4、13.1)。"""
import numpy as np
import pandas as pd

from kaikai_bot import candles

M = 60_000
T0 = 1_767_225_600_000   # 2026-01-01 00:00 UTC


def make_1m(n, start=T0, skip=()):
    rows = []
    for i in range(n):
        if i in skip:
            continue
        rows.append({"ts_open": start + i * M, "open": 100 + i, "high": 101 + i + (i % 7),
                     "low": 99 + i - (i % 5), "close": 100.5 + i, "volume": 1.0 + i})
    return pd.DataFrame(rows)


def test_resample_60_bars_ohlcv():
    df = make_1m(60)
    for tf, n in [("5m", 5), ("15m", 15), ("1h", 60)]:
        out = candles.resample(df, tf)
        assert len(out) == 60 // n
        for k, row in out.iterrows():
            part = df.iloc[k * n:(k + 1) * n]
            assert row.ts_open == T0 + k * n * M
            assert row.open == part.open.iloc[0]
            assert row.high == part.high.max()
            assert row.low == part.low.min()
            assert row.close == part.close.iloc[-1]
            assert row.volume == part.volume.sum()


def test_resample_utc_boundaries():
    # 3分ずれた所から始まる1分足: 最初の不完全な5分足は作らない
    df = make_1m(13, start=T0 + 3 * M)
    out = candles.resample(df, "5m")
    assert list(out.ts_open) == [T0 + 5 * M, T0 + 10 * M]


def test_missing_bar_drops_upper_bar():
    df = make_1m(60, skip={7})
    five = candles.resample(df, "5m")
    assert T0 + 5 * M not in set(five.ts_open)
    assert len(five) == 11
    assert len(candles.resample(df, "15m")) == 3
    assert len(candles.resample(df, "1h")) == 0


def test_4h_from_1m():
    df = make_1m(480)
    out = candles.resample(df, "4h")
    assert list(out.ts_open) == [T0, T0 + 240 * M]


def test_gap_helpers():
    df = make_1m(30, skip={10, 11})
    assert list(np.flatnonzero(candles.gap_after(df, "1m"))) == [9]
    since = candles.bars_since_gap(df, "1m")
    assert since[0] == 1 and since[9] == 10
    assert since[10] == 1     # 欠損明けの最初の足(元の12本目)


def test_excluded_days_over_5_percent():
    day = 1440
    full = make_1m(day)                                   # 1日目: 欠けなし
    ok = make_1m(day, start=T0 + day * M, skip=set(range(72)))       # 2日目: 5%ちょうど欠け → 残す
    bad = make_1m(day, start=T0 + 2 * day * M, skip=set(range(73)))  # 3日目: 5%超 → 除外
    df = pd.concat([full, ok, bad], ignore_index=True)
    ex = candles.excluded_days(df)
    assert ex == [T0 + 2 * day * M]
    dropped = candles.drop_days(df, ex)
    assert dropped.ts_open.max() < T0 + 2 * day * M
