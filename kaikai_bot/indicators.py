"""指標の計算(仕様書 第6章)。すべて終値ベース。計算不能の足は NaN。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TURN_LOOKBACK = 10   # 仕様書 6.1: 上向き = 10本前より高い
RCI_CHUNK = 50_000


def sma(close: pd.Series, n: int) -> pd.Series:
    """6.1 単純移動平均: 直近 n 本の終値の平均。"""
    return close.rolling(n, min_periods=n).mean()


def rising(series: pd.Series, lookback: int = TURN_LOOKBACK) -> pd.Series:
    """6.1 上向き: x[t] > x[t-10]。どちらかが計算不能なら False。"""
    prev = series.shift(lookback)
    return (series > prev) & series.notna() & prev.notna()


def ema(x: pd.Series, n: int) -> pd.Series:
    """指数移動平均。係数 2/(n+1)。最初の値は先頭 n 本の単純平均(TA-Lib と同じ)。

    先頭の計算不能(NaN)は飛ばし、そこから n 本たまった足を最初の値にする。
    """
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    valid = np.flatnonzero(~np.isnan(v))
    if len(valid) >= n:
        s = valid[0]
        k = 2.0 / (n + 1)
        e = v[s:s + n].mean()
        out[s + n - 1] = e
        for i in range(s + n, len(v)):
            e = e + k * (v[i] - e)
            out[i] = e
    return pd.Series(out, index=x.index)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """6.3 MACD 線・シグナル線・マイナス圏のゴールデンクロス。"""
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    gc = (line.shift(1) <= sig.shift(1)) & (line > sig)
    gc_neg = gc & (line < 0)
    return pd.DataFrame({"macd": line, "signal": sig, "gc": gc, "gc_neg": gc_neg})


def rci(close: pd.Series, n: int) -> pd.Series:
    """6.4 順位相関指数。同値は平均順位。範囲は -100〜+100。"""
    values = close.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) < n:
        return pd.Series(out, index=close.index)
    all_win = sliding_window_view(values, n)             # (本数, n)。右端が最新
    time_rank = np.arange(1, n + 1, dtype=float)         # 新しいほど大きい
    res = np.empty(len(all_win))
    for s in range(0, len(all_win), RCI_CHUNK):          # メモリを食いすぎないよう分けて計算
        win = all_win[s:s + RCI_CHUNK]
        less = (win[:, None, :] < win[:, :, None]).sum(axis=2)
        equal = (win[:, None, :] == win[:, :, None]).sum(axis=2)
        price_rank = less + (equal + 1) / 2.0            # 高いほど大きい、同値は平均順位
        d2 = ((time_rank - price_rank) ** 2).sum(axis=1)
        res[s:s + RCI_CHUNK] = (1 - 6 * d2 / (n * (n * n - 1))) * 100
    out[n - 1:] = res
    return pd.Series(out, index=close.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """6.5 ATR: 真の値幅の Wilder 平均(TA-Lib と同じ)。

    真の値幅は前の終値が要るので2本目から。最初の ATR は2本目〜n+1本目の真の値幅の単純平均。
    """
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    trv = tr.to_numpy(dtype=float)
    out = np.full(len(trv), np.nan)
    if len(trv) > n:
        out[n] = trv[1:n + 1].mean()
        for i in range(n + 1, len(trv)):
            out[i] = (out[i - 1] * (n - 1) + trv[i]) / n
    return pd.Series(out, index=close.index)


def psar(high: pd.Series, low: pd.Series, close: pd.Series,
         af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.20) -> pd.Series:
    """6.2 パラボリック SAR(Wilder 方式。広く使われる TA-Lib と同じ決まり)。最初の足は計算不能。

    - 始まりの向き: 2本目で「安値の切り下がり幅」が「高値の切り上がり幅」より大きく正なら下降、それ以外は上昇。
      上昇なら SAR=1本目の安値・極値=2本目の高値、下降なら SAR=1本目の高値・極値=2本目の安値。
    - 各足の SAR はその足を見る前に決まった値。足の安値(下降中は高値)が SAR に触れたら反転する。
    - 反転した足の SAR はそれまでの極値で、その足と1本前の足の値幅の外側に置く。
    close は使わない(引数は他の指標とそろえるため)。
    """
    h = high.to_numpy(dtype=float)
    lo = low.to_numpy(dtype=float)
    out = np.full(len(h), np.nan)
    if len(h) < 2:
        return pd.Series(out, index=high.index)
    diff_minus = lo[0] - lo[1]
    diff_plus = h[1] - h[0]
    up = not (diff_minus > 0 and diff_minus > diff_plus)
    if up:
        ep, sar = h[1], lo[0]
    else:
        ep, sar = lo[1], h[0]
    af = af_start
    new_low, new_high = lo[1], h[1]
    for i in range(1, len(h)):
        prev_low, prev_high = new_low, new_high
        new_low, new_high = lo[i], h[i]
        if up:
            if new_low <= sar:                       # 下降に反転
                up = False
                sar = max(ep, prev_high, new_high)
                out[i] = sar
                af, ep = af_start, new_low
                sar = max(sar + af * (ep - sar), prev_high, new_high)
            else:
                out[i] = sar
                if new_high > ep:
                    ep, af = new_high, min(af + af_step, af_max)
                sar = min(sar + af * (ep - sar), prev_low, new_low)
        else:
            if new_high >= sar:                      # 上昇に反転
                up = True
                sar = min(ep, prev_low, new_low)
                out[i] = sar
                af, ep = af_start, new_high
                sar = min(sar + af * (ep - sar), prev_low, new_low)
            else:
                out[i] = sar
                if new_low < ep:
                    ep, af = new_low, min(af + af_step, af_max)
                sar = max(sar + af * (ep - sar), prev_high, new_high)
    return pd.Series(out, index=high.index)


def psar_bull(high: pd.Series, low: pd.Series, sar: pd.Series) -> pd.Series:
    """6.2 陽: SAR が足の安値より下。"""
    return (sar < low) & sar.notna()


def psar_bear(high: pd.Series, low: pd.Series, sar: pd.Series) -> pd.Series:
    """6.2 陰: SAR が足の高値より上。"""
    return (sar > high) & sar.notna()
