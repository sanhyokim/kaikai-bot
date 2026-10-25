"""試験用の「既知の値」の表を作る(1回だけ手で動かした。結果は tests/data/ に保存済み)。

照合相手は広く使われる計算ライブラリ TA-Lib(仕様書 12.3: 外部の指標ライブラリは試験の照合用にだけ使ってよい)。
本体のコードは TA-Lib を使わない。作り直す時: pip install TA-Lib のあと python tests/make_known_values.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import talib

OUT = Path(__file__).parent / "data"


def price_table(n=120, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.05, 1.0, n)
    low = close - rng.uniform(0.05, 1.0, n)
    return pd.DataFrame({"high": high.round(4), "low": low.round(4), "close": close.round(4)})


def wave_table():
    """上げ15本→下げ15本→上げ15本の、向きがはっきりした値動き(パラボリックの切り替わり確認用)。"""
    steps = [1.0] * 15 + [-1.0] * 15 + [1.0] * 15
    close = 100 + np.cumsum(steps)
    return pd.DataFrame({"high": close + 0.3, "low": close - 0.3, "close": close})


def main():
    OUT.mkdir(exist_ok=True)
    df = price_table()
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    df["sma5"] = talib.SMA(c, 5)
    df["sma25"] = talib.SMA(c, 25)
    df["sma75"] = talib.SMA(c, 75)
    e12, e26 = talib.EMA(c, 12), talib.EMA(c, 26)
    line = e12 - e26
    sig = np.full(len(c), np.nan)
    ok = ~np.isnan(line)
    sig[ok] = talib.EMA(line[ok], 9)
    df["macd"] = line
    df["signal"] = sig
    df["atr14"] = talib.ATR(h, l, c, 14)
    df["sar"] = talib.SAR(h, l, 0.02, 0.2)
    df.to_csv(OUT / "known_values.csv", index=False, float_format="%.10f")

    w = wave_table()
    w["sar"] = talib.SAR(w["high"].to_numpy(), w["low"].to_numpy(), 0.02, 0.2)
    w.to_csv(OUT / "psar_wave.csv", index=False, float_format="%.10f")


if __name__ == "__main__":
    main()
