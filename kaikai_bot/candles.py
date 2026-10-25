"""上位足の生成と欠損の処理(仕様書 5.3、5.4)。"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .config import TF_MINUTES

MINUTE_MS = 60_000
DAY_MS = 1440 * MINUTE_MS
MAX_DAY_MISSING = 0.05   # 仕様書 5.4: 1日の欠損が5%を超える日は除外
COLUMNS = ["ts_open", "open", "high", "low", "close", "volume"]


def load_1m(con: sqlite3.Connection, coin: str,
            start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
    q = "SELECT ts_open, open, high, low, close, volume FROM candles_1m WHERE coin=?"
    args: list = [coin]
    if start_ms is not None:
        q += " AND ts_open >= ?"
        args.append(start_ms)
    if end_ms is not None:
        q += " AND ts_open <= ?"
        args.append(end_ms)
    q += " ORDER BY ts_open"
    return pd.read_sql_query(q, con, params=args)


def excluded_days(df1m: pd.DataFrame) -> list[int]:
    """1分足の欠けが5%を超える UTC の日(その日の0時のミリ秒)を返す。"""
    if df1m.empty:
        return []
    day = df1m["ts_open"].to_numpy() // DAY_MS
    counts = pd.Series(1, index=day).groupby(level=0).sum()
    all_days = np.arange(day.min(), day.max() + 1)
    counts = counts.reindex(all_days, fill_value=0)
    bad = counts[counts < 1440 * (1 - MAX_DAY_MISSING)].index
    return [int(d) * DAY_MS for d in bad]


def drop_days(df1m: pd.DataFrame, days_ms: list[int]) -> pd.DataFrame:
    """除外日の1分足を落とす(除外日は欠損として扱われる)。"""
    if not days_ms:
        return df1m
    day = df1m["ts_open"] // DAY_MS * DAY_MS
    return df1m[~day.isin(days_ms)].reset_index(drop=True)


def resample(df1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """1分足から上位足を作る。1本でも欠けた区間の足は作らない(穴埋めしない)。"""
    n = TF_MINUTES[tf]
    if n == 1:
        return df1m[COLUMNS].reset_index(drop=True)
    if df1m.empty:
        return pd.DataFrame(columns=COLUMNS)
    width = n * MINUTE_MS
    key = df1m["ts_open"] // width * width
    g = df1m.groupby(key, sort=True)
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "count": g["ts_open"].count(),
    })
    out = out[out["count"] == n].drop(columns="count")
    out.index.name = "ts_open"
    return out.reset_index()[COLUMNS]


def gap_after(bars: pd.DataFrame, tf: str) -> np.ndarray:
    """各足の直後の足が欠けているか(最後の足は False)。"""
    ts = bars["ts_open"].to_numpy()
    width = TF_MINUTES[tf] * MINUTE_MS
    out = np.zeros(len(ts), dtype=bool)
    if len(ts) > 1:
        out[:-1] = np.diff(ts) != width
    return out


def bars_since_gap(bars: pd.DataFrame, tf: str) -> np.ndarray:
    """欠損明けから何本目か(欠損明けの最初の足が 1)。先頭の足も欠損明け扱い。"""
    ts = bars["ts_open"].to_numpy()
    width = TF_MINUTES[tf] * MINUTE_MS
    out = np.zeros(len(ts), dtype=np.int64)
    run = 0
    for i in range(len(ts)):
        if i == 0 or ts[i] - ts[i - 1] != width:
            run = 1
        else:
            run += 1
        out[i] = run
    return out
