"""Hyperliquid API の読み書き。API に触るのはこのファイルだけ(仕様書 12.1)。

第1版の段階1〜3では読み取り(情報取得の入口)だけを使う。注文は段階4で足す。
"""
from __future__ import annotations

import time

import requests

INFO_URL = "https://api.hyperliquid.xyz/info"
MAX_RETRIES = 3          # 仕様書 9.4: API 失敗は3回まで再試行
TIMEOUT_SEC = 20
MINUTE_MS = 60_000


class ExchangeError(RuntimeError):
    pass


def _post(payload: dict) -> object:
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(INFO_URL, json=payload, timeout=TIMEOUT_SEC)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(2 ** attempt)
    raise ExchangeError(f"情報取得に{MAX_RETRIES}回失敗: {last}")


def _parse(c: dict) -> dict:
    return {
        "ts_open": int(c["t"]),
        "open": float(c["o"]),
        "high": float(c["h"]),
        "low": float(c["l"]),
        "close": float(c["c"]),
        "volume": float(c["v"]),
        "trades": int(c["n"]) if c.get("n") is not None else None,
    }


def candles_1m(coin: str, start_ms: int, end_ms: int) -> list[dict]:
    """[start_ms, end_ms] の1分足を返す。1回の上限を超える期間は分割して取る(仕様書 5.1)。"""
    out: dict[int, dict] = {}
    cursor = start_ms
    while cursor <= end_ms:
        raw = _post({"type": "candleSnapshot",
                     "req": {"coin": coin, "interval": "1m",
                             "startTime": cursor, "endTime": end_ms}})
        if not raw:
            break
        rows = [_parse(c) for c in raw]
        for row in rows:
            out[row["ts_open"]] = row
        last = max(r["ts_open"] for r in rows)
        if last + MINUTE_MS <= cursor:
            break
        cursor = last + MINUTE_MS
    return [out[k] for k in sorted(out)]
