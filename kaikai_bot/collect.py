"""データ収集(仕様書 5.1〜5.5)。1時間に1回 `python -m kaikai_bot.collect` で動かす。"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config, exchange

MINUTE_MS = 60_000
HOUR_MS = 60 * MINUTE_MS
FAILS_FOR_ALERT = 3      # 仕様書 5.5: 3回連続失敗で警告

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles_1m (
    coin TEXT NOT NULL,
    ts_open INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    trades INTEGER,
    fetched_at INTEGER NOT NULL,
    PRIMARY KEY (coin, ts_open)
)
"""


def now_ms() -> int:
    return int(time.time() * 1000)


def connect(path: Path = config.MARKET_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    return con


def upsert(con: sqlite3.Connection, coin: str, rows: list[dict], fetched_at: int) -> int:
    """同じ足は上書き(仕様書 5.2)。"""
    con.executemany(
        "INSERT OR REPLACE INTO candles_1m VALUES (?,?,?,?,?,?,?,?,?)",
        [(coin, r["ts_open"], r["open"], r["high"], r["low"], r["close"],
          r["volume"], r["trades"], fetched_at) for r in rows])
    con.commit()
    return len(rows)


def latest_ts(con: sqlite3.Connection, coin: str) -> int | None:
    row = con.execute("SELECT MAX(ts_open) FROM candles_1m WHERE coin=?", (coin,)).fetchone()
    return row[0]


def find_gaps(con: sqlite3.Connection, coin: str, start_ms: int, end_ms: int) -> list[tuple[int, int]]:
    """[start_ms, end_ms] の中で1分足が無い区間を (最初の欠け, 最後の欠け) の組で返す。"""
    have = [r[0] for r in con.execute(
        "SELECT ts_open FROM candles_1m WHERE coin=? AND ts_open BETWEEN ? AND ? ORDER BY ts_open",
        (coin, start_ms, end_ms))]
    gaps = []
    expected = start_ms
    for ts in have + [end_ms + MINUTE_MS]:
        if ts > expected:
            gaps.append((expected, ts - MINUTE_MS))
        expected = ts + MINUTE_MS
    return gaps


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_gaps(coin: str, gaps: list[tuple[int, int]], path: Path = config.GAPS_LOG) -> None:
    """仕様書 5.4: 銘柄・開始・終了を data/gaps.log に記録する。同じ行は二度書かない。"""
    if not gaps:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = set(path.read_text(encoding="utf-8").splitlines()) if path.exists() else set()
    with open(path, "a", encoding="utf-8") as f:
        for a, b in gaps:
            line = f"{coin}\t{_iso(a)}\t{_iso(b)}"
            if line not in existing:
                f.write(line + "\n")


def alert(message: str, path: Path = config.ALERTS_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{_iso(now_ms())}\t{message}\n")


def _load_status(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"coins": {}}


def collect_coin(con: sqlite3.Connection, coin: str, now: int,
                 fetch=exchange.candles_1m) -> tuple[int, list[tuple[int, int]]]:
    """1銘柄を取り直して保存し、(保存した本数, 欠損区間) を返す。"""
    last = latest_ts(con, coin)
    # 仕様書 5.5: 保存庫の最新時刻の1時間前から現在まで。初回は取れるだけ取る。
    start = 0 if last is None else last - HOUR_MS
    rows = fetch(coin, start, now)
    # 確定した足(閉じた足)だけ保存する
    rows = [r for r in rows if r["ts_open"] + MINUTE_MS <= now]
    n = upsert(con, coin, rows, now)
    gaps: list[tuple[int, int]] = []
    if rows:
        window_start = rows[0]["ts_open"] if last is None else start
        window_end = rows[-1]["ts_open"]
        gaps = find_gaps(con, coin, window_start, window_end)
    return n, gaps


def run(stage: int | None = None, fetch=exchange.candles_1m,
        db_path: Path = config.MARKET_DB, status_path: Path = config.COLLECT_STATUS,
        gaps_path: Path = config.GAPS_LOG, alerts_path: Path = config.ALERTS_LOG) -> dict:
    con = connect(db_path)
    status = _load_status(status_path)
    now = now_ms()
    for info in config.coins(stage):
        coin = info["coin"]
        st = status["coins"].setdefault(coin, {"consecutive_failures": 0})
        try:
            n, gaps = collect_coin(con, coin, now, fetch)
            log_gaps(coin, gaps, gaps_path)
            st.update(consecutive_failures=0, last_success_ms=now, last_rows=n,
                      latest_ts_open=latest_ts(con, coin))
            print(f"{coin}: {n}本を保存、欠損 {len(gaps)}区間")
        except Exception as e:  # 失敗は次回に自動で取り直す(仕様書 5.5)
            st["consecutive_failures"] = st.get("consecutive_failures", 0) + 1
            st["last_error"] = str(e)
            print(f"{coin}: 失敗 {e}", file=sys.stderr)
            if st["consecutive_failures"] >= FAILS_FOR_ALERT:
                alert(f"収集 {coin} が{st['consecutive_failures']}回連続で失敗: {e}", alerts_path)
    status["last_run_ms"] = now
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    con.close()
    return status


def main() -> None:
    run()


if __name__ == "__main__":
    main()
