"""設定ファイル(config/*.yaml)とパスの読み込み。"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
REPORTS_DIR = ROOT / "reports"
MEMOS_DIR = ROOT / "memos"

MARKET_DB = DATA_DIR / "market.db"
PAPER_DB = DATA_DIR / "paper.db"
GAPS_LOG = DATA_DIR / "gaps.log"
ALERTS_LOG = LOGS_DIR / "alerts.log"
COLLECT_STATUS = LOGS_DIR / "collect_status.json"

TIMEFRAMES = ["1m", "5m", "15m", "1h"]
TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
UPPER_TF = {"1m": "5m", "5m": "15m", "15m": "1h", "1h": "4h"}


def _load(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def coins(stage: int | None = None) -> list[dict]:
    rows = _load("coins.yaml")["coins"]
    if stage is not None:
        rows = [r for r in rows if r["stage"] == stage]
    return rows


def coin_info(coin: str) -> dict:
    for r in coins():
        if r["coin"] == coin:
            return r
    raise KeyError(f"config/coins.yaml にない銘柄: {coin}")


def costs() -> dict:
    return _load("costs.yaml")


def strategy() -> dict:
    return _load("strategy.yaml")


def us_holidays() -> set[str]:
    data = _load("us_holidays.yaml")["holidays"]
    return {str(d) for days in data.values() for d in days}
