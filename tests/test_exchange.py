"""API 読み取りの試験。本物の API の代わりに作り物の応答を使う。"""
from kaikai_bot import exchange

M = 60_000


def test_candles_paging(monkeypatch):
    # 1回で3本しか返さない作り物の API。期間全体(7本)を分割して取り切ること。
    all_ts = [i * M for i in range(7)]
    seen = []

    def fake_post(payload):
        req = payload["req"]
        seen.append(req["startTime"])
        rows = [t for t in all_ts if req["startTime"] <= t <= req["endTime"]][:3]
        return [{"t": t, "T": t + M - 1, "o": "1", "h": "2", "l": "0.5", "c": "1.5",
                 "v": "10", "n": 3, "s": req["coin"], "i": "1m"} for t in rows]

    monkeypatch.setattr(exchange, "_post", fake_post)
    out = exchange.candles_1m("BTC", 0, 6 * M)
    assert [r["ts_open"] for r in out] == all_ts
    assert seen == [0, 3 * M, 6 * M]
    assert out[0] == {"ts_open": 0, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
                      "volume": 10.0, "trades": 3}
