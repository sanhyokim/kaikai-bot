"""収集の試験。API の代わりに作り物の取得関数を渡す(本物の API には触らない)。"""
import json

from kaikai_bot import collect

M = 60_000
T0 = 1_767_225_600_000


def fake_fetch_factory(available, fail=False):
    calls = []

    def fetch(coin, start, end):
        calls.append((coin, start, end))
        if fail:
            raise RuntimeError("つながらない")
        return [dict(ts_open=t, open=1.0, high=2.0, low=0.5, close=1.5, volume=3.0, trades=4)
                for t in available if start <= t <= end]
    return fetch, calls


def test_collect_saves_closed_bars_and_overwrites(tmp_path):
    con = collect.connect(tmp_path / "m.db")
    now = T0 + 10 * M + 30_000          # 10本目の途中
    fetch, calls = fake_fetch_factory([T0 + i * M for i in range(11)])
    n, gaps = collect.collect_coin(con, "BTC", now, fetch)
    assert n == 10                        # 途中の足は保存しない
    assert gaps == []
    assert calls[0][1] == 0               # 初回は取れるだけ
    # 2回目: 最新の1時間前から取り直し、同じ足は上書き
    n2, _ = collect.collect_coin(con, "BTC", now + 5 * M, fetch)
    assert calls[1][1] == T0 + 9 * M - 60 * M
    count = con.execute("SELECT COUNT(*) FROM candles_1m").fetchone()[0]
    assert count == 11


def test_collect_records_gaps(tmp_path):
    con = collect.connect(tmp_path / "m.db")
    avail = [T0 + i * M for i in range(20) if i not in (5, 6, 7)]
    fetch, _ = fake_fetch_factory(avail)
    _, gaps = collect.collect_coin(con, "ETH", T0 + 21 * M, fetch)
    assert gaps == [(T0 + 5 * M, T0 + 7 * M)]
    log = tmp_path / "gaps.log"
    collect.log_gaps("ETH", gaps, log)
    collect.log_gaps("ETH", gaps, log)    # 同じ行は二度書かない
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines == ["ETH\t2026-01-01T00:05:00Z\t2026-01-01T00:07:00Z"]


def test_three_failures_raise_alert(tmp_path, monkeypatch):
    fetch, _ = fake_fetch_factory([], fail=True)
    paths = dict(db_path=tmp_path / "m.db", status_path=tmp_path / "s.json",
                 gaps_path=tmp_path / "g.log", alerts_path=tmp_path / "a.log")
    monkeypatch.setattr(collect.config, "coins", lambda stage=None: [{"coin": "BTC"}])
    for _ in range(2):
        collect.run(fetch=fetch, **paths)
    assert not (tmp_path / "a.log").exists()
    collect.run(fetch=fetch, **paths)
    assert "3回連続" in (tmp_path / "a.log").read_text(encoding="utf-8")
    st = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert st["coins"]["BTC"]["consecutive_failures"] == 3
    ok_fetch, _ = fake_fetch_factory([1_000 * M])
    collect.run(fetch=ok_fetch, **paths)
    st = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert st["coins"]["BTC"]["consecutive_failures"] == 0
