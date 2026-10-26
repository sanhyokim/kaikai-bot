"""日ごとのファイルの保管庫(GitHub の自動収集用)の試験。"""
from kaikai_bot import collect

M = 60_000
T0 = 1_767_225_600_000   # 2026-01-01 00:00 UTC


def fake_fetch(available):
    def fetch(coin, start, end):
        return [dict(ts_open=t, open=1.0, high=2.0, low=0.5, close=1.25, volume=3.0, trades=4)
                for t in available if start <= t <= end]
    return fetch


def test_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(collect.config, "coins", lambda stage=None: [{"coin": "xyz:NVDA"}])
    now = T0 + 1440 * M + 10 * M
    monkeypatch.setattr(collect, "now_ms", lambda: now)
    avail = [T0 + i * M for i in range(1450)]
    collect.run_store(tmp_path, fetch=fake_fetch(avail))
    d = tmp_path / "candles" / "xyz_NVDA"
    assert sorted(f.name for f in d.glob("*.csv")) == ["2026-01-01.csv", "2026-01-02.csv"]
    assert len((d / "2026-01-01.csv").read_text().splitlines()) == 1441
    assert not (tmp_path / "work.db").exists()
    # もう一度: 保管庫から最新を読み戻し、「最新の1時間前」から取り直す
    seen = []

    def fetch2(coin, start, end):
        seen.append(start)
        return fake_fetch(avail)(coin, start, end)
    collect.run_store(tmp_path, fetch=fetch2)
    assert seen == [T0 + 1449 * M - 60 * M]
    # 取り込み
    con = collect.connect(tmp_path / "m.db")
    n = collect.import_store(con, tmp_path)
    assert n == 1450
    row = con.execute("SELECT close, trades FROM candles_1m WHERE ts_open=?", (T0,)).fetchone()
    assert row == (1.25, 4)


def test_pull_from_github_brings_paper(tmp_path, monkeypatch):
    """パソコンへの取り込みで、GitHub の紙上運転の記録と状態も入る。"""
    import io
    import json
    import zipfile

    import requests
    from kaikai_bot import paper

    src = paper.connect(tmp_path / "src.db")
    src.execute("INSERT INTO runs VALUES ('5m_BTC_A_初期値','5m','BTC','A','{}','初期値',0)")
    src.commit()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("kaikai-bot-data/candles/BTC/2026-01-01.csv",
                   ",".join(collect.STORE_COLUMNS) + f"\n{T0},1.0,2.0,0.5,1.25,3.0,4,0\n")
        for t, text in paper.dump_tables(src).items():
            z.writestr(f"kaikai-bot-data/paper/{t}.jsonl", text)
        z.writestr("kaikai-bot-data/paper/status.json", json.dumps({"runs": 1}))
        z.writestr("kaikai-bot-data/alerts.log", "t\tx\n")

    class R:
        content = buf.getvalue()

        def raise_for_status(self):
            pass
    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(collect.config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(collect.config, "ALERTS_LOG", tmp_path / "logs" / "alerts.log")
    monkeypatch.setattr(collect.config, "GAPS_LOG", tmp_path / "gaps.log")
    monkeypatch.setattr(collect.config, "COLLECT_STATUS", tmp_path / "logs" / "collect_status.json")
    n = collect.pull_from_github(tmp_path / "m.db", paper_db=tmp_path / "p.db")
    assert n == 1
    got = paper.connect(tmp_path / "p.db").execute("SELECT run_id FROM runs").fetchall()
    assert got == [("5m_BTC_A_初期値",)]
    assert json.loads((tmp_path / "logs" / "paper_status.json").read_text()) == {"runs": 1}
    assert (tmp_path / "logs" / "alerts.log").read_text() == "t\tx\n"
