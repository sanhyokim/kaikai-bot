"""紙上運転の試験: 足を1本ずつ受け取って動かした結果が、検証エンジンと同じになること(12.1 の要)。"""
import numpy as np
import pytest

from kaikai_bot import backtest as bt, collect, config, paper, strategy

M = 60_000
T0 = 1_767_225_600_000
PRM = {"rci_slope_min": 20, "rci14_max": -50, "macd_lookback": 3, "stop_atr": 1.0, "take_atr": 2.0}


def _rows(n, seed=11):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    op = np.r_[close[0], close[:-1]]
    hi = np.maximum(op, close) * (1 + rng.uniform(0, 0.001, n))
    lo = np.minimum(op, close) * (1 - rng.uniform(0, 0.001, n))
    return [{"ts_open": T0 + i * M, "open": op[i], "high": hi[i], "low": lo[i], "close": close[i],
             "volume": 1.0, "trades": 1} for i in range(n)]


def fake_mask(f, params, us_allowed=None, disabled=frozenset()):
    """入り口の合図を、時刻だけで決まる作り物にする(検証と紙上で同じ合図が出ることを保証)。"""
    k = (f["ts_open"].to_numpy() // M) * 2654435761 % 97
    m = (k < 6) & f["atr"].notna().to_numpy() & ~f["gap_after"].to_numpy()
    return m if us_allowed is None else m & us_allowed


@pytest.mark.parametrize("exit_mode, gap", [("A", False), ("B", False), ("A", True)])
def test_paper_matches_backtest(tmp_path, monkeypatch, exit_mode, gap):
    monkeypatch.setattr(strategy, "entry_mask", fake_mask)
    n = 2 * 1440
    rows = _rows(n)
    if gap:      # 途中に1分足の欠け(5分足が数本作られない)
        rows = [r for r in rows if not (1500 <= (r["ts_open"] - T0) // M < 1517)]
    tf, width = "5m", 5 * M
    cfg, costs = config.strategy(), config.costs()

    # 検証エンジン: 全部の足を一度に
    full = collect.connect(tmp_path / "full.db")
    collect.upsert(full, "BTC", rows, 0)
    df = bt.candles.load_1m(full, "BTC")
    p = bt.prepare(df, "BTC", tf, False, set())
    start_close = T0 + 600 * M                 # 紙上運転を始める時刻(この足の確定から)
    mask = fake_mask(p.features, PRM)
    expected = bt.simulate(p, mask, PRM, exit_mode, start_close, p.end_ms, costs, cfg["fixed"], 1000.0)
    assert len(expected) >= 5

    # 紙上運転: 5分ごとに、その時点までの1分足だけを渡す
    market = collect.connect(tmp_path / "live.db")
    pcon = paper.connect(tmp_path / "paper.db")
    target = {"tf": tf, "coin": "BTC", "exit": exit_mode, "params": PRM, "kind": "初期値"}
    collect.upsert(market, "BTC", rows[:600], 0)
    runner = paper.Runner(pcon, target, cfg, costs, start_close)
    paper.process([runner], market, start_close, set())
    for now in range(start_close + width, T0 + n * M + 1, width):
        collect.upsert(market, "BTC", [r for r in rows if now - width <= r["ts_open"] < now], 0)
        # 途中で落ちて再起動した場合も同じになるよう、毎回 paper.db から読み戻す
        runner = paper.Runner(pcon, target, cfg, costs, now)
        paper.process([runner], market, now, set())
    got = pcon.execute("SELECT entry_open_ms, exit_close_ms, reason, net FROM trades ORDER BY entry_open_ms").fetchall()
    want = [(t["entry_open_ms"], t["exit_close_ms"], t["reason"], t["net"]) for t in expected]
    assert [g[:3] for g in got] == [w[:3] for w in want]
    np.testing.assert_allclose([g[3] for g in got], [w[3] for w in want], rtol=0, atol=1e-9)


def test_select_targets(tmp_path):
    cfg = config.strategy()
    t = paper.select_targets(cfg)
    assert len(t) == 24 and all(x["kind"] == "初期値" and x["params"] == cfg["initial"] for x in t)
    assert {x["coin"] for x in t} == {"BTC", "ETH", "SOL"}
    assert len(paper.select_targets(cfg, (1, 2))) == 48
    head = ",".join(bt.CSV_HEADER)
    prm = "傾き30 / RCI14≤-60 / GC2本 / 損切り0.75ATR / 利確3.0ATR"
    f = tmp_path / "judgement_2027-06-01_formal.csv"
    f.write_text(head + f"\n5m,BTC,A,{prm},a,b,1,400,-,-,1.6,1.5,2,1%,10%,x,-,y,0,合格,\n"
                 + f"1m,BTC,A,初期値({prm}),a,b,1,400,-,-,1.6,1.5,2,1%,10%,x,-,y,0,合格,\n", encoding="utf-8-sig")
    t = paper.select_targets(cfg, (1,), [f])
    extra = [x for x in t if x["kind"] == "選択"]
    assert extra == [{"tf": "5m", "coin": "BTC", "exit": "A", "kind": "選択",
                      "params": {"rci_slope_min": 30.0, "rci14_max": -60.0, "macd_lookback": 2,
                                 "stop_atr": 0.75, "take_atr": 3.0}}]


def test_settings_hold_positions_independently(tmp_path, monkeypatch):
    """2設定が同時に持ち玉を持てる(7.5)。"""
    monkeypatch.setattr(strategy, "entry_mask", fake_mask)
    rows = _rows(1440)
    market = collect.connect(tmp_path / "live.db")
    pcon = paper.connect(tmp_path / "paper.db")
    cfg, costs = config.strategy(), config.costs()
    collect.upsert(market, "BTC", rows[:1000], 0)
    targets = [{"tf": "1m", "coin": "BTC", "exit": e, "params": PRM, "kind": "初期値"} for e in ("A", "B")]
    now = T0 + 1000 * M
    runners = [paper.Runner(pcon, t, cfg, costs, now) for t in targets]
    paper.process(runners, market, now, set())
    both = 0
    for k in range(1001, 1440):
        collect.upsert(market, "BTC", rows[k - 1:k], 0)
        paper.process(runners, market, T0 + k * M, set())
        both += all(r.pos is not None for r in runners)
    assert both > 0


def test_run_results_candidate(tmp_path):
    con = paper.connect(tmp_path / "p.db")
    day = 86_400_000
    con.execute("INSERT INTO runs VALUES ('5m_BTC_A_初期値','5m','BTC','A','{}','初期値',0)")
    con.execute("INSERT INTO runs VALUES ('1m_BTC_A_初期値','1m','BTC','A','{}','初期値',0)")
    for i in range(60):
        net = 3.0 if i % 2 == 0 else -2.0
        for rid in ("5m_BTC_A_初期値", "1m_BTC_A_初期値"):
            con.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (rid, "初期値", i, i, 100.0, "A", "陽", i * 1000, 100.0, "take", 1, 0, 0, 0, net, net))
    head = ",".join(bt.CSV_HEADER)
    j = tmp_path / "j.csv"
    j.write_text(head + "\n5m,BTC,A,x,a,b,1,400,-,-,1.6,2.0,2,1%,10%,x,-,y,0,合格,\n"
                 + "1m,BTC,A,x,a,b,1,400,-,-,1.6,2.0,2,1%,10%,x,-,y,0,不合格,z\n", encoding="utf-8-sig")
    res = {r["run_id"]: r for r in paper.run_results(con, 29 * day, [j])}
    assert res["5m_BTC_A_初期値"]["pf"] == pytest.approx(1.5)
    assert res["5m_BTC_A_初期値"]["candidate"]            # 1.5 >= 0.7 × 2.0、正式判定で合格
    assert not res["1m_BTC_A_初期値"]["candidate"]        # 正式判定で合格していない
    res = {r["run_id"]: r for r in paper.run_results(con, 27 * day, [j])}
    assert not res["5m_BTC_A_初期値"]["candidate"]        # 28日に届かない
    # 正式判定がまだ無い: 基準4は外すが、基準6(正式判定の合格)が無いので候補にならない
    res = {r["run_id"]: r for r in paper.run_results(con, 29 * day, [])}
    assert "正式判定の後半PFの70%以上" not in res["5m_BTC_A_初期値"]["checks"]
    assert not res["5m_BTC_A_初期値"]["candidate"]
