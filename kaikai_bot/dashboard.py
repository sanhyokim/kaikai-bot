"""監視画面(仕様書 第11章)。Streamlit で作り、1分ごとに自動で読み直す。

`python -m kaikai_bot.dashboard` で起動すると、ブラウザで画面が開く。
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):          # streamlit がこのファイルを直接読む時
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kaikai_bot import config  # noqa: E402

ALIVE_MS = 3 * 60_000          # 紙上運転の最終処理が3分以内なら「動いている」
COLLECT_ALIVE_MS = 2 * 3_600_000


def _t(ms) -> str:
    if not ms:
        return "なし"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def status_rows(now_ms: int) -> list[dict]:
    rows = []
    latest = None
    if config.MARKET_DB.exists():
        con = sqlite3.connect(config.MARKET_DB)
        latest = con.execute("SELECT MAX(ts_open) FROM candles_1m").fetchone()[0]
        con.close()
    rows.append({"項目": "収集(このパソコンの保存庫の最新の足)", "時刻": _t(latest),
                 "状態": "生きている" if latest and now_ms - latest < COLLECT_ALIVE_MS else "止まっている/古い"})
    cs = _json(config.COLLECT_STATUS)
    bad = [c for c, v in cs.get("coins", {}).items() if v.get("consecutive_failures", 0) >= 3]
    if bad:
        rows.append({"項目": "収集の警告", "時刻": _t(cs.get("last_run_ms")), "状態": "3回以上連続失敗: " + ", ".join(bad)})
    ps = _json(config.LOGS_DIR / "paper_status.json")
    last = ps.get("last_loop_ms")
    alive = "動いている" if last and now_ms - last < ALIVE_MS else "止まっている"
    if ps.get("runs"):
        alive += f"({ps['runs']}設定、持ち玉 {ps.get('holding', 0)})"
    rows.append({"項目": "紙上運転", "時刻": _t(last), "状態": alive})
    rows.append({"項目": "少額本番", "時刻": "なし", "状態": "未開始(段階4・ボスの指示待ち)"})
    rows.append({"項目": "緊急停止(STOP ファイル)", "時刻": "なし",
                 "状態": "置かれている" if (config.ROOT / "STOP").exists() else "なし"})
    return rows


def render() -> None:
    import streamlit as st
    from kaikai_bot import backtest as bt, paper

    st.set_page_config(page_title="kaikai-bot 監視", layout="wide")
    st.title("kaikai-bot 監視画面")

    @st.fragment(run_every=60)
    def body():
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        st.caption(f"更新: {_t(now)}(1分ごとに自動で読み直します)")

        st.header("1. 状態")
        st.table(status_rows(now))

        st.header("2. 判定表")
        csvs = paper.latest_reports()
        if csvs:
            import pandas as pd
            for path in csvs:
                table = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
                if "判定" in table:
                    counts = table["判定"].value_counts()
                    st.write(f"{path.stem}: " + " / ".join(f"{k} {v}" for k, v in counts.items()))
                else:
                    st.write(f"{path.stem}(中間報告。判定はしない)")
                st.dataframe(table, hide_index=True, use_container_width=True)
        else:
            st.info("判定表はまだありません。")

        st.header("3. 紙上運転の成績")
        if config.PAPER_DB.exists():
            con = paper.connect()
            res = paper.run_results(con, now, paper.latest_formal())
            st.table([{"設定": r["run_id"], "パラメータ": r["kind"], "日数": f"{r['days']:.1f}", "取引回数": r["trades"],
                       "PF": bt._f(r["pf"]), "純損益(ドル)": f"{r['net']:.2f}",
                       "最大の落ち込み": bt._f(r["max_dd"], pct=True),
                       "昇格候補": "★ 昇格候補" if r["candidate"] else ""} for r in res])
            rows = con.execute("SELECT run_id, entry_open_ms, entry_price, mark, psar, exit_close_ms, exit_price, "
                               "reason, net FROM trades ORDER BY exit_close_ms DESC LIMIT 20").fetchall()
            if rows:
                st.subheader("直近の取引")
                st.table([{"設定": r[0], "入った時刻": _t(r[1]), "入値": f"{r[2]:.4f}", "印": r[3], "パラボ": r[4],
                           "出た時刻": _t(r[5]), "出値": f"{r[6]:.4f}", "理由": bt.REASON_NAMES.get(r[7], r[7]),
                           "純損益": f"{r[8]:.2f}"} for r in rows])
            con.close()
        else:
            st.info("紙上運転の記録はまだありません。")

        st.header("4. 本番の成績")
        st.info("段階4(少額本番)はボスの指示があるまで始めません。")

        st.header("5. 警告一覧")
        if config.ALERTS_LOG.exists():
            lines = config.ALERTS_LOG.read_text(encoding="utf-8").splitlines()[-50:][::-1]
            st.table([{"日時(UTC)": x.split("\t")[0], "内容": x.split("\t", 1)[-1]} for x in lines])
        else:
            st.success("警告はありません。")

    body()


def main() -> None:
    subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
                     "--server.headless=false"])


if __name__ == "__main__":
    try:
        from streamlit.runtime import exists as _in_streamlit
    except ImportError:
        _in_streamlit = lambda: False  # noqa: E731
    if _in_streamlit():
        render()
    else:
        main()
