# kaikai-bot

kaikai 氏の1分足スキャルピング手法を機械のルールに書き直し、Hyperliquid で本当に利益が出るかを数字で判定するための道具です。
決まりの正本は [docs/SPEC.md](docs/SPEC.md)、実装役が自分で決めたことは [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md)、今の状態は [docs/snapshot.md](docs/snapshot.md) にあります。

## 動かし方

```
# 準備(最初の1回)
git clone https://github.com/sanhyokim/kaikai-bot.git
cd kaikai-bot
python -m venv .venv
.venv\Scripts\activate          # Windows / Mac は source .venv/bin/activate
pip install -r requirements.txt
```

```
# データ収集(1時間ごとの定時実行に登録。手動なら)
python -m kaikai_bot.collect
```

```
# 検証
python -m kaikai_bot.backtest --stage 1
```

```
# 紙上運転(常駐)
python -m kaikai_bot.paper
```

(落ちたら自動で起動し直したい時は `python -m kaikai_bot.paper --watch`。紙上運転の前に `python -m kaikai_bot.collect --from-github` と `python -m kaikai_bot.backtest --stage 1` を一度実行しておく。)

```
# 監視画面
python -m kaikai_bot.dashboard
```

```
# 少額本番(段階4・ボスの指示後)
python -m kaikai_bot.live --coin BTC --tf 15m --exit A --capital 100
```

## データ収集を1時間ごとに自動で動かす

取得の入口は1分足を直近約3.6日分しか返しません。止まっている間のデータは二度と取れないので、
収集は GitHub の自動実行で1時間ごとに動かし、リポジトリの `data` 枝に「銘柄/日付.csv」で貯めています(ボス決定 2026-10-25)。
パソコンでは何も登録しなくて大丈夫です。

検証や紙上運転の前に、貯まった分をパソコンの保存庫に取り込みます。

```
python -m kaikai_bot.collect --from-github
```

予備としてパソコンでも集める場合(同じ足は上書きなのでぶつかりません):

- Windows: `schtasks /Create /SC HOURLY /TN kaikai-bot-collect /TR "cmd /c cd /d C:\path\to\kaikai-bot && .venv\Scripts\python.exe -m kaikai_bot.collect >> logs\collect.log 2>&1"`
- Mac: `crontab -e` で `5 * * * * cd /path/to/kaikai-bot && .venv/bin/python -m kaikai_bot.collect >> logs/collect.log 2>&1`

## 試験

```
python -m pytest -q
```
