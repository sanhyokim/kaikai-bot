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

```
# 監視画面
python -m kaikai_bot.dashboard
```

```
# 少額本番(段階4・ボスの指示後)
python -m kaikai_bot.live --coin BTC --tf 15m --exit A --capital 100
```

## データ収集を1時間ごとに自動で動かす

取得の入口は1分足を直近約3.6日分しか返しません。パソコンが3日以上止まると、その間のデータは二度と取れません。

### Windows(タスクスケジューラ)

リポジトリのフォルダで、次の1行を一度だけ実行します(`C:\path\to\kaikai-bot` は実際の場所に置き換え)。

```
schtasks /Create /SC HOURLY /TN kaikai-bot-collect /TR "cmd /c cd /d C:\path\to\kaikai-bot && .venv\Scripts\python.exe -m kaikai_bot.collect >> logs\collect.log 2>&1"
```

### Mac(cron)

`crontab -e` で次の1行を足します(`/path/to/kaikai-bot` は実際の場所に置き換え)。

```
5 * * * * cd /path/to/kaikai-bot && .venv/bin/python -m kaikai_bot.collect >> logs/collect.log 2>&1
```

## 試験

```
python -m pytest -q
```
