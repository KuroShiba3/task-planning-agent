# Task Planning Agent

LangGraphとGoogle Geminiを使用した、マルチエージェントタスク計画システム。

ユーザーの質問を複数の独立したサブタスクに分割し、各タスクをWeb検索で並列実行し、結果を統合して包括的な回答を生成。

## 機能

- ユーザーの質問を実行可能な独立したサブタスクに自動分割
- 複数のタスクを並列実行
- 各タスク内で複数の検索クエリを並列実行
- 検索結果とタスク結果の品質を自動評価し、必要に応じて改善（最大2回まで）
- 全タスクの結果を統合して、わかりやすい最終回答を生成

## プロジェクト構造

```
task-planning-agent/
├── src/
│   ├── agents/
│   │   ├── planner/
│   │   │   ├── __init__.py
│   │   │   ├── graph.py       # Plannerグラフの定義
│   │   │   └── nodes.py       # タスク計画・回答生成ノード
│   │   └── websearch/
│   │       ├── __init__.py
│   │       ├── graph.py       # WebSearchグラフの定義
│   │       ├── nodes.py       # 検索・評価ノード
│   │       └── state.py       # WebSearch状態管理
│   ├── config/
│   │   ├── config.py          # 環境変数の読み込み
│   │   └── model.py           # LLMモデルの設定
│   ├── graph/
│   │   ├── __init__.py
│   │   └── builder.py         # グラフの統合
│   ├── state/
│   │   ├── __init__.py
│   │   └── state.py           # ベース状態管理
│   └── utils/
│       ├── __init__.py
│       └── logger.py          # ロガー設定
├── main.py                    # エントリーポイント
├── pyproject.toml             # プロジェクト設定
├── Dockerfile                 # Docker設定
├── .env.sample                # 環境変数のサンプル
└── README.md
```

## 技術スタック

- **LLM**: Google Gemini (gemini-2.5-flash)
- **フレームワーク**: LangGraph 0.2.0+
- **検索**: Google Custom Search API
- **Webスクレイピング**: WebBaseLoader (BeautifulSoup)
- **言語**: Python 3.12+
- **パッケージ管理**: uv

## 必要な環境変数

以下の環境変数が必要：

| 環境変数名 | 説明 | 取得方法 |
|-----------|------|---------|
| `GOOGLE_API_KEY` | Google Gemini APIキー | [Google AI Studio](https://aistudio.google.com/app/apikey)で取得 |
| `GOOGLE_CSE_ID` | Google Custom Search Engine ID | [Programmable Search Engine](https://programmablesearchengine.google.com/)で作成 |

### 環境変数の設定方法

1. `.env.sample`をコピーして`.env`を作成

```bash
cp .env.sample .env
```

2. `.env`ファイルを編集して、実際のAPIキーとCSE IDを設定

## セットアップ

### 1. ローカル環境で実行

#### 前提条件
- Python 3.12以上
- [uv](https://github.com/astral-sh/uv)がインストールされていること

#### インストールと実行

```bash
# 依存関係のインストール
uv sync

# アプリケーションの実行
uv run main.py
```

### 2. Dockerで実行

#### イメージのビルド

```bash
docker build -t task-planning-agent .
```

#### コンテナの実行

```bash
docker run task-planning-agent
```