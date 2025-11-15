from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import END
from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from ...config.model import get_model
from ...state.state import BaseState
from ...utils.logger import get_logger

logger = get_logger(__name__)

async def plan_tasks(state: BaseState) -> Command:
    """ユーザーのリクエストを分析し、実行可能な独立したサブタスクに分割するノード"""

    class Task(BaseModel):
        task_description: str = Field(description="タスクの内容を記述してください。")

    class TaskPlan(BaseModel):
        tasks: list[Task] = Field(description="実行するタスクのリスト（最低1つ以上）")
        reason: str = Field(description="タスク分割の戦略と根拠を説明してください。")

    system_message = SystemMessage(
        content="""
ユーザーのリクエストを実行可能な独立したサブタスクに分割してください。

## システムアーキテクチャの理解:
このシステムは以下の3段階で動作します:
1. **タスク計画（あなたの役割）**: ユーザーのリクエストを複数のタスクに分割する
2. **タスク実行**: 各タスクをwebsearchエージェントが並列実行し、Web検索で情報収集を行って結果を返す
3. **回答生成**: すべてのタスク結果を統合してユーザーに最終回答を提示

**重要**:
- 各タスクはwebsearchエージェントが独立して実行します
- 複数のタスクは並列実行されます
- あなたが作成したタスクの内容が、websearchエージェントへの指示になります

## websearchエージェントについて

websearchエージェントは以下の機能を持っています:
- Google検索を実行し、検索結果のページ内容を取得・分析
- ページの本文コンテンツを読み込み、タスクに関連する情報を抽出
- 最新ニュース、天気、技術情報、製品情報など、Web上の公開情報の取得に最適

## サブタスクの作成ルール

### 基本方針
- **並列実行を活用**: websearchエージェントは複数のタスクを同時に実行できるため、独立したタスクは分割してください
- **検索の効率化**: 異なる対象（場所、期間、項目など）は別々のタスクにすることで、検索精度が向上します

### 必須要件

1. **必ず1つ以上のサブタスクを作成してください**
    - 単一の質問でも、最低1つのサブタスクを作成します

2. **各サブタスクは完全に独立している必要があります**
    - タスク間に依存関係を持たせないでください
    - あるタスクの結果が別のタスクの入力になるような分割は避けてください
    - 各タスクは単独で実行・完了できる内容にしてください
    - 各タスクは並列実行されるため、順序に依存しない設計にしてください

3. **タスクの内容は具体的で明確にしてください**
    - websearchエージェントへの指示として機能するよう、タスクの内容を明確に記述してください
    - Web検索で見つけられる情報に焦点を当ててください
"""    )

    try:
        model = get_model()
        plan = await model.with_structured_output(TaskPlan).ainvoke(
            [system_message] +  state["messages"]
        )

        if not plan.tasks:
            logger.error("plan_tasksでタスクが空です")
            raise

        sends = [
            Send(
                "websearch",
                {
                    "task_id": str(idx),
                    "task_description": task.task_description,
                    "attempt": 0,
                    "completed": False
                }
            )
            for idx, task in enumerate(plan.tasks)
        ]

        initial_tasks = [
            {
                "task_id": str(idx),
                "task_description": task.task_description,
            }
            for idx, task in enumerate(plan.tasks)
        ]

        return Command(
            update={"tasks": initial_tasks},
            goto=sends
        )
    except Exception as e:
        logger.error(f"plan_tasksでエラーが発生しました: {str(e)}", exc_info=True)
        raise

async def generate_final_answer(state: BaseState) -> Command:
    """完了したタスクの結果を統合して、ユーザーへの最終回答を生成"""

    tasks = state.get("tasks", [])

    if not tasks:
        logger.error(f"generate_final_answerでタスクが空です")
        raise

    task_results_text = "\n\n".join([
        f"【タスク{idx+1}】{task['task_description']}\n結果: {task['task_result']}"
        for idx, task in enumerate(tasks)
    ])

    messages = state.get("messages", [])
    user_query = messages[0].content

    system_message = SystemMessage(content=f"""
複数のタスクの実行結果を統合し、ユーザーの質問に対する包括的で分かりやすい回答を生成してください。

## 回答のルール:

1. **統合と一貫性**:
    - 各タスクの結果を適切に統合し、全体として一貫性のある回答にする
    - タスクの結果を単純に羅列するのではなく、自然な文章として統合する

2. **わかりやすさ**:
    - 簡潔で分かりやすい日本語で記述
    - ユーザーの質問に直接答える形式にする
    - 必要に応じて箇条書きや見出しを使って構造化

3. **完全性**:
    - 全てのタスク結果から重要な情報を漏らさず含める
    - ユーザーの質問に対して包括的に答える

5. **情報源の記載形式（必須）**:
    - タスク結果で実際に利用した情報源のURLのみを記載する
    - 利用していないURLは記載しない
    - 同じドメインのURLが複数ある場合は、代表的なURL1つのみを記載する
    - URLの最大個数制限はなし（すべての重要な情報源を記載）
    - フォーマット例:
    ```
    （回答の本文）

    【参考情報】
    • <https://example.com/article>
    • <https://another.com/page>
    ```

## 重要な注意事項:
- タスク結果に含まれる情報のみを使用してください
- タスク結果にない情報は推測しないでください
- 情報が不足している場合は、その旨を明記してください
- 情報源URLは必ず記載してください（タスク結果にURLが含まれている場合）
""")

    human_message = HumanMessage(content=f"""
## ユーザーの質問:
{user_query}

## タスクの実行結果:
{task_results_text}

上記のタスク結果を統合して、ユーザーの質問に対する包括的な回答を生成してください。
""")

    try:
        model = get_model()
        response = await model.ainvoke([system_message, human_message])

        return Command(
            update={
                "messages": [AIMessage(content=response.content)],
                "final_answer": response.content
            },
            goto=END
        )

    except Exception as e:
        logger.error(f"generate_final_answerでエラーが発生しました: {str(e)}", exc_info=True)
        raise