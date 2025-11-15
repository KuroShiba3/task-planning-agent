import asyncio
import re
from datetime import datetime
from typing import Literal, Optional

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langchain_google_community import GoogleSearchAPIWrapper
from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from ...config.config import GOOGLE_API_KEY, GOOGLE_CX
from ...config.model import get_model
from ...utils.logger import get_logger
from .state import WebSearchState

logger = get_logger(__name__)

async def generate_search_queries(state: WebSearchState) -> WebSearchState:
    """ユーザーの質問から最適な検索クエリを生成するノード（最大2個）"""

    class SearchQueries(BaseModel):
        queries: list[str] = Field(description="生成された検索クエリのリスト（最大2個）", max_length=2)
        reason: str = Field(description="これらのクエリを選んだ理由")

    task_description = state.get("task_description", "")
    previous_queries = state.get("search_queries", [])
    feedback = state.get("feedback")

    system_message = SystemMessage(
        content=f"""
あなたは検索クエリ生成の専門家です。割り当てられたタスクに答えるために最適な検索クエリを生成してください。

## 現在の日付:
{datetime.now().strftime("%Y年%m月%d日")}

## クエリ生成のルール:

1. **複数の視点から検索**:
    - 異なる角度から情報を集めるため、1-2個のクエリを生成
    - 重複する内容のクエリは避ける

2. **具体的で明確なクエリ**:
    - 曖昧な表現を避け、固有名詞を使う

3. **時間的文脈の考慮**:
    - タスクが「本日」「今日」を含む場合 → 必ず日付を含める
    - 過去の情報が必要な場合 → 具体的な期間を指定
    - 最新情報が必要な場合 → "最新"や年月を含める

4. **タスク内容の活用**:
    - タスクの要求を正確に理解する
    - 代名詞（「それ」「この」など）がある場合は具体的な名詞に変換
    - 文脈から暗黙の情報を補完

5. **検索エンジン最適化**:
    - 自然な日本語で、検索エンジンが理解しやすい形式
    - キーワードの組み合わせを工夫

## 重要な注意事項:
- 必ず1-2個のクエリを生成してください（1個でも2個でも可）
- タスクの要求を正確に理解してクエリを生成してください
- 前回のクエリと異なる角度からの検索を心がけてください
"""
    )

    human_content_parts = [f"## 割り当てられたタスク:\n{task_description}"]

    if previous_queries:
        queries_text = "\n".join([f"- {q}" for q in previous_queries])
        human_content_parts.append(f"\n## すでに利用した検索クエリ:\n{queries_text}")
        human_content_parts.append("\n**重要**: 前回の検索で十分な結果が得られなかったため、異なる角度からの新しいクエリを生成してください。")

    if feedback:
        human_content_parts.append(f"\n## 改善フィードバック:\n{feedback}")
        human_content_parts.append("\n上記のフィードバックを参考にしてください。")

    human_message = HumanMessage(content="".join(human_content_parts))

    try:
        model = get_model()
        search_queries_result = await model.with_structured_output(SearchQueries).ainvoke(
            [system_message, human_message]
        )

        if not search_queries_result.queries:
            logger.warning("generate_search_queriesでクエリが生成されませんでした")
            return Command(
                update={"task_result": "適切な検索クエリを生成できませんでした。"},
                goto="evaluate_task_result"
            )

        sends = [
            Send("execute_search", {"query": query})
            for query in search_queries_result.queries
        ]

        return Command(
            update={
                "search_queries": search_queries_result.queries,
            },
            goto=sends
        )
    except Exception as e:
        logger.error(f"generate_search_queriesでエラーが発生しました: {str(e)}", exc_info=True)
        raise

async def execute_search(arg: dict) -> dict:
    """単一の検索クエリを実行するノード"""

    def clean_text(text: str) -> str:
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        return '\n'.join(lines)

    query = arg.get("query", "")
    if not query:
        return Command(update={"search_results": []}, goto="generate_task_result")

    try:
        num_results = 2

        search = GoogleSearchAPIWrapper(
            google_api_key=GOOGLE_API_KEY,
            google_cse_id=GOOGLE_CX
        )

        results = search.results(query, num_results=num_results)

        if not results:
            return Command(update={"search_results": []}, goto="generate_task_result")

        search_results = []
        for result in results:
            url = result['link']
            title = result['title']
            snippet = result.get('snippet', '')

            try:
                loader = WebBaseLoader(url)
                load_task = asyncio.to_thread(loader.load)
                docs = await asyncio.wait_for(load_task, timeout=15.0)

                raw_content = docs[0].page_content
                cleaned_content = clean_text(raw_content)
                content = cleaned_content
                search_results.append({
                    "query": query,
                    "title": title,
                    "url": url,
                    "content": content[:2500],
                    "snippet": snippet
                })
            except Exception as e:
                logger.warning(f"execute_searchでWebページ取得エラー: {str(e)}", exc_info=True)
                search_results.append({
                    "query": query,
                    "title": title,
                    "url": url,
                    "snippet": snippet
                })

        return Command(
            update={"search_results": search_results},
            goto="generate_task_result"
        )
    except Exception as e:
        logger.error(f"execute_searchでエラーが発生しました: {str(e)}", exc_info=True)
        return Command(update={"search_results": []}, goto="generate_task_result")

async def generate_task_result(state: WebSearchState) -> WebSearchState:
    """検索結果を元にタスク結果を生成するノード"""

    task_description = state.get("task_description", "")
    search_results = state.get("search_results", [])
    feedback = state.get("feedback")

    system_message = SystemMessage(
        content=f"""
あなたはタスク実行エージェントです。以下の検索結果を元に、割り当てられたタスクの結果をまとめてください。

## 現在の日付:
{datetime.now().strftime("%Y年%m月%d日")}

## システムアーキテクチャの理解:
このシステムは以下の3段階で動作します:
1. **タスク計画**: ユーザーの質問を複数のタスクに分割
2. **タスク実行（あなたの役割）**: 各タスクについて検索を実行し、結果をまとめる
3. **回答生成**: すべてのタスク結果を統合してユーザーに最終回答を提示

**重要**: 回答生成エージェントは検索結果を直接見ることができません。あなたのタスク結果のみを参照します。
そのため、次のエージェントが適切に理解・利用できる内容にする必要があります。

## タスク結果作成のルール:

1. **検索結果のみを使用**:
    - 検索結果に含まれる情報のみを使ってタスク結果をまとめる
    - 検索結果にない情報は推測しない

2. **次のエージェントが理解できる内容にする**:
    - **数字、日付、固有名詞、統計データなど具体的な情報を含める**
    - **専門用語や略語がある場合、検索結果に説明があれば簡潔に補足する**
    - **文脈理解に必要な背景情報があれば含める**
    - タスクに関連する重要な情報を漏らさず含める
    - 例:
        - 「気温は25度」ではなく「最高気温25度、最低気温18度」
        - 「GDP成長率は2.1%」だけでなく「2024年のGDP成長率は前年比2.1%増加」
        - 専門用語がある場合「LLM（大規模言語モデル）」のように補足

3. **タスク結果の構成**:
    - タスクの要求に対する直接的な答えを述べる
    - 必要に応じて補足情報を追加
    - 自然な文章で記述する

4. **情報源の記載形式（必須）**:
    - タスク結果で実際に利用した情報源のURLのみを記載する
    - 利用していないURLは記載しない
    - 同じドメインのURLが複数ある場合は、代表的なURL1つのみを記載する
    - フォーマット:
    ```
    （タスク結果の本文）

    【参考情報】
    • <https://example.com/article>
    • <https://another.com/page>
    ```

5. **不足情報への対応**:
    - 検索結果が不完全な場合は、その旨を明記
    - 得られた情報の範囲で最大限タスク結果をまとめる

## 重要な注意事項:
- **回答生成エージェントが検索結果を見ずに理解できるよう、文脈を含めて記述してください**
- このタスク結果は最終的に他のタスク結果と統合されてユーザーに提示されます
- タスクの要求に直接関係する情報のみを含めてください
- 検索結果にない情報は「検索結果には含まれていません」と明記
- 参照URLは実際にタスク結果に使用したもののみを記載してください
- すべての詳細を含める必要はなく、次のエージェントが適切に利用できる情報量を心がけてください
"""
    )

    human_content_parts = [f"## 割り当てられたタスク:\n{task_description}"]

    if search_results:
        human_content_parts.append("\n## 取得した検索結果:")
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", result.get("snippet", ""))
            query = result.get("query", "")

            human_content_parts.append(f"\n### 検索結果 {i}")
            human_content_parts.append(f"\n**検索クエリ**: {query}")
            human_content_parts.append(f"\n**タイトル**: {title}")
            human_content_parts.append(f"\n**URL**: {url}")
            human_content_parts.append(f"\n**内容**:\n{content}\n")

    if feedback:
        previous_result = state.get("task_result", "")
        human_content_parts.append(f"\n## 改善フィードバック:\n{feedback}")
        if previous_result:
            human_content_parts.append(f"\n## 以前のタスク結果:\n{previous_result}")
        human_content_parts.append("\n**重要**: 上記のフィードバックを参考にして、より良いタスク結果を作成してください。")

    human_message = HumanMessage(content="".join(human_content_parts))

    try:
        model = get_model()
        answer = await model.ainvoke([system_message, human_message])

        return Command(
            update={
                "task_result": answer.content
            },
            goto="evaluate_task_result"
        )
    except Exception as e:
        logger.error(f"generate_task_resultでエラーが発生しました: {str(e)}", exc_info=True)
        raise

async def evaluate_task_result(state: WebSearchState) -> dict:
    """生成されたタスク結果を評価し、再検索や結果改善が必要か判断するノード"""

    class TaskEvaluation(BaseModel):
        is_satisfactory: bool = Field(description="タスク結果がタスクの要求に十分答えているかどうか")
        need: Optional[Literal["search", "generate"]] = Field(description="改善が必要な場合、どの部分の改善が必要か。search: 検索クエリや検索結果の改善、generate: タスク結果の改善。改善不要ならNone。")
        reason: str = Field(description="上記の判断理由。is_satisfactoryの判断理由、または改善が必要な場合はその理由を記述。")
        feedback: Optional[str] = Field(description="改善が必要な場合（needがNoneでない場合）の具体的なフィードバック。searchならクエリに関するアドバイス、generateならタスク結果に関するアドバイス。")

    task_description = state.get("task_description", "")
    task_result = state.get("task_result", "")
    search_queries = state.get("search_queries", [])
    search_results = state.get("search_results", [])

    attempt = state.get("attempt", 0)
    attempt += 1

    system_message = SystemMessage(
        content=f"""
あなたはタスク結果品質を評価する専門家です。検索結果と生成されたタスク結果を比較し、2つの観点から評価してください。

## 現在の日付:
{datetime.now().strftime("%Y年%m月%d日")}

## 評価の流れ:
以下の順序で評価を行ってください:

### 1. 検索結果の確認 (need = "search" かどうか)
まず検索結果を詳細に確認し、タスクに答えるための情報が含まれているかを判断してください。

**need = "search" (検索改善が必要):**
- **検索結果にタスクに答えるための情報が全く含まれていない**
- 検索クエリが明らかに不適切（タスクと無関係なクエリ）
- 検索結果がタスクと全く関連性がない
- 重要な情報が検索できていない（異なる角度からの検索で改善できそう）

### 2. タスク結果の確認 (need = "generate" かどうか)
検索結果が十分な場合、次に検索結果とタスク結果を比較し、適切に活用されているかを判断してください。

**need = "generate" (タスク結果改善が必要):**
- **検索結果にはタスクに答える情報があるのに、タスク結果でその情報を活用できていない**
- **検索結果の重要な情報がタスク結果に含まれていない**
- タスク結果の構成や表現が分かりにくい
- 検索結果を羅列しているだけで、自然な文章になっていない
- タスクに直接関係ない情報が大量に含まれている

### 3. 全体的な満足度 (is_satisfactory)
検索とタスク結果の両方が適切な場合、最終的に満足できるかを判断してください。

**need = None (改善不要):**
- **検索結果に含まれる重要な情報がタスク結果に適切に反映されている**
- タスク結果が自然な文章で構成されている
- タスクの要求に焦点を絞り、簡潔にまとめている

**is_satisfactory:**
- need が None の場合のみ True
- need が "search" または "generate" の場合は False

### 4. 判断理由 (reason)
- 上記の判断理由を具体的に記述してください

### 5. フィードバック (feedback)
- **need = "search" の場合**: 検索クエリに関するアドバイス（「どのようなキーワードで検索すべきか」「どの角度から検索すべきか」など）
- **need = "generate" の場合**: タスク結果に関するアドバイス（「どの情報を追加すべきか」「どう表現を改善すべきか」など）
- **need = None の場合**: None

## 重要な注意事項:
- **優先順位**: 検索結果に問題がある場合は need = "search"、検索結果は十分だがタスク結果に問題がある場合は need = "generate"
- **is_satisfactory は need が None の場合のみ True にしてください**
- **reasonとfeedbackは具体的で実行可能な内容にしてください**
"""
    )

    # HumanMessageで動的な値を渡す
    human_content_parts = [f"## 割り当てられたタスク:\n{task_description}"]

    # 検索クエリを追加
    if search_queries:
        queries_text = "\n".join([f"- {q}" for q in search_queries])
        human_content_parts.append(f"\n## 実行した検索クエリ:\n{queries_text}")

    # 検索結果を追加
    if search_results:
        human_content_parts.append("\n## 取得した検索結果:")
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", result.get("snippet", ""))
            query = result.get("query", "")

            human_content_parts.append(f"\n### 検索結果 {i}")
            human_content_parts.append(f"\n**検索クエリ**: {query}")
            human_content_parts.append(f"\n**タイトル**: {title}")
            human_content_parts.append(f"\n**URL**: {url}")
            human_content_parts.append(f"\n**内容**:\n{content}\n")

    # タスク結果を追加
    human_content_parts.append(f"\n## 生成されたタスク結果:\n{task_result}")

    human_message = HumanMessage(content="".join(human_content_parts))

    try:
        model = get_model()
        evaluation = await model.with_structured_output(TaskEvaluation).ainvoke(
            [system_message, human_message]
        )

        if evaluation.is_satisfactory or attempt >= 2:
            task_id = state.get("task_id", "")
            return Command(
                update={
                    "attempt": attempt,
                    "completed": True,
                    "tasks": [{
                        "task_id": task_id,
                        "task_description": task_description,
                        "task_result": task_result
                    }]
                },
                goto=END
            )

        if evaluation.need == "search":
            return Command(
                update={
                    "attempt": attempt,
                    "search_results": [],
                    "feedback": evaluation.feedback
                },
                goto="generate_search_queries"
            )
        elif evaluation.need == "generate":
            return Command(
                update={
                    "attempt": attempt,
                    "feedback": evaluation.feedback
                },
                goto="generate_task_result"
            )
        else:
            # 念の為（is_satisfactoryがFalseなのにneedがNoneの場合）
            logger.warning(f"evaluate_task_result: is_satisfactory=False but need=None. Completing task anyway.")
            task_id = state.get("task_id", "")
            return Command(
                update={
                    "attempt": attempt,
                    "completed": True,
                    "tasks": [{
                        "task_id": task_id,
                        "task_description": task_description,
                        "task_result": task_result
                    }]
                },
                goto=END
            )
    except Exception as e:
        logger.error(f"evaluate_task_resultでエラーが発生しました: {str(e)}", exc_info=True)
        raise