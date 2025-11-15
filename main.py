import asyncio
from src.graph.builder import create_graph
from langchain_core.messages import HumanMessage


async def main():
    app = create_graph()

    # 初期状態の準備
    initial_state = {
        "messages": [HumanMessage(content="今日の東京と埼玉の天気を教えて")],
        "attempt": 0
    }

    # エージェントの実行
    try:
        result = await app.ainvoke(initial_state)
        print(result["final_answer"])
    except Exception as e:
        print(f"\nエージェント実行中にエラーが発生しました: {e}")


if __name__ == "__main__":
    asyncio.run(main())