from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import AnyMessage, add_messages

class Task(TypedDict):
    task_id: str
    task_description: str
    task_result: str

def update_task(existing: list[Task], new: list[Task]) -> list[Task]:
    """タスクリストを更新"""
    if not new:
        return existing

    new_task_ids = {task['task_id'] for task in new}

    # 既存タスクのうち、更新されないものを保持
    updated = [task for task in existing if task['task_id'] not in new_task_ids]

    # 新しいタスクを追加
    updated.extend(new)

    return updated

class BaseState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tasks: Annotated[list[Task], update_task]
    final_answer: Optional[str]