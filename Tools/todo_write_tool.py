"""In-memory planning tool for multi-step agent tasks."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from chatbot_agent.tool_decorator import agent_tool
from utils.logger_loguru import get_logger

logger = get_logger("TodoWriteTool")
TodoStatus = Literal["pending", "in_progress", "completed"]
CURRENT_TODOS_BY_SESSION: dict[str, list[dict[str, str]]] = {}


class TodoItem(BaseModel):
    content: str = Field(description="任务内容")
    status: TodoStatus = Field(description="任务状态")


class TodoWriteParams(BaseModel):
    todos: list[TodoItem] = Field(description="当前任务列表")
    session_id: str | None = Field(default=None, description="当前会话 ID")


@agent_tool(
    name="todo_write",
    description=(
        "创建或更新当前任务计划。复杂任务开始前先规划，执行过程中更新状态。"
        "每个任务必须包含 content 和 status；status 只能是 pending、in_progress、completed。"
        "该工具只记录计划和显示进度，不执行具体任务。"
    ),
    param_model=TodoWriteParams,
)
def todo_write(params: TodoWriteParams) -> str:
    session_id = params.session_id or "default_chat"
    todos = [{"content": item.content, "status": item.status} for item in params.todos]
    CURRENT_TODOS_BY_SESSION[session_id] = todos

    icons = {"pending": " ", "in_progress": ">", "completed": "✓"}
    lines = [f"\n## Current Tasks ({session_id})"]
    lines.extend(f"  [{icons[item['status']]}] {item['content']}" for item in todos)
    output = "\n".join(lines)
    print(output)
    logger.info(output)
    return f"Updated {len(todos)} tasks"


def get_current_todos(session_id: str = "default_chat") -> list[dict[str, str]]:
    return [dict(item) for item in CURRENT_TODOS_BY_SESSION.get(session_id, [])]
