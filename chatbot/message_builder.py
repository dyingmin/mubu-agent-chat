"""
聊天机器人消息构建器模块

负责构建系统 Prompt 和 LLM 消息列表。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from bridge.context import Context


class MessageBuilder:
    """消息构建器"""

    def __init__(
        self,
        instructions: Optional[List[str]] = None,
    ):
        """
        初始化消息构建器

        Args:
            instructions: 指令列表
        """
        self.instructions = instructions or []
        self.system_prompt = ""

        self._build_system_prompt()

    def _build_system_prompt(self) -> None:
        """构建系统 Prompt"""
        parts = []

        description = """你是一个学习辅导与答疑助手。

你的目标：
- 使用中文进行自然、多轮对话
- 重点帮助用户学习、理解知识、解答疑问
- 回答清晰、友好、简洁
- 如果不确定答案，请明确说明，不要编造
- 根据上下文连续追问或补充解释
- 尽量把复杂问题讲清楚，并给出必要的步骤或例子
"""
        parts.append(description)

        if self.instructions:
            parts.append("---\n" + "\n".join(f"- {i}" for i in self.instructions))

        self.system_prompt = "\n\n".join(parts) if parts else "你是一个智能聊天机器人。"

    def build_dependencies(self, context: Context) -> Dict[str, str]:
        """
        从 Context 构建 dependencies 字典

        Args:
            context: 上下文对象

        Returns:
            dependencies 字典
        """
        kwargs = getattr(context, "kwargs", None)
        if not kwargs:
            return {}

        session_id = getattr(kwargs, "session_id", None)
        user_id = getattr(kwargs, "user_id", None)

        dependencies: Dict[str, str] = {}
        if session_id:
            dependencies["session_id"] = str(session_id)
        if user_id:
            dependencies["user_id"] = str(user_id)

        return dependencies

    def build_messages(
        self,
        query: str,
        history: List[Dict[str, str]],
        dependencies: Dict[str, str] = None,
    ) -> List[Dict[str, str]]:
        """
        构建 LLM 消息列表

        Args:
            query: 用户查询
            history: 历史消息
            dependencies: 依赖字典（保留接口兼容，当前不做占位符替换）

        Returns:
            LLM 消息列表
        """
        messages: List[Dict[str, str]] = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            tool_call_id = msg.get("tool_call_id")

            if not content and role != "tool":
                continue

            if role == "tool":
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content or "",
                }
                messages.append(tool_msg)
            else:
                messages.append({"role": role, "content": content or ""})

        messages.append({"role": "user", "content": query})
        return messages
