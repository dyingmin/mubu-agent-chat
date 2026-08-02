"""
聊天机器人消息构建器模块

负责构建系统 Prompt 和 LLM 消息列表。

Prompt 采用分层架构，便于后续扩展多工具：
- 基础身份层：定义 Agent 角色
- 全局行为规则：通用约束
- 工具使用规则：工具决策与结果处理
- 知识库工具专用规则：RAG 场景约束
- 回答生成规则：输出格式与引用要求
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

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
        """构建系统 Prompt（分层架构）"""
        parts: List[str] = [
            self._base_identity_prompt(),
            self._global_behavior_prompt(),
            self._tool_usage_prompt(),
            self._knowledge_base_tool_prompt(),
            self._answer_generation_prompt(),
        ]

        if self.instructions:
            parts.append("---\n额外指令\n" + "\n".join(f"- {i}" for i in self.instructions))

        self.system_prompt = "\n\n".join(parts)

    @staticmethod
    def _base_identity_prompt() -> str:
        return (
            "你是一个可以使用多种工具完成任务的智能助手。"
            "你需要根据用户问题判断是否需要调用工具，"
            "并基于工具返回结果生成清晰、有条理的最终回答。"
        )

    @staticmethod
    def _global_behavior_prompt() -> str:
        return """# 全局行为规则

- 使用中文进行自然、多轮对话
- 准确理解用户问题，优先回答用户真正想解决的问题
- 如果问题信息不足，先提出必要的澄清问题
- 如果可以直接回答，则直接回答
- 如果需要本地资料、实时信息、私有数据等外部信息，必须调用对应工具
- 不编造信息，不伪造工具结果，不声称调用了未实际调用的工具
- 工具结果与模型已有知识冲突时，以工具结果为准，并说明依据来自工具
- 回答清晰、友好、简洁，避免机械罗列工具返回的原始内容"""

    @staticmethod
    def _tool_usage_prompt() -> str:
        return """# 工具使用规则

- 只有在需要外部信息、检索、查询时才调用工具
- 选择最能解决问题的工具，不要重复调用同一工具
- 缺少必要参数时，先向用户追问
- 工具失败、无结果或结果不足时，应如实说明
- 工具结果只是上下文，不是最终答案
- 必须对工具结果进行理解、筛选、归纳和组织
- 多个工具结果互相补充时应综合回答，冲突时说明并优先采用更权威的结果"""

    @staticmethod
    def _knowledge_base_tool_prompt() -> str:
        return """# 知识库工具专用规则

当使用知识库检索工具时：
- 必须严格基于知识库上下文回答，不得使用上下文之外的事实补充
- 如果上下文中没有答案，回答："知识库中没有找到相关内容。"
- 回答要清晰、有条理，优先使用"结论 + 说明 + 注意事项"的结构
- 不要复述大段原文，不要罗列检索片段
- 回答结束后必须列出使用的来源，格式如下：
  来源：
  - [1] 标题路径：...
    来源文件：...
  如有多个来源，按条目依次列出"""

    @staticmethod
    def _answer_generation_prompt() -> str:
        return """# 回答生成规则

默认输出结构：
1. 结论
2. 关键依据 / 说明
3. 如有必要，给出步骤、建议或注意事项
4. 如使用了需要引用的工具，最后列出来源或依据

如果没有足够信息，直接说明原因，并告诉用户可以补充什么信息。用户要求特定格式时，优先遵循用户格式。"""

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

    def build_final_answer_messages(
        self,
        query: str,
        tool_results: List[Dict[str, Any]],
        history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        基于工具检索结果构建"二次生成"消息列表

        在工具执行完成后，把检索上下文交给 LLM，让它生成有条理的最终回答。

        Args:
            query: 用户原始问题
            tool_results: 工具执行结果列表（含 role/tool_call_id/content）
            history: 会话历史

        Returns:
            用于二次生成回答的 LLM 消息列表
        """
        context_blocks: List[str] = []
        for i, result in enumerate(tool_results, start=1):
            content = (result.get("content") or "").strip()
            if content:
                context_blocks.append(f"--- 片段 {i} ---\n{content}")
        knowledge_context = (
            "\n\n".join(context_blocks) if context_blocks else "（知识库未返回有效内容）"
        )

        user_prompt = f"""请严格基于以下知识库上下文回答用户问题。

用户问题：
{query}

知识库上下文：
{knowledge_context}

输出要求：
1. 严格依据上下文回答，不得编造，不得使用上下文之外的事实
2. 回答要清晰、有条理，避免直接罗列原文片段
3. 如果上下文中没有答案，请明确说明"知识库中没有找到相关内容"
4. 回答结束后列出来源，格式如下：
   来源：
   - [1] 标题路径：...
     来源文件：...
"""

        messages: List[Dict[str, str]] = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # 保留最近的历史上下文，便于多轮对话
        for msg in history[-6:]:
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_prompt})
        return messages
