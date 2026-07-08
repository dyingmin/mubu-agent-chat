"""
聊天机器人实现模块

完全自主实现，不依赖 Agno 框架。

本模块已重构，职责分离为：
- agent_config.py: 配置管理
- llm_client.py: LLM 客户端封装
- message_builder.py: 消息和 Prompt 构建
- session_manager.py: 会话历史管理
- tool_decorator.py / tool_executor.py: 工具注册与执行
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from types import SimpleNamespace

try:
    from bridge.context import Context
except ImportError:  # pragma: no cover - optional runtime dependency
    Context = object  # type: ignore[assignment]

try:
    from bridge.reply import Reply, ReplyType
except ImportError:  # pragma: no cover - optional runtime dependency
    from dataclasses import dataclass

    @dataclass
    class Reply:
        type: str
        content: str

    class ReplyType:
        TEXT = "text"

from chatbot_agent.session_manager import SessionManager
from utils.logger_loguru import get_logger

# 导入重构后的模块
from chatbot_agent.agent_config import (
    AgentConfig,
    DEFAULT_DB_PATH,
    DEFAULT_TOKEN_WINDOW,
    DEFAULT_COMPRESS_RATIO,
    DEFAULT_RETAIN_COUNT,
    DEFAULT_TEMPERATURE,
)
from chatbot_agent.llm_client import LLMClient
from chatbot_agent.message_builder import MessageBuilder
from chatbot_agent.tool_decorator import get_tools_for_llm
from chatbot_agent.tool_executor import ToolExecutor

logger = get_logger("ChatBot")


class ChatBot:
    """
    聊天机器人。

    核心流程：
    1. 加载历史消息
    2. 检查上下文压缩
    3. 构建 messages 列表
    4. 调用 LLM 决定是否使用知识库工具
    5. 如果触发工具，则直接返回工具结果
    6. 保存 user / assistant 到历史
    7. 返回最终回复
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        token_window: int = DEFAULT_TOKEN_WINDOW,
        compress_ratio: float = DEFAULT_COMPRESS_RATIO,
        retain_count: int = DEFAULT_RETAIN_COUNT,
        temperature: float = DEFAULT_TEMPERATURE,
        extra_instructions: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        self._is_initialized = False

        # 配置参数
        self._config = AgentConfig(
            db_path=db_path or DEFAULT_DB_PATH,
            token_window=token_window,
            compress_ratio=compress_ratio,
            retain_count=retain_count,
            max_loops=1,
            temperature=temperature,
        )

        # 额外提示词，用于知识库工具、学习辅导等场景
        self._extra_instructions = extra_instructions or []

        # 工具列表：如果外部未显式提供，则由全局注册表获取
        self._tools = tools

        # 子组件（延迟初始化）
        self._llm_client: Optional[LLMClient] = None
        self._message_builder: Optional[MessageBuilder] = None
        self._session_manager: Optional[SessionManager] = None
        self._tool_executor: Optional[ToolExecutor] = None

        # 运行时状态
        self._last_source: str = "llm"
        self._last_tool_results: List[Dict[str, Any]] = []

        logger.info("聊天机器人实例创建成功")

    @property
    def last_source(self) -> str:
        """返回最近一次回复的来源，llm 或 knowledge_base。"""
        return self._last_source

    @property
    def last_tool_results(self) -> List[Dict[str, Any]]:
        """返回最近一次工具执行结果。"""
        return self._last_tool_results

    async def initialize_async(self) -> bool:
        """异步初始化聊天机器人"""
        if self._is_initialized:
            return True

        try:
            # 1. 从配置文件加载配置
            self._config = AgentConfig.load_from_config()

            # 2. 验证配置
            if not self._config.validate():
                return False

            # 3. 初始化工具列表
            tools = self._tools if self._tools is not None else get_tools_for_llm()
            self._tools = tools or []
            self._tool_executor = ToolExecutor() if self._tools else None

            # 4. 初始化 LLM 客户端
            self._llm_client = LLMClient(
                api_key=self._config.api_key,
                api_base=self._config.api_base,
                model_name=self._config.model_name,
                temperature=self._config.temperature,
                tools=self._tools,
            )
            await self._llm_client.initialize()

            # 5. 初始化会话管理器
            self._session_manager = SessionManager(
                db_path=self._config.db_path,
                token_window=self._config.token_window,
                compress_ratio=self._config.compress_ratio,
                retain_count=self._config.retain_count,
                model_name=self._config.model_name,
            )

            # 6. 初始化消息构建器
            combined_instructions = list(self._config.instructions or [])
            combined_instructions.extend(self._extra_instructions)
            if self._tools:
                combined_instructions.append(
                    "遇到本地知识库、项目说明、文档内容相关的问题时，优先调用知识库工具获取答案。"
                )
            self._message_builder = MessageBuilder(
                instructions=combined_instructions,
            )

            self._is_initialized = True
            logger.info(f"聊天机器人初始化成功: model={self._config.model_name}")
            return True

        except Exception as e:
            logger.error(f"聊天机器人初始化失败: {e}")
            return False

    def _get_session_id(self, context: Context = None) -> str:
        """获取稳定的聊天会话 ID"""
        if context and getattr(context, "kwargs", None):
            kwargs = context.kwargs

            session_id = getattr(kwargs, "session_id", None)
            if session_id:
                return str(session_id)

            user_id = getattr(kwargs, "user_id", None)
            if user_id:
                return f"user_{user_id}"

        return "default_chat"

    async def async_reply(self, query: str, context: Context = None) -> Reply:
        """异步聊天回复接口"""
        if not self._is_initialized:
            if not await self.initialize_async():
                return Reply(ReplyType.TEXT, "聊天机器人初始化失败，请检查配置。")

        self._last_source = "llm"
        self._last_tool_results = []

        try:
            session_id = self._get_session_id(context)
            dependencies: Dict[str, Any] = {}
            if context:
                dependencies = self._message_builder.build_dependencies(context)

            # 加载历史并检查压缩
            history = self._session_manager.get_history(session_id)
            if self._session_manager.should_compress(session_id):
                logger.info(f"触发上下文压缩: session_id={session_id}")
                await self._compress_with_llm(session_id, history)
                history = self._session_manager.get_history(session_id)

            # 构建 messages
            messages = self._message_builder.build_messages(query, history, dependencies)

            # 先让 LLM 判断是否需要调用知识库工具
            response = await self._llm_client.chat(
                messages,
                tool_choice="auto" if self._tools else "none",
            )

            # 方案 A：工具直接返回最终答案。
            # 如果 LLM 触发了工具调用，就直接执行工具并把结果作为最终回复。
            if response.has_tool_calls and self._tool_executor:
                tool_results = await self._tool_executor.execute_parallel(
                    response.tool_calls or [],
                    dependencies,
                )
                self._last_tool_results = [item.to_dict() for item in tool_results]
                self._last_source = "knowledge_base"

                final_content = "\n\n".join(
                    item.content.strip() for item in tool_results if item.content.strip()
                ).strip()
                if not final_content:
                    final_content = "知识库暂时没有返回可用答案。"
            else:
                final_content = response.content or ""
                self._last_source = "llm"

            # 保存用户与助手消息到历史，形成多轮上下文
            self._session_manager.add_message(
                session_id=session_id,
                role="user",
                content=query,
            )
            self._session_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=final_content,
            )

            return Reply(ReplyType.TEXT, final_content or "抱歉，我暂时无法回复。")

        except Exception as e:
            logger.error(f"聊天机器人回复失败: {e}")
            return Reply(ReplyType.TEXT, "抱歉，我现在无法回复，请稍后再试。")

    async def _compress_with_llm(
        self,
        session_id: str,
        history: List[Dict[str, Any]],
    ) -> None:
        """使用 LLM 生成摘要并压缩历史"""

        def summary_llm(messages: List[Dict[str, Any]]) -> str:
            """同步调用 LLM 生成摘要"""
            summary_prompt = (
                "请简洁地总结以下对话的要点，保留关键信息和用户意图。\n\n"
                f"对话内容（共 {len(messages)} 条消息）：\n"
                + "\n".join(
                    f"[{msg.get('role', 'unknown')}]: {msg.get('content', '')[:200]}"
                    for msg in messages
                    if msg.get("content")
                )
            )

            try:
                response = asyncio.run(
                    self._llm_client.chat(
                        messages=[
                            {"role": "system", "content": "你是一个对话摘要助手。请简洁地总结对话要点。"},
                            {"role": "user", "content": summary_prompt},
                        ],
                        tool_choice="none",
                    )
                )
                return response.content or "[摘要生成失败]"
            except RuntimeError:
                return "[对话历史摘要]"
            except Exception:
                return "[摘要生成失败]"

        self._session_manager.compress_history(session_id, summary_llm)


# 兼容旧名字，避免外部导入受影响
CustomerAgent = ChatBot
