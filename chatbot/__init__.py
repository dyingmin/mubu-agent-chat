"""
Agent/CustomerAgent/custom/ - 自定义 Agent 实现（无 Agno 依赖）

包含：
- agent_config: Agent 配置管理
- llm_client: LLM 客户端封装
- message_builder: 消息和 Prompt 构建器
- tool_executor: 工具执行器
- tool_decorator: 工具系统（装饰器 + 注册表）
- session_manager: 会话历史管理（SQLite + 上下文压缩）
- customer_agent: 自定义 CustomerAgent 实现（主入口）
"""

from chatbot_agent.agent_config import (
    AgentConfig,
    DEFAULT_DB_PATH,
    DEFAULT_TOKEN_WINDOW,
    DEFAULT_COMPRESS_RATIO,
    DEFAULT_RETAIN_COUNT,
    DEFAULT_MAX_LOOPS,
    DEFAULT_TEMPERATURE,
)

from chatbot_agent.llm_client import (
    LLMClient,
    LLMResponse,
)

from chatbot_agent.message_builder import MessageBuilder

from chatbot_agent.tool_executor import (
    ToolExecutor,
    ToolResult,
)

from chatbot_agent.tool_decorator import (
    agent_tool,
    TOOL_REGISTRY,
    get_tools_for_llm,
    get_tool_entry,
    execute_tool,
    ToolEntry,
)

from chatbot_agent.session_manager import (
    SessionManager,
    AgentMessage,
    TokenEstimator,
)

from chatbot_agent.chatbot_agent import CustomerAgent

__all__ = [
    # 配置模块
    "AgentConfig",
    "DEFAULT_DB_PATH",
    "DEFAULT_TOKEN_WINDOW",
    "DEFAULT_COMPRESS_RATIO",
    "DEFAULT_RETAIN_COUNT",
    "DEFAULT_MAX_LOOPS",
    "DEFAULT_TEMPERATURE",
    # LLM 客户端
    "LLMClient",
    "LLMResponse",
    # 消息构建器
    "MessageBuilder",
    # 工具执行器
    "ToolExecutor",
    "ToolResult",
    # 工具装饰器
    "agent_tool",
    "TOOL_REGISTRY",
    "get_tools_for_llm",
    "get_tool_entry",
    "execute_tool",
    "ToolEntry",
    # 会话管理
    "SessionManager",
    "AgentMessage",
    "TokenEstimator",
    # 主 Agent
    "CustomerAgent",
]
