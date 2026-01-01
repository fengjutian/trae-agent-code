# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""LLM Client wrapper for OpenAI, Anthropic, Azure, and OpenRouter APIs.

该模块提供了一个统一的LLM客户端接口，支持多个LLM提供商。
通过LLMClient类，可以方便地切换不同的LLM提供商，而无需修改应用代码。
支持的提供商包括：OpenAI、Anthropic、Azure、Ollama、OpenRouter、Doubao、Google、DeepSeek。

设计模式:
- 工厂模式：根据配置自动创建对应的客户端实例
- 策略模式：不同提供商实现相同的接口，便于切换和扩展
- 适配器模式：将不同提供商的API适配到统一的接口

使用示例:
    >>> from trae_agent.utils.config import ModelConfig
    >>> from trae_agent.utils.llm_clients.llm_client import LLMClient
    >>> 
    >>> # 配置OpenAI模型
    >>> config = ModelConfig(...)
    >>> client = LLMClient(config)
    >>> 
    >>> # 发送消息
    >>> response = client.chat(messages, config, tools)
"""

from enum import Enum

from trae_agent.tools.base import Tool
from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.base_client import BaseLLMClient
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse
from trae_agent.utils.trajectory_recorder import TrajectoryRecorder


class LLMProvider(Enum):
    """支持的LLM提供商枚举。

    定义了所有支持的LLM提供商及其标识符。
    每个提供商都有对应的客户端实现类。

    提供商说明:
        OPENAI: OpenAI官方API，支持GPT系列模型（GPT-3.5、GPT-4等）
        ANTHROPIC: Anthropic API，支持Claude系列模型（Claude-3、Claude-3.5等）
        AZURE: Azure OpenAI Service，微软托管的OpenAI服务
        OLLAMA: 本地部署的开源模型服务
        OPENROUTER: 路由服务，支持多种模型的统一接口
        DOUBAO: 字节跳动的Doubao（豆包）模型服务
        GOOGLE: Google AI API，支持Gemini等模型
        DEEPSEEK: DeepSeek API，支持DeepSeek系列模型

    使用示例:
        >>> provider = LLMProvider("openai")
        >>> provider == LLMProvider.OPENAI  # True
        >>> provider.value  # "openai"
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    DOUBAO = "doubao"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"


class LLMClient:
    """主要的LLM客户端类，支持多个提供商。

    该类是所有LLM交互的统一入口，根据配置自动创建对应的客户端实例。
    使用Python的match语句进行提供商识别和实例化，确保代码清晰易维护。

    主要功能:
    - 根据模型配置自动选择合适的LLM提供商
    - 提供统一的接口，隐藏不同提供商的差异
    - 支持轨迹记录，便于调试和分析
    - 支持消息历史管理，保持对话上下文
    - 支持工具调用，扩展LLM的能力

    设计特点:
    - 统一的接口：无论使用哪个提供商，API调用方式都相同
    - 易于扩展：添加新的提供商只需添加新的case分支
    - 延迟导入：只导入当前需要的客户端模块，减少启动时间
    - 类型安全：使用类型注解，提供更好的IDE支持

    使用示例:
        >>> from trae_agent.utils.config import ModelConfig
        >>> from trae_agent.utils.llm_clients.llm_client import LLMClient, LLMProvider
        >>> 
        >>> # 创建客户端（自动选择提供商）
        >>> config = ModelConfig(...)
        >>> client = LLMClient(config)
        >>> 
        >>> # 设置轨迹记录器
        >>> from trae_agent.utils.trajectory_recorder import TrajectoryRecorder
        >>> client.set_trajectory_recorder(TrajectoryRecorder(...))
        >>> 
        >>> # 设置聊天历史
        >>> client.set_chat_history(messages)
        >>> 
        >>> # 发送消息
        >>> response = client.chat(messages, config, tools)
        >>> 
        >>> # 检查是否支持工具调用
        >>> if client.supports_tool_calling(config):
        >>>     print("当前模型支持工具调用")
    """

    def __init__(self, model_config: ModelConfig):
        """初始化LLM客户端。

        根据模型配置中的provider字段，自动创建对应的客户端实例。
        使用延迟导入策略，只在需要时才导入特定提供商的客户端模块。

        参数:
            model_config: 模型配置对象，包含提供商信息、API密钥、模型名称等

        抛出:
            ValueError: 如果提供商名称无效或不支持

        注意:
            - 客户端实例创建时会根据provider自动选择对应的实现类
            - 如果provider不在支持的列表中，Python会抛出异常
            - 每个提供商都有独立的客户端实现，但都继承自BaseLLMClient

        支持的提供商:
            - OpenAI: 使用OpenAIClient
            - Anthropic: 使用AnthropicClient
            - Azure: 使用AzureClient
            - Ollama: 使用OllamaClient
            - OpenRouter: 使用OpenRouterClient
            - Doubao: 使用DoubaoClient
            - Google: 使用GoogleClient
            - DeepSeek: 使用DeepSeekClient
        """
        self.provider: LLMProvider = LLMProvider(model_config.model_provider.provider)
        self.model_config: ModelConfig = model_config

        # 根据提供商类型创建对应的客户端实例
        # 使用延迟导入策略，只导入当前需要的模块
        match self.provider:
            case LLMProvider.OPENAI:
                from .openai_client import OpenAIClient

                self.client: BaseLLMClient = OpenAIClient(model_config)
            case LLMProvider.ANTHROPIC:
                from .anthropic_client import AnthropicClient

                self.client = AnthropicClient(model_config)
            case LLMProvider.AZURE:
                from .azure_client import AzureClient

                self.client = AzureClient(model_config)
            case LLMProvider.OPENROUTER:
                from .openrouter_client import OpenRouterClient

                self.client = OpenRouterClient(model_config)
            case LLMProvider.DOUBAO:
                from .doubao_client import DoubaoClient

                self.client = DoubaoClient(model_config)
            case LLMProvider.OLLAMA:
                from .ollama_client import OllamaClient

                self.client = OllamaClient(model_config)
            case LLMProvider.GOOGLE:
                from .google_client import GoogleClient

                self.client = GoogleClient(model_config)
            case LLMProvider.DEEPSEEK:
                from .deepseek_client import DeepSeekClient

                self.client = DeepSeekClient(model_config)

    def set_trajectory_recorder(self, recorder: TrajectoryRecorder | None) -> None:
        """设置轨迹记录器。

        轨迹记录器用于记录所有LLM交互的详细信息，包括：
        - 发送的消息
        - 接收的响应
        - 工具调用
        - 使用的模型和提供商

        这些信息对于调试、分析和优化LLM应用非常重要。

        参数:
            recorder: 轨迹记录器实例，传入None表示取消记录

        使用场景:
        - 调试：查看详细的交互过程，定位问题
        - 分析：统计使用量、成本、性能指标
        - 优化：了解LLM的行为模式，改进提示词
        - 审计：记录所有交互，满足合规要求

        示例:
            >>> from trae_agent.utils.trajectory_recorder import TrajectoryRecorder
            >>> recorder = TrajectoryRecorder(...)
            >>> client.set_trajectory_recorder(recorder)
            >>> # 之后的所有交互都会被记录
        """
        self.client.set_trajectory_recorder(recorder)

    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """设置聊天历史记录。

        用于初始化或更新客户端的对话上下文。
        在某些场景下，可能需要预加载历史消息以保持对话的连续性。

        参数:
            messages: 消息列表，每条消息包含角色、内容等信息

        使用场景:
        - 从数据库恢复对话历史
        - 在不同的会话之间保持上下文
        - 预加载系统提示词和初始对话

        示例:
            >>> history = [
            ...     LLMMessage(role="system", content="你是一个助手"),
            ...     LLMMessage(role="user", content="你好"),
            ...     LLMMessage(role="assistant", content="你好！")
            ... ]
            >>> client.set_chat_history(history)

        注意:
            - 设置新的历史会覆盖之前的历史
            - 消息顺序很重要，应该按时间顺序排列
            - 系统提示词（role="system"）通常放在最前面
        """
        self.client.set_chat_history(messages)

    def chat(
        self,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """发送聊天消息到LLM并获取响应。

        这是LLM交互的核心方法，支持普通对话和工具调用。
        会自动处理消息历史、重试机制、错误处理等逻辑。

        参数:
            messages: 要发送的消息列表
            model_config: 模型配置，包括温度、令牌限制、重试次数等参数
            tools: 工具列表（可选），用于函数调用场景
                   如果提供，LLM可以请求调用这些工具来执行特定任务
            reuse_history: 是否复用消息历史（默认True）
                         - True: 在现有历史基础上追加新消息
                         - False: 仅使用当前消息，不包含历史

        返回:
            LLMResponse: LLM的响应对象，包含：
                - content: 生成的文本内容
                - usage: 令牌使用量统计
                - model: 使用的模型名称
                - finish_reason: 响应结束的原因
                - tool_calls: 工具调用列表（如果LLM请求调用工具）

        使用示例:
            >>> # 简单对话
            >>> messages = [LLMMessage(role="user", content="你好")]
            >>> response = client.chat(messages, config)
            >>> print(response.content)

            >>> # 带工具调用的对话
            >>> response = client.chat(messages, config, tools=[search_tool])
            >>> if response.tool_calls:
            ...     # 处理工具调用
            ...     for call in response.tool_calls:
            ...         result = execute_tool(call)
            ...         # 将结果返回给LLM
            ...         messages.append(LLMMessage(tool_result=result))
            ...         response = client.chat(messages, config)

        注意:
            - 当启用工具调用时，LLM可能返回tool_calls而非文本内容
            - 复用历史对多轮对话很重要，可以保持上下文连贯性
            - 如果遇到API错误，系统会自动重试（根据配置的重试次数）
        """
        return self.client.chat(messages, model_config, tools, reuse_history)

    def supports_tool_calling(self, model_config: ModelConfig) -> bool:
        """检查当前客户端是否支持工具调用功能。

        工具调用允许LLM执行预定义的函数或命令，极大地扩展了其能力范围。
        不同的提供商和模型对工具调用的支持情况可能不同。

        参数:
            model_config: 模型配置对象，包含模型名称等信息

        返回:
            bool: 如果模型支持工具调用返回True，否则返回False

        工具调用的能力:
        - 执行代码或脚本
        - 搜索信息
        - 访问数据库
        - 调用外部API
        - 执行文件操作（在安全限制内）

        使用示例:
            >>> if client.supports_tool_calling(config):
            ...     response = client.chat(messages, config, tools=my_tools)
            ... else:
            ...     print("当前模型不支持工具调用，使用基础对话")
            ...     response = client.chat(messages, config)

        注意:
            - 某些旧版本模型可能不支持工具调用
            - 即使支持，工具调用的性能和准确性也可能因模型而异
            - 建议在使用前检查支持情况，避免调用失败
        """
        return hasattr(self.client, "supports_tool_calling") and self.client.supports_tool_calling(
            model_config
        )

