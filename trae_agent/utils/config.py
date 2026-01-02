# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""配置管理模块。

该模块提供了 Trae Agent 的配置管理功能，包括：
1. 多种模型提供商的配置（OpenAI、Anthropic、DeepSeek、Doubao、Azure 等）
2. 模型参数配置（temperature、top_p、top_k、max_tokens 等）
3. MCP（Model Context Protocol）服务器配置
4. Agent 配置（最大步数、工具列表、Lakeview 配置等）
5. 配置解析和优先级处理（CLI > 环境变量 > 配置文件）
6. 支持从 YAML 配置文件和 JSON 格式的旧版配置文件加载配置

主要类：
- ConfigError: 配置错误异常
- ModelProvider: 模型提供商配置（API key、base URL 等）
- ModelConfig: 模型配置（模型名称、参数等）
- MCPServerConfig: MCP 服务器连接配置
- AgentConfig: Agent 基础配置
- TraeAgentConfig: Trae Agent 特定配置
- LakeviewConfig: Lakeview 分析工具配置
- Config: 主配置类，加载和管理所有配置
"""

import os
from dataclasses import dataclass, field

import yaml

from trae_agent.utils.legacy_config import LegacyConfig


class ConfigError(Exception):
    """配置错误异常类。

    在配置解析、验证或使用过程中发现错误时抛出此异常。
    常见场景包括：
    - 配置文件不存在或格式错误
    - 必需的配置项缺失
    - 配置值无效（如模型提供商未找到）
    - 配置冲突（如同时提供 config_file 和 config_string）
    """
    pass


@dataclass
class ModelProvider:
    """模型提供商配置类。

    该类封装了 LLM 提供商的连接和认证信息，支持多种官方和自定义提供商。

    对于官方模型提供商（如 OpenAI、Anthropic），base_url 是可选的，
    因为它们有标准的 API 端点。
    api_version 参数主要用于 Azure OpenAI。

    Attributes:
        api_key: API 密钥，用于身份认证
        provider: 提供商名称（如 "openai"、"anthropic"、"deepseek"、"doubao"、"azure" 等）
        base_url: API 基础 URL，用于自定义端点或非官方提供商。
                  对于官方提供商，如果不提供则使用默认端点。
        api_version: API 版本，主要供 Azure OpenAI 使用。
                     Azure OpenAI 需要在请求中指定 API 版本（如 "2024-02-15-preview"）。

    使用示例：
        # OpenAI 官方提供商
        openai_provider = ModelProvider(
            api_key="sk-...",
            provider="openai"
        )

        # Azure OpenAI
        azure_provider = ModelProvider(
            api_key="...",
            provider="azure",
            base_url="https://your-resource.openai.azure.com",
            api_version="2024-02-15-preview"
        )

        # 自定义提供商（如通过代理访问）
        custom_provider = ModelProvider(
            api_key="...",
            provider="custom",
            base_url="https://your-proxy.com/v1"
        )
    """

    api_key: str
    provider: str
    base_url: str | None = None
    api_version: str | None = None


@dataclass
class ModelConfig:
    """模型配置类。

    该类封装了单个 LLM 模型的所有配置参数，包括模型名称、生成参数、
    工具调用支持等。支持不同提供商的特定参数。

    Attributes:
        model: 模型名称（如 "gpt-4"、"claude-3-opus"、"deepseek-chat"、"doubao-pro" 等）
        model_provider: 模型提供商配置（ModelProvider 对象），包含 API 认证和端点信息
        temperature: 采样温度，控制输出的随机性。
                     值范围通常为 0.0-2.0，值越高输出越随机，值越低输出越确定。
        top_p: 核采样（Nucleus Sampling）参数，控制词汇表的采样范围。
               值范围通常为 0.0-1.0，较小的值限制只从最可能的 token 中采样。
        top_k: Top-K 采样参数，只从概率最高的 K 个 token 中采样。
               值为正整数，较大的值允许更多可能性。
        parallel_tool_calls: 是否允许并行调用工具。为 True 时，模型可以同时请求多个工具调用。
        max_retries: API 请求失败时的最大重试次数。
        max_tokens: 传统的最大输出 token 数限制。这是旧版参数，优先级低于 max_completion_tokens。
                   可选参数，如果未设置则使用默认值。
        supports_tool_calling: 模型是否支持工具调用。为 False 时，工具将被禁用。
        candidate_count: 候选回复数量，仅适用于 Gemini 等支持生成多个候选的模型。
                        可选参数。
        stop_sequences: 停止序列列表。当生成内容包含这些字符串时，模型会停止生成。
                       可选参数。
        max_completion_tokens: 最大完成 token 数，这是 Azure OpenAI 新模型（如 gpt-5）的参数。
                              它与 max_tokens 类似，但语义更清晰，表示生成的完成部分的 token 数。
                              可选参数，仅用于特定模型。

    优先级说明：
    - 对于 Azure OpenAI 的新模型（gpt-5、o3、o4-mini），优先使用 max_completion_tokens
    - 对于其他模型或未设置 max_completion_tokens 时，使用 max_tokens
    - 两者都未设置时，使用默认值 4096
    """

    model: str
    model_provider: ModelProvider
    temperature: float
    top_p: float
    top_k: int
    parallel_tool_calls: bool
    max_retries: int
    max_tokens: int | None = None  # Legacy max_tokens parameter, optional
    supports_tool_calling: bool = True
    candidate_count: int | None = None  # Gemini specific field
    stop_sequences: list[str] | None = None
    max_completion_tokens: int | None = None  # Azure OpenAI specific field

    def get_max_tokens_param(self) -> int:
        """获取最大 token 参数值。

        该方法根据配置返回应该使用的最大 token 限制。
        优先级顺序：max_completion_tokens > max_tokens > 默认值（4096）

        Returns:
            应该使用的最大 token 数（整数）

        使用场景：
        - 在调用 LLM API 时，需要指定生成内容的最大长度
        - 不同的 API 使用不同的参数名（Azure OpenAI 新模型使用 max_completion_tokens）
        - 该方法统一处理不同模型的参数差异
        """
        # 优先使用 max_completion_tokens（适用于 Azure OpenAI 新模型）
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        # 其次使用传统的 max_tokens
        elif self.max_tokens is not None:
            return self.max_tokens
        # 如果两者都未设置，返回默认值 4096
        else:
            return 4096

    def should_use_max_completion_tokens(self) -> bool:
        """判断是否应该使用 max_completion_tokens 参数。

        该方法主要用于 Azure OpenAI 的新模型（如 gpt-5、o3、o4-mini），
        这些模型推荐使用 max_completion_tokens 而非 max_tokens。

        Returns:
            是否使用 max_completion_tokens（True/False）

        判断条件：
        1. max_completion_tokens 参数已设置（非 None）
        2. 提供商是 Azure
        3. 模型名称包含 "gpt-5"、"o3" 或 "o4-mini"

        使用场景：
        - 在构建 API 请求时，决定使用哪个参数名
        - 确保新模型使用推荐的参数格式
        """
        return (
            self.max_completion_tokens is not None
            and self.model_provider.provider == "azure"
            and ("gpt-5" in self.model or "o3" in self.model or "o4-mini" in self.model)
        )

    def resolve_config_values(
        self,
        *,
        model_providers: dict[str, ModelProvider] | None = None,
        provider: str | None = None,
        model: str | None = None,
        model_base_url: str | None = None,
        api_key: str | None = None,
    ):
        """解析并覆盖配置值。

        当某些配置值通过 CLI 参数或环境变量提供时，它们将覆盖配置文件中的值。
        优先级：CLI 参数 > 环境变量 > 配置文件值

        Args:
            model_providers: 可用的模型提供商字典（提供商名称 -> ModelProvider 对象）。
                           用于查找已注册的提供商。
            provider: CLI 提供的提供商名称。如果提供，将切换到该提供商。
            model: CLI 提供的模型名称。如果提供，将覆盖配置文件中的模型。
            model_base_url: CLI 提供的基础 URL。如果提供，将覆盖配置文件中的 base_url。
            api_key: CLI 提供的 API 密钥。如果提供，将覆盖配置文件中的 API key。

        Raises:
            ConfigError: 当尝试注册新的提供商但未提供 API key 时抛出。

        处理逻辑：
        1. 解析并覆盖模型名称（优先使用 CLI 参数）
        2. 如果提供了 provider 参数：
           - 如果提供商存在于 model_providers 中，使用该提供商配置
           - 如果提供商不存在，则创建新的 ModelProvider（需要提供 api_key）
        3. 根据提供商名称构建环境变量名（如 OPENAI_API_KEY、DEEPSEEK_BASE_URL）
        4. 解析 API key 和 base URL（优先级：CLI > 环境变量 > 配置文件）

        使用场景：
        - 用户通过命令行参数临时更改模型或提供商
        - 使用环境变量管理敏感信息（如 API key）
        - 支持灵活的配置覆盖机制
        """
        # 解析模型名称
        self.model = str(resolve_config_value(cli_value=model, config_value=self.model))

        # 如果用户想要更改模型提供商，他们应该：
        # * 确保提供商名称在 model_providers 字典中可用；
        # * 如果不可用，应该提供 base_url 和 api_key 来注册新的提供商
        if provider:
            if model_providers and provider in model_providers:
                # 使用已注册的提供商
                self.model_provider = model_providers[provider]
            elif api_key is None:
                # 要注册新的提供商，必须提供 api_key
                raise ConfigError("To register a new model provider, an api_key should be provided")
            else:
                # 创建新的提供商配置
                self.model_provider = ModelProvider(
                    api_key=api_key,
                    provider=provider,
                    base_url=model_base_url,
                )

        # 将提供商映射到其环境变量名称
        # 例如：openai -> OPENAI_API_KEY, deepseek -> DEEPSEEK_BASE_URL
        env_var_api_key = str(self.model_provider.provider).upper() + "_API_KEY"
        env_var_api_base_url = str(self.model_provider.provider).upper() + "_BASE_URL"

        # 解析 API key（优先级：CLI > 环境变量 > 配置文件）
        resolved_api_key = resolve_config_value(
            cli_value=api_key,
            config_value=self.model_provider.api_key,
            env_var=env_var_api_key,
        )

        # 解析 base URL（优先级：CLI > 环境变量 > 配置文件）
        resolved_api_base_url = resolve_config_value(
            cli_value=model_base_url,
            config_value=self.model_provider.base_url,
            env_var=env_var_api_base_url,
        )

        # 更新提供商配置
        if resolved_api_key:
            self.model_provider.api_key = str(resolved_api_key)

        if resolved_api_base_url:
            self.model_provider.base_url = str(resolved_api_base_url)


@dataclass
class MCPServerConfig:
    """MCP（Model Context Protocol）服务器配置类。

    该类定义了 MCP 服务器的连接配置，支持多种传输方式（transport types）：
    1. stdio：通过标准输入/输出进行通信
    2. sse：通过 Server-Sent Events（SSE）进行通信
    3. streamable http：通过流式 HTTP 进行通信
    4. websocket：通过 WebSocket 进行通信

    Attributes:
        # stdio 传输方式的配置
        command: 启动 MCP 服务器的命令（如 "npx", "python"）
        args: 启动命令的参数列表
        env: 启动时设置的环境变量字典
        cwd: 工作目录路径

        # sse 传输方式的配置
        url: SSE 服务器的 URL

        # streamable http 传输方式的配置
        http_url: HTTP 服务器的 URL
        headers: HTTP 请求头字典

        # websocket 传输方式的配置
        tcp: WebSocket 服务器的 TCP 地址

        # 通用配置
        timeout: 连接和请求的超时时间（秒）
        trust: 是否信任不受信任的证书（如自签名证书）

        # 元数据
        description: MCP 服务器的描述信息

    使用示例：
        # stdio 传输
        stdio_server = MCPServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
            timeout=30
        )

        # sse 传输
        sse_server = MCPServerConfig(
            url="https://example.com/mcp/sse",
            timeout=30
        )

        # http 传输
        http_server = MCPServerConfig(
            http_url="https://example.com/mcp/api",
            headers={"Authorization": "Bearer token"},
            timeout=30
        )
    """
    # For stdio transport
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None

    # For sse transport
    url: str | None = None

    # For streamable http transport
    http_url: str | None = None
    headers: dict[str, str] | None = None

    # For websocket transport
    tcp: str | None = None

    # Common
    timeout: int | None = None
    trust: bool | None = None

    # Metadata
    description: str | None = None


@dataclass
class AgentConfig:
    """Agent 配置基类。

    该类定义了所有 Agent 配置的基础结构。具体类型的 Agent（如 TraeAgent）
    可以继承此类并添加特定配置。

    Attributes:
        allow_mcp_servers: 允许使用的 MCP 服务器名称列表。Agent 只能访问此列表中的服务器。
        mcp_servers_config: 所有可用的 MCP 服务器配置字典。
                           键为服务器名称，值为 MCPServerConfig 对象。
        max_steps: Agent 执行的最大步数限制。
                  超过此步数后，Agent 将停止执行。
        model: Agent 使用的模型配置（ModelConfig 对象）。
        tools: Agent 可使用的工具名称列表。

    设计说明：
    - allow_mcp_servers 提供了访问控制，防止 Agent 访问未授权的 MCP 服务器
    - mcp_servers_config 包含所有服务器配置，allow_mcp_servers 决定实际可用哪些
    - max_steps 防止无限循环或过长的执行链
    - tools 列表定义了 Agent 可以使用的能力
    """

    allow_mcp_servers: list[str]
    mcp_servers_config: dict[str, MCPServerConfig]
    max_steps: int
    model: ModelConfig
    tools: list[str]


@dataclass
class TraeAgentConfig(AgentConfig):
    """Trae Agent 配置类。

    Trae Agent 是项目中的主要 Agent 类型，继承自 AgentConfig。
    提供了 Trae Agent 特定的配置和默认工具列表。

    Attributes:
        enable_lakeview: 是否启用 Lakeview 分析功能。
                        Lakeview 用于分析和可视化 Agent 的执行步骤。
        tools: Trae Agent 默认使用的工具列表。
               默认工具包括：
               - bash: 执行 shell 命令
               - str_replace_based_edit_tool: 基于字符串替换的文件编辑工具
               - sequentialthinking: 顺序思考工具，帮助 Agent 进行复杂推理
               - task_done: 任务完成工具，用于标记任务完成

    默认工具说明：
    - bash: 允许 Agent 执行系统命令，访问文件系统、运行脚本等
    - str_replace_based_edit_tool: 编辑文件内容，支持精确的字符串替换
    - sequentialthinking: 辅助 Agent 进行多步骤推理，提高决策质量
    - task_done: 告知系统任务已完成，用于终止 Agent 循环

    继承的属性（来自 AgentConfig）：
    - allow_mcp_servers
    - mcp_servers_config
    - max_steps
    - model
    """

    enable_lakeview: bool = True
    tools: list[str] = field(
        default_factory=lambda: [
            "bash",
            "str_replace_based_edit_tool",
            "sequentialthinking",
            "task_done",
        ]
    )

    def resolve_config_values(
        self,
        *,
        max_steps: int | None = None,
    ):
        """解析并覆盖配置值。

        当配置值通过 CLI 参数或环境变量提供时，覆盖配置文件中的值。

        Args:
            max_steps: CLI 提供的最大步数。如果提供，将覆盖配置文件中的值。

        使用场景：
        - 用户通过命令行参数临时更改最大步数
        - 支持配置值的灵活覆盖
        """
        resolved_value = resolve_config_value(cli_value=max_steps, config_value=self.max_steps)
        if resolved_value:
            self.max_steps = int(resolved_value)


@dataclass
class LakeviewConfig:
    """Lakeview 配置类。

    Lakeview 是一个用于分析和可视化 Agent 执行步骤的工具。
    该类配置 Lakeview 使用的模型参数。

    Attributes:
        model: Lakeview 分析使用的模型配置（ModelConfig 对象）。

    功能说明：
    - Lakeview 使用 LLM 分析 Agent 的每个执行步骤
    - 生成步骤摘要、识别问题、提供改进建议
    - 帮助开发者理解和调试 Agent 的行为

    使用场景：
    - Agent 执行完成后，自动分析整个执行轨迹
    - 对每个步骤生成详细的分析报告
    - 识别执行中的错误或低效操作
    """


@dataclass
class Config:
    """主配置类。

    该类是配置管理的中心，负责加载、解析和管理所有配置信息。
    支持从 YAML 配置文件或旧版 JSON 格式配置文件加载配置。

    主要功能：
    1. 从 YAML 文件加载配置
    2. 解析和验证模型提供商配置
    3. 解析和验证模型配置
    4. 解析 Lakeview 配置
    5. 解析 MCP 服务器配置
    6. 解析 Agent 配置
    7. 支持从 CLI 参数和环境变量覆盖配置
    8. 支持从旧版 JSON 配置文件迁移

    Attributes:
        lakeview: Lakeview 配置（LakeviewConfig 对象）。
                 如果未启用 Lakeview 或未配置，则为 None。
        model_providers: 模型提供商配置字典。
                        键为提供商名称（如 "openai", "deepseek"），
                        值为 ModelProvider 对象。
        models: 模型配置字典。
               键为模型名称（如 "gpt-4", "deepseek-chat"），
               值为 ModelConfig 对象。
        trae_agent: Trae Agent 配置（TraeAgentConfig 对象）。
                    如果未配置，则为 None。

    配置文件结构（YAML 格式）：
        model_providers:
          openai:
            api_key: ${OPENAI_API_KEY}
            provider: openai
          deepseek:
            api_key: ${DEEPSEEK_API_KEY}
            provider: deepseek
            base_url: https://api.deepseek.com

        models:
          gpt4:
            model: gpt-4
            model_provider: openai
            temperature: 0.7
            top_p: 0.9
            top_k: 40
            parallel_tool_calls: true
            max_retries: 3
            max_tokens: 4096

          deepseek-chat:
            model: deepseek-chat
            model_provider: deepseek
            temperature: 0.7
            top_p: 0.9
            top_k: 40
            parallel_tool_calls: true
            max_retries: 3
            max_tokens: 4096

        lakeview:
          model: gpt4

        mcp_servers:
          filesystem:
            command: npx
            args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]

        agents:
          trae_agent:
            model: deepseek-chat
            max_steps: 30
            enable_lakeview: true
            allow_mcp_servers: ["filesystem"]

    使用示例：
        # 从 YAML 文件加载配置
        config = Config.create(config_file="trae_config.yaml")

        # 从 YAML 字符串加载配置
        config_yaml = """
        model_providers:
          openai:
            api_key: "sk-..."
            provider: openai
        models:
          gpt4:
            model: gpt-4
            model_provider: openai
            temperature: 0.7
            top_p: 0.9
            top_k: 40
            parallel_tool_calls: true
            max_retries: 3
        agents:
          trae_agent:
            model: gpt4
            max_steps: 30
            """
        config = Config.create(config_string=config_yaml)

        # 从旧版 JSON 配置文件加载
        config = Config.create(config_file="trae_config.json")

        # 覆盖配置值（CLI 参数或环境变量）
        config = config.resolve_config_values(
            provider="deepseek",
            model="deepseek-chat",
            max_steps=50
        )
    """

    lakeview: LakeviewConfig | None = None
    model_providers: dict[str, ModelProvider] | None = None
    models: dict[str, ModelConfig] | None = None

    trae_agent: TraeAgentConfig | None = None

    @classmethod
    def create(
        cls,
        *,
        config_file: str | None = None,
        config_string: str | None = None,
    ) -> "Config":
        """创建配置对象。

        从指定的文件或字符串加载并解析配置。支持 YAML 格式的配置文件和
        旧版 JSON 格式的配置文件。

        Args:
            config_file: 配置文件路径。可以是 YAML 文件（.yaml, .yml）或
                        JSON 文件（旧版格式）。必须提供 config_file 或 config_string 之一。
            config_string: 配置内容的字符串（YAML 格式）。
                          必须提供 config_file 或 config_string 之一。

        Returns:
            解析后的 Config 对象。

        Raises:
            ConfigError: 在以下情况下抛出：
                - 同时提供了 config_file 和 config_string
                - 未提供 config_file 或 config_string
                - YAML 解析失败
                - 缺少必需的配置项（model_providers, models, agents）
                - 引用的模型提供商或模型不存在
                - Lakeview 已启用但未配置

        解析流程：
        1. 检查参数合法性（不能同时提供文件和字符串）
        2. 如果是 JSON 文件，使用旧版解析器（create_from_legacy_config）
        3. 解析 YAML 配置（从文件或字符串）
        4. 解析模型提供商配置（model_providers）
        5. 解析模型配置（models），并关联到对应的提供商
        6. 解析 Lakeview 配置（如果存在）
        7. 解析 MCP 服务器配置（mcp_servers）
        8. 解析 Agent 配置（agents），目前只支持 trae_agent
        9. 验证配置的完整性和合法性

        使用示例：
            # 从文件加载
            config = Config.create(config_file="trae_config.yaml")

            # 从字符串加载
            config = Config.create(config_string=yaml_string)

            # 从旧版 JSON 文件加载
            config = Config.create(config_file="trae_config.json")
        """
        # 验证参数：不能同时提供文件和字符串
        if config_file and config_string:
            raise ConfigError("Only one of config_file or config_string should be provided")

        # 从文件或字符串解析 YAML 配置
        try:
            if config_file is not None:
                # 如果是 JSON 文件，使用旧版解析器
                if config_file.endswith(".json"):
                    return cls.create_from_legacy_config(config_file=config_file)
                # 读取 YAML 文件
                with open(config_file, "r") as f:
                    yaml_config = yaml.safe_load(f)
            elif config_string is not None:
                # 解析 YAML 字符串
                yaml_config = yaml.safe_load(config_string)
            else:
                # 未提供任何配置源
                raise ConfigError("No config file or config string provided")
        except yaml.YAMLError as e:
            # YAML 解析失败
            raise ConfigError(f"Error parsing YAML config: {e}") from e

        # 创建空的配置对象
        config = cls()

        # 解析模型提供商配置
        model_providers = yaml_config.get("model_providers", None)
        if model_providers is not None and len(model_providers.keys()) > 0:
            config_model_providers: dict[str, ModelProvider] = {}
            # 遍历所有提供商配置
            for model_provider_name, model_provider_config in model_providers.items():
                config_model_providers[model_provider_name] = ModelProvider(**model_provider_config)
            config.model_providers = config_model_providers
        else:
            # 必须提供至少一个模型提供商
            raise ConfigError("No model providers provided")

        # 解析模型配置并填充 model_provider 字段
        models = yaml_config.get("models", None)
        if models is not None and len(models.keys()) > 0:
            config_models: dict[str, ModelConfig] = {}
            for model_name, model_config in models.items():
                # 验证模型提供商是否存在
                if model_config["model_provider"] not in config_model_providers:
                    raise ConfigError(f"Model provider {model_config['model_provider']} not found")
                # 创建模型配置对象
                config_models[model_name] = ModelConfig(**model_config)
                # 关联到对应的提供商
                config_models[model_name].model_provider = config_model_providers[
                    model_config["model_provider"]
                ]
            config.models = config_models
        else:
            # 必须提供至少一个模型
            raise ConfigError("No models provided")

        # 解析 Lakeview 配置
        lakeview = yaml_config.get("lakeview", None)
        if lakeview is not None:
            lakeview_model_name = lakeview.get("model", None)
            if lakeview_model_name is None:
                raise ConfigError("No model provided for lakeview")
            # 获取 Lakeview 使用的模型
            lakeview_model = config_models[lakeview_model_name]
            config.lakeview = LakeviewConfig(
                model=lakeview_model,
            )
        else:
            config.lakeview = None

        # 解析 MCP 服务器配置
        mcp_servers_config = {
            k: MCPServerConfig(**v) for k, v in yaml_config.get("mcp_servers", {}).items()
        }
        # 获取允许的 MCP 服务器列表
        allow_mcp_servers = yaml_config.get("allow_mcp_servers", [])

        # 解析 Agent 配置
        agents = yaml_config.get("agents", None)
        if agents is not None and len(agents.keys()) > 0:
            for agent_name, agent_config in agents.items():
                # 获取 Agent 使用的模型
                agent_model_name = agent_config.get("model", None)
                if agent_model_name is None:
                    raise ConfigError(f"No model provided for {agent_name}")
                try:
                    agent_model = config_models[agent_model_name]
                except KeyError as e:
                    raise ConfigError(f"Model {agent_model_name} not found") from e

                # 根据不同的 Agent 类型创建配置
                match agent_name:
                    case "trae_agent":
                        # 创建 Trae Agent 配置
                        trae_agent_config = TraeAgentConfig(
                            **agent_config,
                            mcp_servers_config=mcp_servers_config,
                            allow_mcp_servers=allow_mcp_servers,
                        )
                        # 设置 Agent 使用的模型
                        trae_agent_config.model = agent_model
                        # 验证：如果启用了 Lakeview，必须提供 Lakeview 配置
                        if trae_agent_config.enable_lakeview and config.lakeview is None:
                            raise ConfigError("Lakeview is enabled but no lakeview config provided")
                        config.trae_agent = trae_agent_config
                    case _:
                        # 未知的 Agent 类型
                        raise ConfigError(f"Unknown agent: {agent_name}")
        else:
            # 必须提供至少一个 Agent 配置
            raise ConfigError("No agent configs provided")
        return config

    def resolve_config_values(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        model_base_url: str | None = None,
        api_key: str | None = None,
        max_steps: int | None = None,
    ):
        """解析并覆盖配置值。

        当配置值通过 CLI 参数或环境变量提供时，覆盖配置文件中的值。
        此方法会递归地调用子配置对象的 resolve_config_values 方法。

        Args:
            provider: CLI 提供的提供商名称。
            model: CLI 提供的模型名称。
            model_base_url: CLI 提供的基础 URL。
            api_key: CLI 提供的 API 密钥。
            max_steps: CLI 提供的最大步数。

        Returns:
            返回自身（Config 对象），支持链式调用。

        处理流程：
        1. 如果存在 trae_agent 配置：
           - 调用 trae_agent.resolve_config_values() 处理 Agent 特定配置（如 max_steps）
           - 调用 trae_agent.model.resolve_config_values() 处理模型配置（如 model、provider、api_key）

        2. 所有子配置使用相同的优先级：CLI > 环境变量 > 配置文件

        使用示例：
            # 从文件加载配置
            config = Config.create(config_file="trae_config.yaml")

            # 使用 CLI 参数覆盖配置
            config = config.resolve_config_values(
                provider="deepseek",
                model="deepseek-chat",
                max_steps=50
            )

            # 可以链式调用
            config = config.resolve_config_values(
                model="gpt-4",
                api_key="sk-..."
            ).resolve_config_values(
                max_steps=100
            )
        """
        # 如果存在 Trae Agent 配置，递归解析其配置值
        if self.trae_agent:
            # 解析 Agent 特定配置（如 max_steps）
            self.trae_agent.resolve_config_values(
                max_steps=max_steps,
            )
            # 解析模型配置（如 model、provider、api_key、base_url）
            self.trae_agent.model.resolve_config_values(
                model_providers=self.model_providers,
                provider=provider,
                model=model,
                model_base_url=model_base_url,
                api_key=api_key,
            )
        return self

    @classmethod
    def create_from_legacy_config(
        cls,
        *,
        legacy_config: LegacyConfig | None = None,
        config_file: str | None = None,
    ) -> "Config":
        """从旧版配置创建配置对象。

        该方法支持从旧版的 JSON 格式配置文件创建 Config 对象。
        旧版配置使用 LegacyConfig 类表示。

        Args:
            legacy_config: 旧版配置对象（LegacyConfig 对象）。如果提供，
                          则忽略 config_file 参数。
            config_file: 旧版配置文件路径（JSON 格式）。如果提供，
                        则忽略 legacy_config 参数。

        Returns:
            转换后的 Config 对象。

        Raises:
            ConfigError: 在以下情况下抛出：
                - 同时提供了 legacy_config 和 config_file
                - 未提供 legacy_config 或 config_file

        转换逻辑：
        1. 加载旧版配置（从对象或文件）
        2. 将旧版的模型提供商配置转换为新的 ModelProvider 对象
        3. 将旧版的模型参数转换为新的 ModelConfig 对象
        4. 将旧版的 MCP 服务器配置转换为新的 MCPServerConfig 对象
        5. 创建 TraeAgentConfig 对象
        6. 根据配置创建 LakeviewConfig 对象（如果启用）
        7. 组装成新的 Config 对象

        旧版配置说明：
        - 旧版配置只有一个默认提供商（default_provider）
        - 旧版配置只有一个默认模型
        - 配置结构相对简单，新版支持多提供商、多模型、多 Agent

        使用示例：
            # 从旧版 JSON 文件加载
            config = Config.create_from_legacy_config(config_file="trae_config.json")

            # 从现有的 LegacyConfig 对象创建
            legacy = LegacyConfig("trae_config.json")
            config = Config.create_from_legacy_config(legacy_config=legacy)
        """
        # 验证参数：不能同时提供旧版配置对象和文件路径
        if legacy_config and config_file:
            raise ConfigError("Only one of legacy_config or config_file should be provided")

        # 从文件加载旧版配置（如果提供了文件路径）
        if config_file:
            legacy_config = LegacyConfig(config_file)
        elif not legacy_config:
            # 未提供任何旧版配置源
            raise ConfigError("No legacy_config or config_file provided")

        # 创建模型提供商配置
        model_provider = ModelProvider(
            api_key=legacy_config.model_providers[legacy_config.default_provider].api_key,
            base_url=legacy_config.model_providers[legacy_config.default_provider].base_url,
            api_version=legacy_config.model_providers[legacy_config.default_provider].api_version,
            provider=legacy_config.default_provider,
        )

        # 创建模型配置
        model_config = ModelConfig(
            model=legacy_config.model_providers[legacy_config.default_provider].model,
            model_provider=model_provider,
            max_tokens=legacy_config.model_providers[legacy_config.default_provider].max_tokens,
            temperature=legacy_config.model_providers[legacy_config.default_provider].temperature,
            top_p=legacy_config.model_providers[legacy_config.default_provider].top_p,
            top_k=legacy_config.model_providers[legacy_config.default_provider].top_k,
            parallel_tool_calls=legacy_config.model_providers[
                legacy_config.default_provider
            ].parallel_tool_calls,
            max_retries=legacy_config.model_providers[legacy_config.default_provider].max_retries,
            candidate_count=legacy_config.model_providers[
                legacy_config.default_provider
            ].candidate_count,
            stop_sequences=legacy_config.model_providers[
                legacy_config.default_provider
            ].stop_sequences,
        )

        # 转换 MCP 服务器配置
        mcp_servers_config = {
            k: MCPServerConfig(**vars(v)) for k, v in legacy_config.mcp_servers.items()
        }

        # 创建 Trae Agent 配置
        trae_agent_config = TraeAgentConfig(
            max_steps=legacy_config.max_steps,
            enable_lakeview=legacy_config.enable_lakeview,
            model=model_config,
            allow_mcp_servers=legacy_config.allow_mcp_servers,
            mcp_servers_config=mcp_servers_config,
        )

        # 如果启用了 Lakeview，创建 Lakeview 配置
        if trae_agent_config.enable_lakeview:
            lakeview_config = LakeviewConfig(
                model=model_config,
            )
        else:
            lakeview_config = None

        # 返回转换后的 Config 对象
        return cls(
            trae_agent=trae_agent_config,
            lakeview=lakeview_config,
            model_providers={
                legacy_config.default_provider: model_provider,
            },
            models={
                "default_model": model_config,
            },
        )


def resolve_config_value(
    *,
    cli_value: int | str | float | None,
    config_value: int | str | float | None,
    env_var: str | None = None,
) -> int | str | float | None:
    """解析配置值，按优先级返回值。

    该函数实现了配置值的优先级解析机制，支持从多个来源获取配置值。
    优先级顺序：CLI 参数 > 环境变量 > 配置文件 > 默认值

    Args:
        cli_value: 从命令行参数提供的值。优先级最高。
        config_value: 从配置文件中读取的值。
        env_var: 环境变量名称。如果提供，会检查该环境变量是否已设置。

    Returns:
        解析后的配置值，类型与输入值相同。如果所有来源都无值，则返回 None。

    优先级说明：
    1. CLI 参数：用户通过命令行直接指定的值，优先级最高
    2. 环境变量：从系统环境变量读取的值，适合管理敏感信息（如 API key）
    3. 配置文件：从 YAML/JSON 配置文件中读取的值，优先级较低
    4. 默认值：如果以上所有来源都没有值，返回 None

    使用场景：
        # 在配置解析时使用
        api_key = resolve_config_value(
            cli_value=cli_args.api_key,
            config_value=config.api_key,
            env_var="OPENAI_API_KEY"
        )

        # 如果用户通过 CLI 提供了值，使用 CLI 值
        # 否则检查 OPENAI_API_KEY 环境变量
        # 否则使用配置文件中的值
        # 否则返回 None

        # 环境变量示例：
        # export OPENAI_API_KEY="sk-..."
        # export DEEPSEEK_API_KEY="..."

    特性：
    - 支持多种类型：int、str、float
    - 灵活的配置覆盖机制
    - 适合 CI/CD 环境和环境变量管理
    - 保持向后兼容性（配置文件优先级较低）
    """
    # 优先级 1: CLI 参数
    if cli_value is not None:
        return cli_value

    # 优先级 2: 环境变量
    # 只有当 env_var 参数提供且环境变量存在时才使用
    if env_var and os.getenv(env_var):
        return os.getenv(env_var)

    # 优先级 3: 配置文件
    if config_value is not None:
        return config_value

    # 优先级 4: 默认值（None）
    return None
