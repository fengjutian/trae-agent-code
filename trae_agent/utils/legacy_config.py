# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

# TODO: remove these annotations by defining fine-grained types
# pyright: reportAny=false
# pyright: reportUnannotatedClassAttribute=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

"""Trae Agent 配置管理模块。

该模块提供了Trae Agent的配置管理功能，包括：
- 模型提供商配置
- MCP（Model Context Protocol）服务器配置
- Lake View分析配置
- 配置文件加载和解析

设计特点:
- 支持JSON配置文件和直接字典配置
- 提供默认值和类型转换
- 支持多种模型提供商（Anthropic、OpenAI等）
- 支持MCP服务器的多种传输方式（stdio、sse、http、websocket）

使用示例:
    >>> config = LegacyConfig("trae_config.json")
    >>> print(config.default_provider)
    >>> print(config.model_providers)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, override


# data class for model parameters
@dataclass
class ModelParameters:
    """模型参数数据类。

    存储单个LLM模型提供商的所有配置参数。
    这些参数用于初始化和管理LLM客户端连接。

    属性说明:
        model: 模型名称（如"claude-3-5-sonnet"、"gpt-4"等）
        api_key: API访问密钥，用于身份验证
        max_tokens: 最大生成令牌数，控制响应长度
        temperature: 温度参数（0.0-2.0），控制输出的随机性
                    0.0=确定性输出，1.0=平衡，2.0=高度随机
        top_p: 核采样参数（0.0-1.0），控制词汇选择的多样性
        top_k: 顶部K采样参数（0=禁用），限制每次采样考虑的候选token数
        parallel_tool_calls: 是否支持并行工具调用（True/False）
        max_retries: 最大重试次数，用于API调用失败时的自动重试
        base_url: API基础URL（可选），某些提供商需要自定义API端点
        api_version: API版本（可选），某些提供商支持多个API版本
        candidate_count: 候选数量（可选），Gemini模型专用参数
        stop_sequences: 停止序列（可选），用于指定停止生成的特殊字符串

    注意:
        - 不同提供商可能支持不同的参数子集
        - 某些参数对特定模型才有意义（如candidate_count仅用于Gemini）
    """

    model: str
    api_key: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    parallel_tool_calls: bool
    max_retries: int
    base_url: str | None = None
    api_version: str | None = None
    candidate_count: int | None = None  # Gemini specific field
    stop_sequences: list[str] | None = None


@dataclass
class LakeviewConfig:
    """Lake View配置数据类。

    存储Lake View分析器的配置参数。
    Lake View用于分析和标记AI Agent在解决bug过程中的行为。

    属性说明:
        model_provider: 模型提供商名称（如"anthropic"、"deepseek"等）
        model_name: 具体模型名称（如"claude-3.5-sonnet"、"deepseek-chat"等）

    使用场景:
        - 配置用于分析agent行为的LLM模型
        - 指定不同提供商的模型以优化成本或性能

    注意:
        - 该配置仅在enable_lakeview=True时生效
        - model_provider必须是已配置的提供商之一
    """

    model_provider: str
    model_name: str


@dataclass
class MCPServerConfig:
    """MCP（Model Context Protocol）服务器配置数据类。

    存储单个MCP服务器的连接和配置信息。
    支持多种传输方式：stdio、SSE、HTTP、WebSocket。

    属性说明:
        # stdio传输相关
        command: 启动MCP服务器的命令（可选）
        args: 命令参数列表（可选）
        env: 环境变量字典（可选）
        cwd: 工作目录（可选）

        # SSE（Server-Sent Events）传输相关
        url: SSE服务器URL（可选）

        # 可流式HTTP传输相关
        http_url: HTTP服务器URL（可选）
        headers: HTTP请求头字典（可选）

        # WebSocket传输相关
        tcp: WebSocket TCP地址（可选）

        # 通用配置
        timeout: 超时时间（秒，可选）
        trust: 是否信任该服务器（可选），用于安全控制

        # 元数据
        description: 服务器描述（可选），用于标识服务器用途

    使用场景:
        - 连接本地MCP服务器（使用stdio）
        - 连接远程MCP服务器（使用SSE或HTTP）
        - 连接WebSocket MCP服务器

    注意:
        - 不同传输方式使用不同的配置项
        - 至少需要配置一种传输方式
        - trust参数用于安全控制，谨慎使用
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
class LegacyConfig:
    """Trae Agent配置管理器。

    负责加载、解析和管理Trae Agent的所有配置。
    支持从文件或直接从字典加载配置。

    主要功能:
    - 从JSON文件或字典加载配置
    - 管理多个模型提供商的配置
    - 管理MCP服务器的配置
    - 管理Lake View的配置
    - 提供默认值和类型转换

    属性说明:
        default_provider: 默认的LLM提供商名称
        max_steps: Agent执行的最大步数限制
        model_providers: 所有模型提供商的配置字典，键为提供商名，值为ModelParameters
        mcp_servers: 所有MCP服务器的配置字典，键为服务器名，值为MCPServerConfig
        lakeview_config: Lake View分析器配置（可选）
        enable_lakeview: 是否启用Lake View功能（默认True）
        allow_mcp_servers: 允许使用的MCP服务器列表（默认空列表）

    使用示例:
        >>> # 从文件加载
        >>> config = LegacyConfig("trae_config.json")
        >>>
        >>> # 从字典加载
        >>> config_dict = {
        ...     "default_provider": "anthropic",
        ...     "model_providers": {...},
        ...     "enable_lakeview": True
        ... }
        >>> config = LegacyConfig(config_dict)
        >>>
        >>> # 访问配置
        >>> print(config.default_provider)
        >>> print(config.max_steps)
    """

    default_provider: str
    max_steps: int
    model_providers: dict[str, ModelParameters]
    mcp_servers: dict[str, MCPServerConfig]
    lakeview_config: LakeviewConfig | None = None
    enable_lakeview: bool = True
    allow_mcp_servers: list[str] = field(default_factory=list)

    def __init__(self, config_or_config_file: str | dict = "trae_config.json"):  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
        """初始化配置管理器。

        支持从JSON文件或直接从字典加载配置。
        如果是文件路径，则读取并解析JSON文件。

        参数:
            config_or_config_file: 配置文件路径或配置字典
                                  如果是字典，直接使用；如果是字符串，则作为文件路径
                                  默认为"trae_config.json"

        实现细节:
            - 检查输入类型，判断是文件路径还是配置字典
            - 如果是文件路径，检查文件是否存在
            - 使用json.load加载配置文件
            - 捕获并报告加载错误，使用空配置作为后备
            - 如果文件不存在，使用空配置

        注意:
            - 文件不存在时不会抛出异常，而是使用空配置
            - 加载失败时会打印警告信息
            - 配置加载失败后，后续会使用默认值
        """
        # Accept either file path or direct config dict
        if isinstance(config_or_config_file, dict):
            self._config = config_or_config_file
        else:
            config_path = Path(config_or_config_file)
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        self._config = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load config file {config_or_config_file}: {e}")
                    self._config = {}
            else:
                self._config = {}

        # 从配置中读取并设置各个属性
        # 使用get方法提供默认值，确保配置缺失时的行为

        # 默认提供商，默认为"anthropic"
        self.default_provider = self._config.get("default_provider", "anthropic")
        # 最大步数限制，默认为20
        self.max_steps = self._config.get("max_steps", 20)
        # 模型提供商配置字典，初始化为空
        self.model_providers = {}
        # 是否启用Lake View，默认为True
        self.enable_lakeview = self._config.get("enable_lakeview", True)
        # MCP服务器配置字典，从配置中读取并转换为MCPServerConfig对象
        self.mcp_servers = {
            k: MCPServerConfig(**v) for k, v in self._config.get("mcp_servers", {}).items()
        }
        # 允许的MCP服务器列表
        self.allow_mcp_servers = self._config.get("allow_mcp_servers", [])

        # 检查是否配置了模型提供商，如果没有则使用默认配置
        if len(self._config.get("model_providers", [])) == 0:
            # 使用默认的Anthropic配置
            # 这是在配置文件中没有配置任何提供商时的后备配置
            self.model_providers = {
                "anthropic": ModelParameters(
                    model="claude-sonnet-4-20250514",
                    api_key="",
                    base_url="https://api.anthropic.com",
                    max_tokens=4096,
                    temperature=0.5,
                    top_p=1,
                    top_k=0,
                    parallel_tool_calls=False,
                    max_retries=10,
                ),
            }
        else:
            # 遍历配置文件中定义的每个模型提供商
            for provider in self._config.get("model_providers", {}):
                # 获取特定提供商的配置字典
                provider_config: dict[str, Any] = self._config.get("model_providers", {}).get(
                    provider, {}
                )

                # 获取Gemini特定的candidate_count参数
                candidate_count = provider_config.get("candidate_count")
                # 构建ModelParameters对象，进行类型转换以确保正确性
                self.model_providers[provider] = ModelParameters(
                    model=str(provider_config.get("model", "")),
                    api_key=str(provider_config.get("api_key", "")),
                    base_url=str(provider_config.get("base_url"))
                    if "base_url" in provider_config
                    else None,
                    max_tokens=int(provider_config.get("max_tokens", 1000)),
                    temperature=float(provider_config.get("temperature", 0.5)),
                    top_p=float(provider_config.get("top_p", 1)),
                    top_k=int(provider_config.get("top_k", 0)),
                    max_retries=int(provider_config.get("max_retries", 10)),
                    parallel_tool_calls=bool(provider_config.get("parallel_tool_calls", False)),
                    api_version=str(provider_config.get("api_version"))
                    if "api_version" in provider_config
                    else None,
                    candidate_count=int(candidate_count) if candidate_count is not None else None,
                    stop_sequences=provider_config.get("stop_sequences")
                    if "stop_sequences" in provider_config
                    else None,
                )

        # 配置Lake View，默认使用default_provider的设置
        # Lake View用于分析和标记agent的执行步骤
        lakeview_config_data = self._config.get("lakeview_config", {})

        # 只在enable_lakeview=True时配置
        if self.enable_lakeview:
            # 获取Lake View的模型提供商配置
            model_provider = lakeview_config_data.get("model_provider", None)
            # 获取具体的模型名称
            model_name = lakeview_config_data.get("model_name", None)

            # 如果未指定提供商，使用默认提供商
            if model_provider is None:
                model_provider = self.default_provider

            # 如果未指定模型名称，使用该提供商的默认模型
            if model_name is None:
                model_name = self.model_providers[model_provider].model

            # 创建LakeviewConfig对象
            self.lakeview_config = LakeviewConfig(
                model_provider=str(model_provider),
                model_name=str(model_name),
            )

        return

    @override
    def __str__(self) -> str:
        """返回配置的字符串表示。

        提供配置对象的可读字符串摘要，用于调试和日志记录。

        返回:
            str: 包含默认提供商、最大步数和模型提供商的格式化字符串

        示例:
            >>> config = LegacyConfig(...)
            >>> print(config)
            "Config(default_provider=anthropic, max_steps=20, model_providers={'anthropic': ModelParameters(...)})"
        """
        return f"Config(default_provider={self.default_provider}, max_steps={self.max_steps}, model_providers={self.model_providers})"
