# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""DeepSeek provider configuration.

DeepSeek是字节跳动旗下的AI模型提供商，提供高性能的通用大语言模型。
该模块实现了DeepSeek API的客户端封装，支持工具调用和OpenAI兼容接口。
"""

import openai

from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.openai_compatible_base import (
    OpenAICompatibleClient,
    ProviderConfig,
)


class DeepSeekProvider(ProviderConfig):
    """DeepSeek提供商配置类。

    实现了ProviderConfig接口，提供DeepSeek特定的配置和行为。
    包括客户端创建、工具调用支持判断、额外参数和头信息的处理。
    """

    def create_client(
        self, api_key: str, base_url: str | None, api_version: str | None
    ) -> openai.OpenAI:
        """创建DeepSeek的OpenAI兼容客户端。

        使用OpenAI SDK连接DeepSeek API，因为DeepSeek提供了OpenAI兼容的API接口。

        参数:
            api_key: DeepSeek API密钥
            base_url: API基础URL，如果为空则使用默认的DeepSeek API地址
            api_version: API版本（DeepSeek当前不使用此参数）

        返回:
            openai.OpenAI: 配置好DeepSeek API地址的OpenAI客户端实例

        注意:
            DeepSeek的默认API地址为 https://api.deepseek.com/v1
        """
        if base_url is None or base_url == "":
            base_url = "https://api.deepseek.com/v1"
        return openai.OpenAI(api_key=api_key, base_url=base_url)

    def get_service_name(self) -> str:
        """获取服务名称，用于重试日志记录。

        返回:
            str: 服务名称，固定返回"DeepSeek"
        """
        return "DeepSeek"

    def get_provider_name(self) -> str:
        """获取提供商名称，用于轨迹记录。

        该名称用于在轨迹文件中标识使用的LLM提供商。

        返回:
            str: 提供商名称，固定返回"deepseek"
        """
        return "deepseek"

    def get_extra_headers(self) -> dict[str, str]:
        """获取DeepSeek特定的请求头信息。

        DeepSeek目前不需要特殊的请求头，因此返回空字典。
        子类可以覆盖此方法来添加自定义头信息。

        返回:
            dict[str, str]: 空字典
        """
        return {}

    def supports_tool_calling(self, model_name: str) -> bool:
        """检查指定模型是否支持工具调用功能。

        工具调用允许LLM执行预定义的函数，扩展其交互能力。
        不同模型的工具调用支持情况可能不同。

        参数:
            model_name: 模型名称，如"deepseek-chat"、"deepseek-coder"等

        返回:
            bool: 如果模型支持工具调用返回True，否则返回False

        注意:
            支持工具调用的DeepSeek模型：
            - deepseek-chat: 通用对话模型，支持工具调用
            - deepseek-coder: 代码生成模型，支持工具调用

            不支持工具调用的模型：
            - deep-reasoner: 推理增强模型，主要用于复杂推理任务
        """
        # 支持工具调用的DeepSeek模型
        tool_capable_patterns = [
            "deepseek-chat",
            "deepseek-coder",
        ]
        return any(pattern in model_name.lower() for pattern in tool_capable_patterns)


class DeepSeekClient(OpenAICompatibleClient):
    """DeepSeek客户端包装类，保持OpenAI兼容性。

    该类继承自OpenAICompatibleClient，提供了DeepSeek特定的客户端实现。
    通过组合DeepSeekProvider配置，确保与DeepSeek API的正确交互。

    主要功能:
    - 自动配置DeepSeek API的默认地址
    - 继承OpenAI兼容客户端的所有功能，包括聊天、工具调用、重试机制等
    - 支持DeepSeek的模型特性和参数

    使用示例:
        >>> from trae_agent.utils.config import ModelConfig
        >>> config = ModelConfig(...)  # 配置模型
        >>> client = DeepSeekClient(config)
        >>> response = client.chat(messages, config, tools)
    """

    def __init__(self, model_config: ModelConfig):
        """初始化DeepSeek客户端。

        如果配置中没有提供base_url，则自动设置DeepSeek的默认API地址。
        然后调用父类构造函数，传入DeepSeekProvider配置。

        参数:
            model_config: 模型配置对象，包含API密钥、base_url等信息

        注意:
            如果model_config.model_provider.base_url为空或None，
            将自动设置为 "https://api.deepseek.com/v1"
        """
        if (
            model_config.model_provider.base_url is None
            or model_config.model_provider.base_url == ""
        ):
            model_config.model_provider.base_url = "https://api.deepseek.com/v1"
        super().__init__(model_config, DeepSeekProvider())