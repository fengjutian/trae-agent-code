# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Doubao client wrapper with tool integrations

Doubao（豆包）是字节跳动旗下的大语言模型服务，提供高性能的通用对话和任务处理能力。
该模块实现了Doubao API的客户端封装，支持工具调用和OpenAI兼容接口。
通过OpenAI兼容的API接口，可以方便地集成Doubao模型到现有的应用中。
"""

import openai

from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.openai_compatible_base import (
    OpenAICompatibleClient,
    ProviderConfig,
)


class DoubaoProvider(ProviderConfig):
    """Doubao提供商配置类。

    实现了ProviderConfig接口，提供Doubao特定的配置和行为。
    包括客户端创建、工具调用支持判断、额外参数和头信息的处理。

    特点:
    - Doubao模型普遍支持工具调用功能
    - 使用OpenAI兼容的API接口
    - 不需要额外的请求头信息
    """

    def create_client(
        self, api_key: str, base_url: str | None, api_version: str | None
    ) -> openai.OpenAI:
        """创建Doubao的OpenAI兼容客户端。

        使用OpenAI SDK连接Doubao API，因为Doubao提供了OpenAI兼容的API接口。

        参数:
            api_key: Doubao API密钥，用于身份验证
            base_url: API基础URL，Doubao提供的API地址
            api_version: API版本（Doubao当前不使用此参数）

        返回:
            openai.OpenAI: 配置好Doubao API地址的OpenAI客户端实例

        注意:
            - base_url参数必须由调用者提供，不像DeepSeek那样有默认值
            - 确保api_key有效且具有足够的权限
        """
        return openai.OpenAI(base_url=base_url, api_key=api_key)

    def get_service_name(self) -> str:
        """获取服务名称，用于重试日志记录。

        返回:
            str: 服务名称，固定返回"Doubao"

        注意:
            该名称会在API调用失败时出现在重试日志中，便于问题排查
        """
        return "Doubao"

    def get_provider_name(self) -> str:
        """获取提供商名称，用于轨迹记录。

        该名称用于在轨迹文件中标识使用的LLM提供商。

        返回:
            str: 提供商名称，固定返回"doubao"

        注意:
            轨迹记录用于分析和调试LLM交互过程
        """
        return "doubao"

    def get_extra_headers(self) -> dict[str, str]:
        """获取Doubao特定的请求头信息。

        Doubao目前不需要特殊的请求头，因此返回空字典。
        如果未来需要添加自定义头信息，可以在此方法中实现。

        返回:
            dict[str, str]: 空字典，表示不需要额外的请求头
        """
        return {}

    def supports_tool_calling(self, model_name: str) -> bool:
        """检查指定模型是否支持工具调用功能。

        工具调用允许LLM执行预定义的函数，扩展其交互能力。
        Doubao模型普遍支持工具调用功能。

        参数:
            model_name: 模型名称，如"doubao-pro"、"doubao-lite"等

        返回:
            bool: 固定返回True，表示Doubao模型都支持工具调用

        注意:
            - Doubao系列模型（包括不同版本和规模）普遍支持工具调用
            - 如果未来某些特殊模型不支持工具调用，可以在此方法中添加判断逻辑
        """
        # Doubao模型普遍支持工具调用
        return True


class DoubaoClient(OpenAICompatibleClient):
    """Doubao客户端包装类，保持兼容性的同时使用新架构。

    该类继承自OpenAICompatibleClient，提供了Doubao特定的客户端实现。
    通过组合DoubaoProvider配置，确保与Doubao API的正确交互。

    主要功能:
    - 自动使用OpenAI兼容接口连接Doubao API
    - 继承OpenAI兼容客户端的所有功能，包括聊天、工具调用、重试机制等
    - 支持Doubao的模型特性和参数
    - 完全兼容新的架构设计，便于维护和扩展

    使用示例:
        >>> from trae_agent.utils.config import ModelConfig
        >>> config = ModelConfig(...)  # 配置模型，包括base_url和api_key
        >>> client = DoubaoClient(config)
        >>> response = client.chat(messages, config, tools)

    注意:
        - 必须在model_config中提供有效的base_url和api_key
        - Doubao客户端使用与DeepSeek相同的底层架构，便于统一管理
    """

    def __init__(self, model_config: ModelConfig):
        """初始化Doubao客户端。

        直接调用父类构造函数，传入DoubaoProvider配置。

        参数:
            model_config: 模型配置对象，包含API密钥、base_url等信息

        注意:
            - 与DeepSeek不同，Doubao不会自动设置默认的base_url
            - 确保在model_config中正确配置了base_url和api_key
        """
        super().__init__(model_config, DoubaoProvider())
