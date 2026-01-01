# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT


from abc import ABC, abstractmethod

from trae_agent.tools.base import Tool
from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse
from trae_agent.utils.trajectory_recorder import TrajectoryRecorder


class BaseLLMClient(ABC):
    """LLM客户端基类。

    定义了所有LLM客户端必须实现的核心接口和通用功能。
    该基类提供了轨迹记录、配置管理等通用能力，具体实现由子类完成。

    属性:
        api_key: API访问密钥
        base_url: API基础URL（可选，某些提供商不需要）
        api_version: API版本（可选，某些提供商不需要）
        trajectory_recorder: 轨迹记录器实例，用于记录LLM交互过程
    """

    def __init__(self, model_config: ModelConfig):
        """初始化LLM客户端。

        参数:
            model_config: 模型配置对象，包含API密钥、URL等连接信息
        """
        self.api_key: str = model_config.model_provider.api_key
        self.base_url: str | None = model_config.model_provider.base_url
        self.api_version: str | None = model_config.model_provider.api_version
        self.trajectory_recorder: TrajectoryRecorder | None = None  # 轨迹记录器实例，用于记录LLM交互

    def set_trajectory_recorder(self, recorder: TrajectoryRecorder | None) -> None:
        """设置轨迹记录器。

        轨迹记录器用于记录LLM的交互历史，包括消息、响应、工具调用等信息，
        便于调试、分析和优化。

        参数:
            recorder: 轨迹记录器实例，传入None表示取消记录
        """
        self.trajectory_recorder = recorder

    @abstractmethod
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """设置聊天历史记录。

        用于初始化或更新客户端的对话上下文。某些场景下需要预加载历史消息
        以保持对话的连续性。

        参数:
            messages: 消息列表，包含角色、内容等信息
        """
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """发送聊天消息到LLM并获取响应。

        这是LLM交互的核心方法，支持工具调用和消息历史管理。

        参数:
            messages: 要发送的消息列表
            model_config: 模型配置，包括温度、令牌限制等参数
            tools: 工具列表（可选），用于函数调用场景
            reuse_history: 是否复用历史消息，True表示在现有历史基础上追加消息，
                          False表示仅使用当前消息

        返回:
            LLMResponse: LLM的响应对象，包含内容、工具调用、使用量等信息

        注意:
            - 当启用工具调用时，LLM可能返回工具调用而非文本内容
            - 消息历史管理对多轮对话很重要
        """
        pass

    def supports_tool_calling(self, model_config: ModelConfig) -> bool:
        """检查当前模型是否支持工具调用功能。

        工具调用允许LLM执行特定的函数或命令，扩展了其能力范围。

        参数:
            model_config: 模型配置对象

        返回:
            bool: 如果模型支持工具调用返回True，否则返回False

        注意:
            此方法使用配置中的supports_tool_calling标志，实际支持情况
            可能因模型版本和提供商策略而异
        """
        return model_config.supports_tool_calling
