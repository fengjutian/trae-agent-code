# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT


from dataclasses import dataclass

from trae_agent.tools.base import ToolCall, ToolResult


@dataclass
class LLMMessage:
    """标准LLM消息格式。

    定义了与LLM交互时的标准消息结构，支持多种消息类型：
    - 普通对话消息（角色+内容）
    - 工具调用消息（LLM请求调用工具）
    - 工具结果消息（工具执行结果的返回）

    属性:
        role: 消息角色，可以是"system"、"user"、"assistant"或"function"
        content: 消息内容，对于文本消息为主要文本内容
        tool_call: 工具调用对象，当assistant角色请求调用工具时使用
        tool_result: 工具执行结果，当返回工具调用结果时使用

    使用示例:
        >>> # 普通用户消息
        >>> user_msg = LLMMessage(role="user", content="你好")
        
        >>> # 助手消息
        >>> assistant_msg = LLMMessage(role="assistant", content="你好！有什么可以帮助你的？")
        
        >>> # 工具调用消息
        >>> tool_msg = LLMMessage(role="assistant", tool_call=ToolCall(...))
        
        >>> # 工具结果消息
        >>> result_msg = LLMMessage(role="function", tool_result=ToolResult(...))

    注意:
        - content、tool_call、tool_result三者为互斥关系，同一消息只能有一种
        - "function"角色专门用于工具调用结果
        - "assistant"角色可以包含tool_call表示要调用工具
    """

    role: str
    content: str | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None


@dataclass
class LLMUsage:
    """LLM使用量统计格式。

    记录LLM调用的令牌（token）使用情况，包括各种类型的使用量统计。
    不同的LLM提供商可能提供不同类型的令牌统计信息。

    属性:
        input_tokens: 输入令牌数，发送给LLM的文本所消耗的令牌
        output_tokens: 输出令牌数，LLM生成的响应所消耗的令牌
        cache_creation_input_tokens: 缓存创建输入令牌数（默认0）
                                   用于记录创建新缓存时消耗的输入令牌
        cache_read_input_tokens: 缓存读取输入令牌数（默认0）
                                 用于记录从缓存读取时消耗的输入令牌
        reasoning_tokens: 推理令牌数（默认0）
                        用于记录推理增强模型的推理过程令牌数

    使用示例:
        >>> usage = LLMUsage(
        ...     input_tokens=1000,
        ...     output_tokens=500,
        ...     cache_read_input_tokens=200
        ... )
        >>> print(usage)
        LLMUsage(input_tokens=1000, output_tokens=500, ...)

        >>> # 累加多个请求的使用量
        >>> total_usage = usage1 + usage2

    注意:
        - cache_creation_input_tokens和cache_read_input_tokens用于支持缓存的模型
        - reasoning_tokens用于支持推理增强模式的模型（如DeepSeek的deep-reasoner）
        - 总成本计算应该考虑所有类型的令牌，不同类型可能有不同的定价
    """

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        """累加两个LLMUsage对象。

        用于统计多次LLM调用的总使用量。

        参数:
            other: 另一个LLMUsage对象

        返回:
            LLMUsage: 新的LLMUsage对象，包含累加后的令牌数

        使用示例:
            >>> usage1 = LLMUsage(input_tokens=100, output_tokens=50)
            >>> usage2 = LLMUsage(input_tokens=200, output_tokens=100)
            >>> total = usage1 + usage2
            >>> total.input_tokens  # 300
            >>> total.output_tokens  # 150
        """
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def __str__(self) -> str:
        """返回LLMUsage对象的字符串表示。

        用于调试和日志记录。

        返回:
            str: 包含所有令牌统计信息的字符串

        示例:
            >>> usage = LLMUsage(input_tokens=100, output_tokens=50)
            >>> print(usage)
            LLMUsage(input_tokens=100, output_tokens=50, cache_creation_input_tokens=0, cache_read_input_tokens=0, reasoning_tokens=0)
        """
        return f"LLMUsage(input_tokens={self.input_tokens}, output_tokens={self.output_tokens}, cache_creation_input_tokens={self.cache_creation_input_tokens}, cache_read_input_tokens={self.cache_read_input_tokens}, reasoning_tokens={self.reasoning_tokens})"


@dataclass
class LLMResponse:
    """标准LLM响应格式。

    定义了LLM返回的响应结构，包含响应内容、使用量、状态等信息。

    属性:
        content: 响应内容，LLM生成的文本
        usage: 使用量统计，记录本次调用的令牌消耗（可选）
        model: 模型名称，标识使用的具体模型版本（可选）
        finish_reason: 结束原因，说明响应结束的原因（可选）
                      常见值: "stop"（正常结束）、"length"（达到最大长度）、
                              "tool_calls"（需要调用工具）、"content_filter"（内容被过滤）
        tool_calls: 工具调用列表，当LLM请求调用工具时返回（可选）
                   每个ToolCall包含工具名称、调用ID和参数

    使用示例:
        >>> # 普通文本响应
        >>> response = LLMResponse(
        ...     content="这是一个答案",
        ...     model="deepseek-chat",
        ...     finish_reason="stop"
        ... )

        >>> # 工具调用响应
        >>> response = LLMResponse(
        ...     content="我将帮你调用搜索工具",
        ...     tool_calls=[ToolCall(name="search", call_id="123", arguments={"query": "AI"})],
        ...     model="deepseek-chat",
        ...     finish_reason="tool_calls"
        ... )

        >>> # 带使用量的响应
        >>> response = LLMResponse(
        ...     content="答案内容",
        ...     usage=LLMUsage(input_tokens=100, output_tokens=50),
        ...     model="deepseek-chat"
        ... )

    注意:
        - content和tool_calls可以同时存在，LLM可能在调用工具前给出一些说明
        - finish_reason对于调试和优化很重要，可以了解LLM的决策过程
        - usage信息对于成本控制和性能分析很有价值
    """

    content: str
    usage: LLMUsage | None = None
    model: str | None = None
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None

