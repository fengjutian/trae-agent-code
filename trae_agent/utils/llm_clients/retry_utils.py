# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""重试工具模块。

该模块提供了在 API 调用失败时自动重试的装饰器功能。
主要用于处理 LLM API 调用时的临时性错误，如网络问题、服务暂时不可用等。

核心功能：
- 带有随机退避策略的重试机制
- 可配置的最大重试次数
- 详细的错误日志记录
- 保留原始函数的元信息（函数名、文档字符串等）

主要用途：
- LLM API 调用的错误处理
- 网络请求的自动重试
- 提高系统在临时故障下的稳定性

重试策略：
- 每次重试之间随机等待 3-30 秒
- 使用随机退避避免多个客户端同时重试导致的"惊群效应"
- 最大重试次数可配置（默认为 3 次）
"""

import random
import time
import traceback
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def retry_with(
    func: Callable[..., T],
    provider_name: str = "OpenAI",
    max_retries: int = 3,
) -> Callable[..., T]:
    """重试装饰器，为函数添加带有随机退避的重试逻辑。

    该装饰器用于包装可能会因临时性错误而失败的函数（如 API 调用）。
    当函数抛出异常时，会自动进行重试，重试次数达到上限后才抛出异常。

    Args:
        func: 要装饰的函数。该函数在失败时会被重试。
              函数可以是任何可调用对象，通常会抛出异常来指示失败。
        provider_name: 模型提供商的名称，用于日志记录。
                      默认值为 "OpenAI"，但可以是 "DeepSeek"、"Doubao"、"Anthropic" 等。
                      该名称会出现在错误日志中，帮助识别是哪个服务的调用失败了。
        max_retries: 最大重试次数。包括首次调用在内的总尝试次数为 max_retries + 1。
                    默认值为 3，表示总共尝试 4 次（1 次初始调用 + 3 次重试）。

    Returns:
        带有重试逻辑的装饰函数。该函数的签名和原始函数完全相同。

    重试机制：
    1. 首次调用函数，如果成功则直接返回结果
    2. 如果失败，捕获异常并记录错误信息
    3. 随机等待 3-30 秒（随机退避）
    4. 重试函数调用，重复步骤 1-3
    5. 如果重试次数达到 max_retries，抛出最后一次的异常

    随机退避策略：
    - 每次重试前等待 3-30 秒之间的随机时间
    - 使用 random.randint(3, 30) 生成随机等待时间
    - 随机性可以避免多个客户端同时重试导致的"惊群效应"
    - 给服务器时间从临时故障中恢复

    错误日志：
    - 每次重试失败时会打印错误信息
    - 日志包含提供商名称、错误消息、等待时间
    - 使用 traceback.format_exc() 打印完整的堆栈跟踪
    - 便于调试和问题排查

    使用示例：
        # 装饰一个 API 调用函数
        @retry_with(provider_name="DeepSeek", max_retries=5)
        def call_deepseek_api(prompt: str) -> str:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={"messages": [{"role": "user", "content": prompt}]}
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        # 调用该函数，如果失败会自动重试
        result = call_deepseek_api("Hello, world!")

        # 直接作为装饰器使用
        from trae_agent.utils.llm_clients.retry_utils import retry_with

        @retry_with(provider_name="OpenAI", max_retries=3)
        def openai_chat(messages):
            # ... API 调用代码 ...
            pass

    注意事项：
    - 被装饰的函数必须是幂等的，即多次调用产生相同的结果（或至少是可接受的）
    - 重试会增加总执行时间，特别是当 API 服务不可用时
    - 确保重试不会产生副作用（如重复写入数据库）
    - 随机退避时间范围（3-30 秒）可以根据实际需求调整
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        """包装函数，实现重试逻辑。

        Args:
            *args: 位置参数，传递给原始函数
            **kwargs: 关键字参数，传递给原始函数

        Returns:
            原始函数的返回值

        Raises:
            Exception: 当所有重试都失败时，抛出最后一次的异常
        """
        last_exception = None  # 保存最后一次异常，用于最终抛出

        # 尝试调用函数，最多重试 max_retries 次
        # range(max_retries + 1) 包括初始调用，所以总尝试次数为 max_retries + 1
        for attempt in range(max_retries + 1):
            try:
                # 尝试调用原始函数
                return func(*args, **kwargs)
            except Exception as e:
                # 捕获异常并保存
                last_exception = e

                # 如果是最后一次尝试，直接抛出异常
                if attempt == max_retries:
                    raise

                # 计算随机退避时间（3-30 秒）
                sleep_time = random.randint(3, 30)
                this_error_message = str(e)

                # 打印错误日志
                print(
                    f"{provider_name} API call failed: {this_error_message}. "
                    f"Will sleep for {sleep_time} seconds and will retry.\n{traceback.format_exc()}"
                )

                # 等待随机时间后重试
                time.sleep(sleep_time)

        # 这行代码理论上不会执行，因为最后一次失败会直接抛出异常
        # 但为了类型检查器的安全，保留此行代码
        raise last_exception or Exception("Retry failed for unknown reason")

    return wrapper
