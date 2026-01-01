# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

# TODO: remove these annotations by defining fine-grained types
# pyright: reportExplicitAny=false
# pyright: reportArgumentType=false
# pyright: reportAny=false

"""轨迹记录模块。

该模块提供了 TrajectoryRecorder 类，用于记录和保存 AI Agent 执行过程中的完整轨迹数据。
轨迹数据包括：
- 任务信息（任务描述、开始/结束时间、执行状态等）
- LLM 交互记录（输入消息、输出响应、token 使用情况等）
- Agent 执行步骤（每一步的状态、工具调用、工具结果、反思等）
- Lakeview 分析摘要

轨迹数据以 JSON 格式保存到文件中，便于后续分析和调试。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from trae_agent.tools.base import ToolCall, ToolResult
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse


class TrajectoryRecorder:
    """轨迹记录器类。

    该类负责记录和保存 AI Agent 执行过程中的完整轨迹数据。
    主要功能：
    1. 记录任务的基本信息（任务描述、模型配置、执行时间等）
    2. 记录每次 LLM 调用的详细信息（输入消息、输出响应、token 使用等）
    3. 记录每个 Agent 执行步骤的完整信息（状态、工具调用、结果、反思等）
    4. 保存轨迹数据到 JSON 文件，便于后续分析和可视化
    5. 支持增量保存，每次记录后立即更新文件

    Attributes:
        trajectory_path: 轨迹文件保存路径（Path 对象）
        trajectory_data: 轨迹数据字典，包含任务信息、LLM 交互记录、Agent 步骤等
        _start_time: 任务开始时间（datetime 对象），用于计算总执行时间
    """

    def __init__(self, trajectory_path: str | None = None):
        """初始化轨迹记录器。

        如果未指定轨迹文件路径，则自动生成默认路径：
        格式为 trajectories/trajectory_{timestamp}.json
        其中 timestamp 为当前时间的格式化字符串（YYYYMMDD_HHMMSS）

        Args:
            trajectory_path: 轨迹文件保存路径。如果为 None，则自动生成默认路径。
                            可以是相对路径或绝对路径。

        初始化时会自动创建轨迹数据字典，包含以下字段：
        - task: 任务描述（初始为空字符串）
        - start_time: 任务开始时间（初始为空字符串）
        - end_time: 任务结束时间（初始为空字符串）
        - provider: LLM 提供商名称（初始为空字符串）
        - model: 使用的模型名称（初始为空字符串）
        - max_steps: 最大执行步数（初始为 0）
        - llm_interactions: LLM 交互记录列表（初始为空列表）
        - agent_steps: Agent 执行步骤列表（初始为空列表）
        - success: 任务是否成功（初始为 False）
        - final_result: 任务最终结果（初始为 None）
        - execution_time: 总执行时间（秒，初始为 0.0）
        """
        # 如果未指定路径，自动生成带时间戳的文件名
        if trajectory_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trajectory_path = f"trajectories/trajectory_{timestamp}.json"

        # 将路径转换为绝对路径并保存
        self.trajectory_path: Path = Path(trajectory_path).resolve()

        # 尝试创建轨迹文件所在目录
        try:
            self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            print("Error creating trajectory directory. Trajectories may not be properly saved.")

        # 初始化轨迹数据字典
        self.trajectory_data: dict[str, Any] = {
            "task": "",
            "start_time": "",
            "end_time": "",
            "provider": "",
            "model": "",
            "max_steps": 0,
            "llm_interactions": [],
            "agent_steps": [],
            "success": False,
            "final_result": None,
            "execution_time": 0.0,
        }

        # 记录任务开始时间，用于后续计算总执行时间
        self._start_time: datetime | None = None

    def start_recording(self, task: str, provider: str, model: str, max_steps: int) -> None:
        """开始记录新的轨迹。

        该方法用于初始化一条新的轨迹记录，设置任务基本信息。
        调用后会立即保存一次轨迹文件，确保数据持久化。

        Args:
            task: 待执行的任务描述字符串
            provider: 使用的 LLM 提供商名称（如 "openai", "deepseek", "doubao" 等）
            model: 使用的模型名称（如 "gpt-4", "deepseek-chat", "doubao-pro" 等）
            max_steps: Agent 执行的最大步数限制

        该方法会执行以下操作：
        1. 记录当前时间作为任务开始时间
        2. 更新轨迹数据字典中的任务信息字段
        3. 重置 llm_interactions 和 agent_steps 为空列表（清除之前的数据）
        4. 调用 save_trajectory() 保存初始状态到文件
        """
        # 记录任务开始时间
        self._start_time = datetime.now()

        # 更新轨迹数据字典中的任务信息
        self.trajectory_data.update(
            {
                "task": task,
                "start_time": self._start_time.isoformat(),
                "provider": provider,
                "model": model,
                "max_steps": max_steps,
                "llm_interactions": [],  # 清空之前的 LLM 交互记录
                "agent_steps": [],  # 清空之前的 Agent 步骤记录
            }
        )

        # 立即保存轨迹文件
        self.save_trajectory()

    def record_llm_interaction(
        self,
        messages: list[LLMMessage],
        response: LLMResponse,
        provider: str,
        model: str,
        tools: list[Any] | None = None,
    ) -> None:
        """记录一次 LLM 交互。

        该方法详细记录每次 LLM API 调用的完整信息，包括输入消息、输出响应、
        token 使用情况、工具调用等。记录后会立即保存到文件。

        Args:
            messages: 输入给 LLM 的消息列表（LLMMessage 对象列表）
            response: LLM 返回的响应对象（LLMResponse 对象）
            provider: 使用的 LLM 提供商名称
            model: 使用的模型名称
            tools: 本次交互时可用的工具列表（Tool 对象列表）。如果为 None，表示无工具可用。

        记录的交互数据包含：
        - timestamp: 交互时间戳（ISO 格式）
        - provider/model: 使用的提供商和模型
        - input_messages: 输入消息序列化后的列表
        - response: 响应的详细内容，包括：
            - content: 响应文本内容
            - model: 实际使用的模型名称
            - finish_reason: 结束原因（如 "stop", "tool_calls" 等）
            - usage: token 使用情况，包括：
                - input_tokens: 输入 token 数
                - output_tokens: 输出 token 数
                - cache_creation_input_tokens: 缓存创建的输入 token（如果支持）
                - cache_read_input_tokens: 缓存读取的输入 token（如果支持）
                - reasoning_tokens: 推理 token 数（某些模型如 DeepSeek 支持）
            - tool_calls: 模型请求的工具调用列表
        - tools_available: 本次交互可用的工具名称列表
        """
        # 构建交互记录字典
        interaction = {
            "timestamp": datetime.now().isoformat(),  # 记录交互时间
            "provider": provider,
            "model": model,
            "input_messages": [self._serialize_message(msg) for msg in messages],  # 序列化输入消息
            "response": {
                "content": response.content,  # 响应内容
                "model": response.model,  # 实际使用的模型
                "finish_reason": response.finish_reason,  # 结束原因
                "usage": {
                    "input_tokens": response.usage.input_tokens if response.usage else 0,
                    "output_tokens": response.usage.output_tokens if response.usage else 0,
                    # 缓存相关的 token 使用情况（如果模型支持）
                    "cache_creation_input_tokens": getattr(
                        response.usage, "cache_creation_input_tokens", None
                    )
                    if response.usage
                    else None,
                    "cache_read_input_tokens": getattr(
                        response.usage, "cache_read_input_tokens", None
                    )
                    if response.usage
                    else None,
                    # 推理 token（如 DeepSeek 的推理模型）
                    "reasoning_tokens": getattr(response.usage, "reasoning_tokens", None)
                    if response.usage
                    else None,
                },
                # 工具调用记录
                "tool_calls": [self._serialize_tool_call(tc) for tc in response.tool_calls]
                if response.tool_calls
                else None,
            },
            # 记录本次可用的工具列表
            "tools_available": [tool.name for tool in tools] if tools else None,
        }

        # 将交互记录添加到轨迹数据中
        self.trajectory_data["llm_interactions"].append(interaction)

        # 立即保存轨迹文件
        self.save_trajectory()

    def record_agent_step(
        self,
        step_number: int,
        state: str,
        llm_messages: list[LLMMessage] | None = None,
        llm_response: LLMResponse | None = None,
        tool_calls: list[ToolCall] | None = None,
        tool_results: list[ToolResult] | None = None,
        reflection: str | None = None,
        error: str | None = None,
    ) -> None:
        """记录一个 Agent 执行步骤。

        该方法记录 Agent 执行过程中的每个步骤的完整信息，包括 LLM 调用、
        工具调用、工具结果、反思和错误等。这是 Agent 行为分析的关键数据源。

        Args:
            step_number: 步骤编号（从 1 开始计数）
            state: 当前 Agent 状态（如 "thinking", "tool_execution", "reflection" 等）
            llm_messages: 本步骤发送给 LLM 的消息列表。如果为 None，表示本步骤无 LLM 调用。
            llm_response: 本步骤 LLM 返回的响应。如果为 None，表示无响应或步骤失败。
            tool_calls: 本步骤执行的工具调用列表。如果为 None，表示无工具调用。
            tool_results: 本步骤工具执行的结果列表。如果为 None，表示无工具结果。
            reflection: Agent 对本步骤的反思内容。如果为 None，表示无反思。
            error: 本步骤失败时的错误消息。如果为 None，表示步骤成功。

        记录的步骤数据包含：
        - step_number: 步骤编号
        - timestamp: 步骤时间戳（ISO 格式）
        - state: Agent 状态
        - llm_messages: LLM 输入消息序列化列表（如果存在）
        - llm_response: LLM 响应详细信息（如果存在），包括：
            - content: 响应内容
            - model: 使用的模型
            - finish_reason: 结束原因
            - usage: token 使用情况
            - tool_calls: 工具调用列表
        - tool_calls: 工具调用序列化列表（如果存在）
        - tool_results: 工具结果序列化列表（如果存在）
        - reflection: 反思内容（如果存在）
        - error: 错误信息（如果存在）
        - lakeview_summary: Lakeview 分析摘要（通过 update_lakeview 方法后续添加）
        """
        # 构建步骤数据字典
        step_data = {
            "step_number": step_number,  # 步骤编号
            "timestamp": datetime.now().isoformat(),  # 步骤时间
            "state": state,  # Agent 状态
            # LLM 消息记录
            "llm_messages": [self._serialize_message(msg) for msg in llm_messages]
            if llm_messages
            else None,
            # LLM 响应记录
            "llm_response": {
                "content": llm_response.content,
                "model": llm_response.model,
                "finish_reason": llm_response.finish_reason,
                "usage": {
                    "input_tokens": llm_response.usage.input_tokens if llm_response.usage else None,
                    "output_tokens": llm_response.usage.output_tokens
                    if llm_response.usage
                    else None,
                }
                if llm_response.usage
                else None,
                "tool_calls": [self._serialize_tool_call(tc) for tc in llm_response.tool_calls]
                if llm_response.tool_calls
                else None,
            }
            if llm_response
            else None,
            # 工具调用记录
            "tool_calls": [self._serialize_tool_call(tc) for tc in tool_calls]
            if tool_calls
            else None,
            # 工具结果记录
            "tool_results": [self._serialize_tool_result(tr) for tr in tool_results]
            if tool_results
            else None,
            # 反思和错误信息
            "reflection": reflection,
            "error": error,
        }

        # 将步骤数据添加到轨迹数据中
        self.trajectory_data["agent_steps"].append(step_data)

        # 立即保存轨迹文件
        self.save_trajectory()

    def update_lakeview(self, step_number: int, lakeview_summary: str):
        """更新指定步骤的 Lakeview 分析摘要。

        Lakeview 是一个用于分析和可视化 Agent 执行步骤的工具，可以为每个步骤
        生成分析摘要。该方法将摘要信息添加到对应的步骤记录中。

        Args:
            step_number: 要更新的步骤编号
            lakeview_summary: Lakeview 生成的分析摘要字符串

        该方法会在 agent_steps 列表中查找匹配的步骤，并添加 lakeview_summary 字段。
        如果找不到匹配的步骤，则不进行任何操作。
        """
        # 遍历所有步骤，查找匹配的步骤编号
        for step_data in self.trajectory_data["agent_steps"]:
            if step_data["step_number"] == step_number:
                # 找到匹配的步骤，添加 Lakeview 摘要
                step_data["lakeview_summary"] = lakeview_summary
                break

        # 保存更新后的轨迹文件
        self.save_trajectory()

    def finalize_recording(self, success: bool, final_result: str | None = None) -> None:
        """完成轨迹记录。

        该方法在任务执行完成或失败时调用，用于设置最终状态并保存完整的轨迹数据。
        会计算并记录总执行时间。

        Args:
            success: 任务是否成功完成（True/False）
            final_result: 任务的最终结果或输出字符串。如果为 None，则不记录结果。

        该方法会执行以下操作：
        1. 记录当前时间作为任务结束时间
        2. 设置任务执行成功状态
        3. 记录最终结果（如果提供）
        4. 计算总执行时间（从开始时间到结束时间，单位为秒）
        5. 调用 save_trajectory() 保存完整轨迹到文件

        注意：如果在任务执行过程中 _start_time 未被正确设置（即 start_recording 未被调用），
        则 execution_time 会被设置为 0.0。
        """
        # 记录任务结束时间
        end_time = datetime.now()

        # 更新轨迹数据中的结束状态
        self.trajectory_data.update(
            {
                "end_time": end_time.isoformat(),  # 结束时间
                "success": success,  # 执行成功标志
                "final_result": final_result,  # 最终结果
                # 计算总执行时间（秒）
                "execution_time": (end_time - self._start_time).total_seconds()
                if self._start_time
                else 0.0,
            }
        )

        # 保存完整的轨迹文件
        self.save_trajectory()

    def save_trajectory(self) -> None:
        """保存当前轨迹数据到文件。

        该方法将 trajectory_data 字典序列化为 JSON 格式并写入到 trajectory_path 指定的文件中。
        使用 UTF-8 编码，并保留中文字符（ensure_ascii=False）。
        每次调用都会创建缩进为 2 个空格的格式化 JSON，便于阅读。

        如果保存过程中发生异常（如权限问题、磁盘空间不足等），会打印警告信息
        但不会抛出异常，以确保轨迹记录失败不会影响主程序的执行。

        注意：该方法会在每次记录操作后自动调用，确保数据及时持久化。
        """
        try:
            # 确保轨迹文件所在目录存在
            self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)

            # 以 UTF-8 编码写入 JSON 文件
            with open(self.trajectory_path, "w", encoding="utf-8") as f:
                # indent=2 保留缩进格式，ensure_ascii=False 保留中文字符
                json.dump(self.trajectory_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            # 保存失败时打印警告，但不抛出异常
            print(f"Warning: Failed to save trajectory to {self.trajectory_path}: {e}")

    def _serialize_message(self, message: LLMMessage) -> dict[str, Any]:
        """将 LLM 消息对象序列化为字典。

        该方法将 LLMMessage 对象转换为可 JSON 序列化的字典格式，
        用于保存到轨迹文件中。

        Args:
            message: 要序列化的 LLMMessage 对象

        Returns:
            包含消息信息的字典，包括：
            - role: 消息角色（如 "user", "assistant", "system", "tool"）
            - content: 消息内容
            - tool_call: 工具调用信息（如果消息包含工具调用）
            - tool_result: 工具结果信息（如果消息包含工具结果）
        """
        # 基本字段：角色和内容
        data: dict[str, Any] = {"role": message.role, "content": message.content}

        # 如果消息包含工具调用，序列化工具调用
        if message.tool_call:
            data["tool_call"] = self._serialize_tool_call(message.tool_call)

        # 如果消息包含工具结果，序列化工具结果
        if message.tool_result:
            data["tool_result"] = self._serialize_tool_result(message.tool_result)

        return data

    def _serialize_tool_call(self, tool_call: ToolCall) -> dict[str, Any]:
        """将工具调用对象序列化为字典。

        Args:
            tool_call: 要序列化的 ToolCall 对象

        Returns:
            包含工具调用信息的字典，包括：
            - call_id: 工具调用唯一标识符
            - name: 工具名称
            - arguments: 工具参数（通常为 JSON 字符串）
            - id: 可选的 ID 字段（如果对象中存在）
        """
        return {
            "call_id": tool_call.call_id,
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            # 使用 getattr 安全获取可选的 id 字段
            "id": getattr(tool_call, "id", None),
        }

    def _serialize_tool_result(self, tool_result: ToolResult) -> dict[str, Any]:
        """将工具结果对象序列化为字典。

        Args:
            tool_result: 要序列化的 ToolResult 对象

        Returns:
            包含工具结果信息的字典，包括：
            - call_id: 工具调用唯一标识符（与 tool_call 对应）
            - success: 工具执行是否成功（True/False）
            - result: 工具执行结果（成功时）
            - error: 错误信息（失败时）
            - id: 可选的 ID 字段（如果对象中存在）
        """
        return {
            "call_id": tool_result.call_id,
            "success": tool_result.success,
            "result": tool_result.result,
            "error": tool_result.error,
            # 使用 getattr 安全获取可选的 id 字段
            "id": getattr(tool_result, "id", None),
        }

    def get_trajectory_path(self) -> str:
        """获取轨迹文件保存路径。

        Returns:
            轨迹文件的绝对路径字符串。

        该方法返回轨迹文件的完整路径，可用于日志输出、用户提示等场景。
        """
        return str(self.trajectory_path)
