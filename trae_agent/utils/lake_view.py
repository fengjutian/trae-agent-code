# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Lake View - AI Agent步骤分析和可视化模块。

该模块用于分析和标记AI Agent在解决软件bug过程中的各个步骤。
通过LLM分析agent的行为轨迹，提取任务描述和分类标签，便于理解和可视化agent的工作流程。

主要功能:
- 从agent步骤中提取任务描述（简洁概括+详细说明）
- 为每个步骤自动分配行为标签（如测试、代码审查、修复等）
- 提供友好的emoji表示，便于可视化展示

设计理念:
- 使用LLM自动分析agent行为，减少人工标注成本
- 两级粒度描述：简洁的任务标签+详细的bug特定信息
- 预定义的行为分类，便于统计分析

使用场景:
- Agent行为分析和调试
- 自动化测试报告生成
- Agent性能评估
- 交互式agent界面展示
"""

import re
from dataclasses import dataclass

from trae_agent.agent.agent_basics import AgentStep
from trae_agent.utils.config import LakeviewConfig
from trae_agent.utils.llm_clients.llm_basics import LLMMessage
from trae_agent.utils.llm_clients.llm_client import LLMClient

StepType = tuple[
    str,  # 用于人类阅读的内容（会写入结果文件）
    str
    | None,  # 用于LLM分析的内容，如果不需要分析则为None（即次要步骤），注意长度限制
]
"""步骤类型定义。

定义了Lake View系统中表示agent步骤的数据结构。
第一个元素: 人类可读的内容描述
第二个元素: LLM分析用内容（可能为None）
"""


EXTRACTOR_PROMPT = """
Given the preceding excerpt, your job is to determine "what task is the agent performing in <this_step>".
Output your answer in two granularities: <task>...</task><details>...</details>.
In the <task> tag, the answer should be concise and general. It should omit ANY bug-specific details, and contain at most 10 words.
In the <details> tag, the answer should complement the <task> tag by adding bug-specific details. It should be informative and contain at most 30 words.

Examples:

<task>The agent is writing a reproduction test script.</task><details>The agent is writing "test_bug.py" to reproduce the bug in XXX-Project's create_foo method not comparing sizes correctly.</details>
<task>The agent is examining source code.</task><details>The agent is searching for "function_name" in the code repository, that is related to the "foo.py:function_name" line in the stack trace.</details>
<task>The agent is fixing the reproduction test script.</task><details>The agent is fixing "test_bug.py" that forgets to import the function "foo", causing a NameError.</details>

Now, answer the question "what task is the agent performing in <this_step>".
Again, provide only the answer with no other commentary. The format should be "<task>...</task><details>...</details>".
"""
"""任务提取提示词。

该提示词用于指导LLM从agent步骤中提取任务描述。
采用两级粒度的输出格式：
1. <task>: 简洁概括任务（最多10个单词），不包含bug特定细节
2. <details>: 补充bug特定的详细信息（最多30个单词）

设计要点:
- 任务描述要通用化，便于跨不同bug的对比分析
- 详细说明要包含具体的bug信息，便于理解上下文
- 提供多个示例帮助LLM理解期望的输出格式

输出格式:
<task>简洁任务描述</task><details>bug特定详细信息</details>
"""

TAGGER_PROMPT = """
Given the trajectory, your job is to determine "what task is the agent performing in the current step".
Output your answer by choosing the applicable tags in the below list for the current step.
If it is performing multiple tasks in one step, choose ALL applicable tags, separated by a comma.

<tags>
WRITE_TEST: It writes a test script to reproduce the bug, or modifies a non-working test script to fix problems found in testing.
VERIFY_TEST: It runs the reproduction test script to verify the testing environment is working.
EXAMINE_CODE: It views, searches, or explores the code repository to understand the cause of the bug.
WRITE_FIX: It modifies the source code to fix the identified bug.
VERIFY_FIX: It runs the reproduction test or existing tests to verify the fix indeed solves the bug.
REPORT: It reports to the user that the job is completed or some progress has been made.
THINK: It analyzes the bug through thinking, but does not perform concrete actions right now.
OUTLIER: A major part in this step does not fit into any tag above, such as running a shell command to install dependencies.
</tags>

<examples>
If the agent is opening a file to examine, output <tags>EXAMINE_CODE</tags>.
If the agent is fixing a known problem in the reproduction test script and then running it again, output <tags>WRITE_TEST,VERIFY_TEST</tags>.
If the agent is merely thinking about the root cause of the bug without other actions, output <tags>THINK</tags>.
</examples>

Output only the tags with no other commentary. The format should be <tags>...</tags>
"""
"""行为标签提示词。

该提示词用于指导LLM为agent步骤分配行为标签。
标签分类覆盖了agent在解决bug过程中的主要行为类型。

标签分类说明:
- WRITE_TEST (☑️): 编写或修改测试脚本以复现bug
- VERIFY_TEST (✅): 运行测试验证环境
- EXAMINE_CODE (👁️): 查看、搜索或探索代码仓库
- WRITE_FIX (📝): 修改源代码修复bug
- VERIFY_FIX (🔥): 验证修复是否有效
- REPORT (📣): 向用户报告进度或完成情况
- THINK (🧠): 分析和思考bug原因
- OUTLIER (⁉️): 其他不符合上述标签的操作

设计要点:
- 支持多标签组合，一个步骤可能包含多个行为
- 提供清晰的行为定义和示例
- 标签设计便于可视化和统计分析

输出格式:
<tags>标签1,标签2,...</tags>
"""

KNOWN_TAGS = {
    "WRITE_TEST": "☑️",
    "VERIFY_TEST": "✅",
    "EXAMINE_CODE": "👁️",
    "WRITE_FIX": "📝",
    "VERIFY_FIX": "🔥",
    "REPORT": "📣",
    "THINK": "🧠",
    "OUTLIER": "⁉️",
}
"""已知标签及其对应的emoji。

该字典定义了所有支持的行为标签及其可视化表示。
使用emoji可以更直观地在界面中展示agent的行为类型。

标签说明:
- WRITE_TEST (☑️): 测试编写 - 复选框表示任务
- VERIFY_TEST (✅): 测试验证 - 勾选标记表示验证通过
- EXAMINE_CODE (👁️): 代码检查 - 眼睛表示查看
- WRITE_FIX (📝): 代码修复 - 笔记表示修改
- VERIFY_FIX (🔥): 修复验证 - 火焰表示测试运行
- REPORT (📣): 进度报告 - 扩音器表示通知
- THINK (🧠): 思考分析 - 大脑表示思维
- OUTLIER (⁉️): 其他操作 - 问号表示未知
"""

tags_re = re.compile(r"<tags>([A-Z_,\s]+)</tags>")
"""正则表达式模式，用于从LLM响应中提取标签。

匹配格式: <tags>标签内容</tags>
捕获组包含标签内容，可能包含多个标签（用逗号分隔）
"""


@dataclass
class LakeViewStep:
    """Lake View步骤数据类。

    存储分析后的agent步骤信息，包括任务描述和行为标签。
    这个结构用于在UI中展示agent的行为轨迹。

    属性:
        desc_task: 简洁的任务描述（来自<task>标签）
        desc_details: 详细的bug特定信息（来自<details>标签）
        tags_emoji: 行为标签的emoji表示（用于可视化）
    """
    desc_task: str
    desc_details: str
    tags_emoji: str


class LakeView:
    """Lake View主类，负责分析和标记agent步骤。

    该类实现了AI Agent步骤的自动分析和标签化功能。
    通过LLM分析agent的执行轨迹，提取任务描述并分配行为标签。

    主要功能:
    - 任务描述提取：简洁概括+详细说明的两级描述
    - 行为标签分配：自动识别agent的行为类型
    - 可视化支持：提供emoji标签便于界面展示

    使用示例:
        >>> config = LakeviewConfig(...)
        >>> lakeview = LakeView(config)
        >>> step = AgentStep(...)
        >>> lakeview_step = await lakeview.create_lakeview_step(step)
        >>> print(lakeview_step.desc_task)
    """
    def __init__(self, lake_view_config: LakeviewConfig | None):
        """初始化Lake View分析器。

        如果配置为None，则创建一个空实例（禁用功能）。
        否则，初始化LLM客户端和步骤存储。

        参数:
            lake_view_config: Lake View配置对象，如果为None则禁用分析功能

        注意:
            - 如果配置为None，初始化后会立即返回
            - steps列表用于存储分析历史
            - temperature设置为0.1以确保稳定的输出
        """
        if lake_view_config is None:
            return

        self.model_config = lake_view_config.model
        self.lakeview_llm_client: LLMClient = LLMClient(self.model_config)

        self.steps: list[str] = []

    def get_label(self, tags: None | list[str], emoji: bool = True) -> str:
        """获取标签的可视化表示。

        将标签列表转换为人类可读的字符串格式。
        可以选择使用emoji或纯文本形式。

        参数:
            tags: 标签列表，如果为None或空列表则返回空字符串
            emoji: 是否使用emoji表示（默认True）

        返回:
            str: 格式化的标签字符串，多个标签用" · "分隔

        示例:
            >>> get_label(["WRITE_TEST", "EXAMINE_CODE"])
            "☑️WRITE_TEST · 👁️EXAMINE_CODE"

            >>> get_label(["WRITE_TEST"], emoji=False)
            "WRITE_TEST"
        """
        if not tags:
            return ""

        return " · ".join([KNOWN_TAGS[tag] + tag if emoji else tag for tag in tags])

    async def extract_task_in_step(self, prev_step: str, this_step: str) -> tuple[str, str]:
        """从步骤中提取任务描述。

        使用LLM分析agent步骤，提取简洁的任务描述和详细的bug特定信息。
        采用两级粒度：<task>标签用于通用描述，<details>标签用于具体信息。

        参数:
            prev_step: 前一个步骤的内容（用于上下文）
            this_step: 当前步骤的内容

        返回:
            tuple[str, str]: (任务描述, 详细信息)，如果提取失败返回("", "")

        实现细节:
            - 使用EXTRACTOR_PROMPT指导LLM提取任务
            - temperature设置为0.1确保输出稳定
            - 最多重试10次以确保格式正确
            - 使用rpartition解析响应以提取两个部分

        注意:
            - 期望LLM返回格式: <task>...</task><details>...</details>
            - 如果格式错误，最多重试10次
            - 如果最终仍无法解析，返回空字符串
        """
        # 构建LLM消息序列，指导LLM提取任务描述
        llm_messages = [
            LLMMessage(
                role="user",
                content=f"The following is an excerpt of the steps trying to solve a software bug by an AI agent: <previous_step>{prev_step}</previous_step><this_step>{this_step}</this_step>",
            ),
            LLMMessage(role="assistant", content="I understand."),
            LLMMessage(role="user", content=EXTRACTOR_PROMPT),
            LLMMessage(
                role="assistant",
                content="Sure. Here is the task the agent is performing: <task>The agent",
            ),
        ]

        # 设置低温度以确保稳定的输出格式
        self.model_config.temperature = 0.1
        llm_response = self.lakeview_llm_client.chat(
            model_config=self.model_config,
            messages=llm_messages,
            reuse_history=False,
        )

        content = llm_response.content.strip()

        # 重试机制：如果输出格式不正确，最多重试10次
        retry = 0
        while retry < 10 and (
            "</task>" not in content or "<details>" not in content or "</details>" not in content
        ):
            retry += 1
            llm_response = self.lakeview_llm_client.chat(
                model_config=self.model_config,
                messages=llm_messages,
                reuse_history=False,
            )
            content = llm_response.content.strip()

        # 如果最终仍无法解析格式，返回空字符串
        if "</task>" not in content or "<details>" not in content or "</details>" not in content:
            return "", ""

        # 使用rpartition分割响应，提取任务和详细信息
        # rpartition从右侧开始分割，确保获取正确的部分
        desc_task, _, desc_details = content.rpartition("</task>")
        # 将<details>标签转换为斜体格式以便显示
        # [italic]和[/italic]是特定的显示格式标记
        desc_details = desc_details.replace("<details>", "[italic]").replace(
            "</details>", "[/italic]"
        )
        return desc_task, desc_details

    async def extract_tag_in_step(self, step: str) -> list[str]:
        """从步骤中提取行为标签。

        使用LLM分析整个轨迹，为当前步骤分配合适的行为标签。
        基于预定义的标签分类系统，识别agent的行为类型。

        参数:
            step: 当前步骤的内容字符串

        返回:
            list[str]: 标签列表，如果提取失败或内容过长则返回空列表

        实现细节:
            - 将历史步骤格式化为<step>标签包围的XML格式
            - 如果总长度超过300,000字符，跳过标签化
            - 使用TAGGER_PROMPT指导LLM分配标签
            - 最多重试10次确保标签有效
            - 使用正则表达式tags_re提取标签

        注意:
            - 限制长度是为了避免超过LLM的输入限制
            - 如果LLM返回未知标签，会重试直到返回有效标签
            - 返回的标签必须在KNOWN_TAGS中定义
        """
        # 将历史步骤格式化为XML格式，每个步骤用<step>标签包围
        steps_fmt = "\n\n".join(
            f'<step id="{ind + 1}">\n{s.strip()}\n</step>' for ind, s in enumerate(self.steps)
        )

        # 检查长度限制，如果超过300,000字符则跳过标签化
        if len(steps_fmt) > 300_000:
            # step_fmt is too long, skip tagging
            return []

        # 构建LLM消息序列，指导LLM分配行为标签
        llm_messages = [
            LLMMessage(
                role="user",
                content=f"Below is the trajectory of an AI agent solving a software bug until the current step. Each step is marked within a <step> tag.\n\n{steps_fmt}\n\n<current_step>{step}</current_step>",
            ),
            LLMMessage(role="assistant", content="I understand."),
            LLMMessage(role="user", content=TAGGER_PROMPT),
            LLMMessage(role="assistant", content="Sure. The tags are: <tags>"),
        ]
        # 设置低温度以确保稳定的输出
        self.model_config.temperature = 0.1

        # 重试机制：最多重试10次直到获得有效标签
        retry = 0
        while retry < 10:
            llm_response = self.lakeview_llm_client.chat(
                model_config=self.model_config,
                messages=llm_messages,
                reuse_history=False,
            )

            # 添加<tags>前缀以确保正则表达式能正确匹配
            content = "<tags>" + llm_response.content.lstrip()

            # 使用正则表达式提取标签
            matched_tags: list[str] = tags_re.findall(content)
            # 分割标签并去除空格
            tags: list[str] = [tag.strip() for tag in matched_tags[0].split(",")]
            # 检查所有标签是否都是已知标签
            if all(tag in KNOWN_TAGS for tag in tags):
                return tags

            retry += 1

        # 重试次数耗尽，返回空列表
        return []

    def _agent_step_str(self, agent_step: AgentStep) -> str | None:
        """将AgentStep转换为字符串表示。

        提取LLM响应的内容，如果有工具调用则一并包含。
        用于后续的任务提取和标签化处理。

        参数:
            agent_step: AgentStep对象，包含LLM响应信息

        返回:
            str | None: 格式化的步骤字符串，如果llm_response为None则返回None

        实现细节:
            - 提取LLM响应内容并去除首尾空格
            - 如果有工具调用，格式化为"[`工具名`] `参数`"形式
            - 将工具调用附加到内容后面

        示例:
            >>> step = AgentStep(llm_response=LLMResponse(content="Hello"))
            >>> _agent_step_str(step)
            "Hello"

            >>> step = AgentStep(llm_response=LLMResponse(
            ...     content="I'll help you",
            ...     tool_calls=[ToolCall(name="search", arguments={"query": "test"})]
            ... ))
            >>> _agent_step_str(step)
            "I'll help you\n\nTool calls:\n[`search`] `{'query': 'test'}"`
        """
        if agent_step.llm_response is None:
            return None

        # 提取响应内容并去除首尾空格
        content = agent_step.llm_response.content.strip()

        # 如果有工具调用，格式化并添加到内容中
        tool_calls_content = ""
        if agent_step.llm_response.tool_calls is not None:
            # 将每个工具调用格式化为"[`工具名`] `参数`"形式
            tool_calls_content = "\n".join(
                f"[`{tool_call.name}`] `{tool_call.arguments}`"
                for tool_call in agent_step.llm_response.tool_calls
            )
            tool_calls_content = tool_calls_content.strip()
            # 将工具调用信息附加到内容后面
            content = f"{content}\n\nTool calls:\n{tool_calls_content}"

        return content

    async def create_lakeview_step(self, agent_step: AgentStep) -> LakeViewStep | None:
        """创建Lake View步骤对象。

        这是Lake View的主要入口方法，完成步骤的完整分析流程：
        1. 将AgentStep转换为字符串表示
        2. 提取任务描述（两级粒度）
        3. 分配行为标签
        4. 创建LakeViewStep对象

        参数:
            agent_step: AgentStep对象，包含agent执行的一个步骤

        返回:
            LakeViewStep | None: 分析后的步骤对象，如果步骤内容为空则返回None

        实现细节:
            - 获取前一个步骤作为上下文
            - 使用_agent_step_str转换当前步骤
            - 并行调用extract_task_in_step和extract_tag_in_step
            - 使用get_label将标签转换为emoji表示

        注意:
            - 如果steps列表为空或只有一个元素，previous_step_str为"(none)"
            - 如果agent_step的llm_response为None，返回None
            - this_step_str会添加到steps列表中供后续使用

        使用示例:
            >>> lakeview = LakeView(config)
            >>> step = AgentStep(llm_response=LLMResponse(content="Fixing the bug..."))
            >>> lakeview_step = await lakeview.create_lakeview_step(step)
            >>> print(lakeview_step.desc_task)
            "The agent is fixing the bug."
            >>> print(lakeview_step.tags_emoji)
            "📝WRITE_FIX"
        """
        # 获取前一个步骤作为上下文，如果没有则使用"(none)"
        previous_step_str = "(none)"
        if len(self.steps) > 1:
            previous_step_str = self.steps[-1]

        # 将AgentStep转换为字符串表示
        this_step_str = self._agent_step_str(agent_step)

        # 如果步骤内容有效，进行分析
        if this_step_str:
            # 提取任务描述（两级粒度）
            desc_task, desc_details = await self.extract_task_in_step(
                previous_step_str, this_step_str
            )
            # 分配行为标签
            tags = await self.extract_tag_in_step(this_step_str)
            # 将标签转换为emoji表示
            tags_emoji = self.get_label(tags)
            return LakeViewStep(desc_task, desc_details, tags_emoji)

        return None
