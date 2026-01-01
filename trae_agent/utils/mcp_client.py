# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""MCP（Model Context Protocol）客户端模块。

该模块实现了MCP客户端功能，用于连接和管理MCP服务器。
MCP是一种标准协议，允许LLM应用与外部工具和服务进行交互。

主要功能:
- 连接和管理多个MCP服务器
- 发现和列出服务器提供的工具
- 调用MCP工具并返回结果
- 管理服务器连接状态和发现状态
- 支持stdio传输方式

支持的功能:
- 服务器连接状态跟踪（连接中、已连接、已断开）
- 工具发现和调用
- 异步操作支持
- 资源清理和断开连接

使用示例:
    >>> client = MCPClient()
    >>> await client.connect_and_discover(
    ...     "filesystem",
    ...     MCPServerConfig(command="npx", args=["@modelcontextprotocol/server"]),
    ...     tools_container,
    ...     model_provider
    ... )
    >>> result = await client.call_tool("read_file", {"path": "/tmp/file.txt"})
"""

from contextlib import AsyncExitStack
from enum import Enum

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..tools.mcp_tool import MCPTool
from .config import MCPServerConfig


class MCPServerStatus(Enum):
    """MCP服务器状态枚举。

    定义了MCP服务器可能的各种连接状态。
    用于跟踪和显示服务器的当前状态。

    状态说明:
        DISCONNECTED: 服务器已断开或出现错误
        CONNECTING: 服务器正在连接过程中
        CONNECTED: 服务器已连接并可以使用
    """

    DISCONNECTED = "disconnected"  # Server is disconnected or experiencing errors
    CONNECTING = "connecting"  # Server is in the process of connecting
    CONNECTED = "connected"  # Server is connected and ready to use


class MCPDiscoveryState(Enum):
    """MCP发现过程状态枚举。

    定义了MCP工具发现过程的各个阶段状态。
    用于跟踪发现工具的进度。

    状态说明:
        NOT_STARTED: 发现过程尚未开始
        IN_PROGRESS: 发现过程正在进行中
        COMPLETED: 发现过程已完成（可能有错误或成功）
    """

    NOT_STARTED = "not_started"  # Discovery has not started yet
    IN_PROGRESS = "in_progress"  # Discovery is currently in progress
    # Discovery has completed (with or without errors)
    COMPLETED = "completed"


class MCPClient:
    """MCP客户端主类。

    负责管理和与MCP服务器的交互。
    支持多个服务器的并发连接和工具调用。

    主要功能:
    - 管理多个MCP服务器的连接
    - 跟踪每个服务器的状态
    - 发现和注册MCP工具
    - 调用工具并返回结果
    - 清理资源和断开连接

    使用示例:
        >>> client = MCPClient()
        >>>
        >>> # 连接并发现工具
        >>> await client.connect_and_discover(...)
        >>>
        >>> # 调用工具
        >>> result = await client.call_tool("search", {"query": "test"})
        >>>
        >>> # 清理
        >>> await client.cleanup("filesystem")
    """

    def __init__(self):
        """初始化MCP客户端。

        创建必要的内部对象来管理连接和资源。
        """
        # Initialize session and client objects
        self.session: ClientSession | None = None
        # 用于管理异步资源的退出栈
        self.exit_stack = AsyncExitStack()
        # 跟踪各个MCP服务器的连接状态
        self.mcp_servers_status: dict[str, MCPServerStatus] = {}

    def get_mcp_server_status(self, mcp_server_name: str) -> MCPServerStatus:
        """获取指定MCP服务器的当前状态。

        参数:
            mcp_server_name: MCP服务器的名称

        返回:
            MCPServerStatus: 服务器状态，如果服务器不存在则返回DISCONNECTED

        注意:
            - 使用get方法提供默认值，避免KeyError
        """
        return self.mcp_servers_status.get(mcp_server_name, MCPServerStatus.DISCONNECTED)

    def update_mcp_server_status(self, mcp_server_name, status: MCPServerStatus):
        """更新MCP服务器的状态。

        参数:
            mcp_server_name: MCP服务器的名称
            status: 要设置的新状态

        注意:
            - 直接更新字典中的状态值
            - 状态更新会立即生效
        """
        self.mcp_servers_status[mcp_server_name] = status

    async def connect_and_discover(
        self,
        mcp_server_name: str,
        mcp_server_config: MCPServerConfig,
        mcp_tools_container: list,
        model_provider,
    ):
        """连接到MCP服务器并发现可用的工具。

        这是连接MCP服务器的主要入口方法，负责：
        1. 根据配置选择合适的传输方式
        2. 建立连接
        3. 初始化会话
        4. 列出可用的工具
        5. 创建MCPTool对象并添加到容器中

        参数:
            self: MCPClient实例
            mcp_server_name: MCP服务器的名称
            mcp_server_config: MCP服务器配置对象
            mcp_tools_container: 用于存储发现的MCP工具的列表
            model_provider: 模型提供商对象，用于创建MCPTool

        抛出:
            NotImplementedError: 当使用HTTP或WebSocket传输时（尚未实现）
            ValueError: 当配置无效时（既没有command也没有url）

        注意:
            - 当前仅支持stdio传输方式
            - HTTP和WebSocket传输尚未实现
            - 工具发现失败会抛出异常
        """
        transport = None
        # 检查传输类型并验证支持
        if mcp_server_config.http_url:
            raise NotImplementedError("HTTP transport is not implemented yet")
        elif mcp_server_config.url:
            raise NotImplementedError("WebSocket transport is not implemented yet")
        elif mcp_server_config.command:
            # 使用stdio传输方式
            params = StdioServerParameters(
                command=mcp_server_config.command,
                args=mcp_server_config.args,
                env=mcp_server_config.env,
                cwd=mcp_server_config.cwd,
            )
            # 创建stdio客户端并进入异步上下文
            transport = await self.exit_stack.enter_async_context(stdio_client(params))
        else:
            # error: 配置无效
            raise ValueError(
                f"Invalid MCP server configuration for {mcp_server_name}. "
                "Please provide either a command or a URL."
            )
        try:
            # 连接到MCP服务器
            await self.connect_to_server(mcp_server_name, transport)
            # 列出服务器提供的所有工具
            mcp_tools = await self.list_tools()
            # 为每个工具创建MCPTool对象并添加到容器
            for tool in mcp_tools.tools:
                mcp_tool = MCPTool(self, tool, model_provider)
                mcp_tools_container.append(mcp_tool)
        except Exception as e:
            # 任何错误都会向上抛出
            raise e

    async def connect_to_server(self, mcp_server_name, transport):
        """连接到MCP服务器。

        建立与服务器的连接并初始化会话。

        参数:
            mcp_server_name: MCP服务器的名称
            transport: 传输对象（stdio、write等）

        实现细节:
            - 检查服务器是否已连接
            - 如果未连接，更新状态为CONNECTING
            - 使用传输对象创建ClientSession
            - 初始化会话
            - 成功后更新状态为CONNECTED
            - 失败后更新状态为DISCONNECTED并重新抛出异常

        注意:
            - 使用AsyncExitStack管理资源清理
            - 连接失败会保持服务器状态为DISCONNECTED
        """
        if self.get_mcp_server_status(mcp_server_name) != MCPServerStatus.CONNECTED:
            self.update_mcp_server_status(mcp_server_name, MCPServerStatus.CONNECTING)
            try:
                stdio, write = transport
                # 创建ClientSession并进入异步上下文
                self.session = await self.exit_stack.enter_async_context(
                    ClientSession(stdio, write)
                )
                # 初始化MCP会话
                await self.session.initialize()
                self.update_mcp_server_status(mcp_server_name, MCPServerStatus.CONNECTED)
            except Exception as e:
                self.update_mcp_server_status(mcp_server_name, MCPServerStatus.DISCONNECTED)
                raise e

    async def call_tool(self, name, args):
        """调用MCP工具。

        通过已建立的会话调用指定的MCP工具。

        参数:
            name: 工具名称
            args: 工具参数字典

        返回:
            工具调用的输出结果

        注意:
            - 要求session已经建立
            - 直接代理调用到session.call_tool
        """
        output = await self.session.call_tool(name, args)
        return output

    async def list_tools(self):
        """列出MCP服务器提供的所有工具。

        获取服务器上可用的所有MCP工具。

        返回:
            tools: 可用工具的列表

        注意:
            - 要求session已经建立
            - 直接代理调用到session.list_tools
        """
        tools = await self.session.list_tools()
        return tools

    async def cleanup(self, mcp_server_name):
        """清理MCP服务器连接资源。

        断开与MCP服务器的连接并释放所有相关资源。

        参数:
            mcp_server_name: 要清理的MCP服务器名称

        实现细节:
            - 关闭AsyncExitStack，会自动清理所有注册的资源
            - 更新服务器状态为DISCONNECTED

        注意:
            - 应该在不再需要MCP服务器时调用此方法
            - 清理后不能再调用任何MCP方法，除非重新连接
        """
        await self.exit_stack.aclose()
        self.update_mcp_server_status(mcp_server_name, MCPServerStatus.DISCONNECTED)
