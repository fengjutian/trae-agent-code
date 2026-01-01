# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Trae Agent 的命令行界面 (CLI)

该模块实现了 Trae Agent 的主要命令行功能，包括：
- 运行单次任务
- 交互式会话模式
- 配置显示
- 工具列表展示
- Docker 环境支持
"""

import asyncio
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from trae_agent.agent import Agent
from trae_agent.utils.cli import CLIConsole, ConsoleFactory, ConsoleMode, ConsoleType
from trae_agent.utils.config import Config, TraeAgentConfig

# Load environment variables
_ = load_dotenv()

console = Console()


def resolve_config_file(config_file: str) -> str:
    """
    解析配置文件路径，支持向后兼容
    
    首先尝试指定的文件，如果 YAML 文件不存在，则回退到 JSON 格式的配置文件
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        str: 实际存在的配置文件路径
        
    Raises:
        SystemExit: 如果配置文件不存在且无法回退到 JSON 格式
    """
    if config_file.endswith(".yaml") or config_file.endswith(".yml"):
        yaml_path = Path(config_file)
        json_path = Path(config_file.replace(".yaml", ".json").replace(".yml", ".json"))
        if yaml_path.exists():
            return str(yaml_path)
        elif json_path.exists():
            console.print(f"[yellow]YAML config not found, using JSON config: {json_path}[/yellow]")
            return str(json_path)
        else:
            console.print(
                "[red]Error: Config file not found. Please specify a valid config file in the command line option --config-file[/red]"
            )
            sys.exit(1)
    else:
        return config_file


def check_docker(timeout=3):
    """
    检查 Docker 环境是否可用
    
    Args:
        timeout: 检查 Docker 守护进程的超时时间（秒）
        
    Returns:
        dict: 包含 Docker 状态信息的字典，包含以下字段：
            - cli: bool - Docker CLI 是否可用
            - daemon: bool - Docker 守护进程是否可访问
            - version: str - Docker 版本号（如果可用）
            - error: str - 错误信息（如果发生错误）
    """
    # 1) 检查 Docker CLI 是否已安装
    if shutil.which("docker") is None:
        return {
            "cli": False,
            "daemon": False,
            "version": None,
            "error": "docker CLI not found",
        }
    # 2) 检查 Docker 守护进程是否可访问（这会发起实际请求）
    try:
        cp = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if cp.returncode == 0 and cp.stdout.strip():
            return {
                "cli": True,
                "daemon": True,
                "version": cp.stdout.strip(),
                "error": None,
            }
        else:
            # The daemon may not be running or permissions may be insufficient
            return {
                "cli": True,
                "daemon": False,
                "version": None,
                "error": (cp.stderr or cp.stdout).strip(),
            }
    except Exception as e:
        return {"cli": True, "daemon": False, "version": None, "error": str(e)}


def build_with_pyinstaller():
    """
    使用 PyInstaller 构建工具可执行文件
    
    为 Docker 模式构建必要的工具二进制文件，包括：
    - edit_tool: 文件编辑工具
    - json_edit_tool: JSON 文件编辑工具
    
    Note: 这些工具在 Docker 容器中运行，用于执行文件操作
    """
    # 清理旧的构建目录
    os.system("rm -rf trae_agent/dist")
    
    # 构建 edit_tool
    print("--- Building edit_tool ---")
    subprocess.run(
        [
            "pyinstaller",
            "--name",
            "edit_tool",
            "trae_agent/tools/edit_tool_cli.py",
        ],
        check=True,
    )
    
    # 构建 json_edit_tool（需要额外的隐藏导入）
    print("\n--- Building json_edit_tool ---")
    subprocess.run(
        [
            "pyinstaller",
            "--name",
            "json_edit_tool",
            "--hidden-import=jsonpath_ng",
            "trae_agent/tools/json_edit_tool_cli.py",
        ],
        check=True,
    )
    
    # 创建目标目录并复制构建结果
    os.system("mkdir trae_agent/dist")
    os.system("cp dist/edit_tool/edit_tool trae_agent/dist")
    os.system("cp -r dist/json_edit_tool/_internal trae_agent/dist")
    os.system("cp dist/json_edit_tool/json_edit_tool trae_agent/dist")
    
    # 清理临时构建目录
    os.system("rm -rf dist")


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    Trae Agent - 基于 LLM 的软件工程任务代理
    
    主要功能包括：
    - run: 执行单次任务
    - interactive: 启动交互式会话
    - show-config: 显示当前配置
    - tools: 显示可用工具列表
    
    使用示例：
    - trae-cli run "创建 Python 脚本"
    - trae-cli interactive --provider deepseek
    - trae-cli show-config
    """
    pass


@cli.command()
@click.argument("task", required=False)
@click.option("--file", "-f", "file_path", help="Path to a file containing the task description.")
@click.option("--provider", "-p", help="LLM provider to use")
@click.option("--model", "-m", help="Specific model to use")
@click.option("--model-base-url", help="Base URL for the model API")
@click.option("--api-key", "-k", help="API key (or set via environment variable)")
@click.option("--max-steps", help="Maximum number of execution steps", type=int)
@click.option("--working-dir", "-w", help="Working directory for the agent")
@click.option("--must-patch", "-mp", is_flag=True, help="Whether to patch the code")
@click.option(
    "--config-file",
    help="Path to configuration file",
    default="trae_config.yaml",
    envvar="TRAE_CONFIG_FILE",
)
@click.option("--trajectory-file", "-t", help="Path to save trajectory file")
@click.option("--patch-path", "-pp", help="Path to patch file")
# --- Docker Mode Start ---
@click.option(
    "--docker-image",
    type=str,
    default=None,
    help="Specify a Docker image to run the task in a new container",
)
@click.option(
    "--docker-container-id",
    type=str,
    default=None,
    help="Attach to an existing Docker container by ID",
)
@click.option(
    "--dockerfile-path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Absolute path to a Dockerfile to build an environment",
)
@click.option(
    "--docker-image-file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Path to a local Docker image file (tar archive) to load.",
)
@click.option(
    "--docker-keep",
    type=bool,
    default=True,
    help="Keep or remove the Docker container after finishing the task",
)
# --- Docker Mode End ---


@click.option(
    "--console-type",
    "-ct",
    default="simple",
    type=click.Choice(["simple", "rich"], case_sensitive=False),
    help="Type of console to use (simple or rich)",
)
@click.option(
    "--agent-type",
    "-at",
    type=click.Choice(["trae_agent"], case_sensitive=False),
    help="Type of agent to use (trae_agent)",
    default="trae_agent",
)
def run(
    task: str | None,
    file_path: str | None,
    patch_path: str,
    provider: str | None = None,
    model: str | None = None,
    model_base_url: str | None = None,
    api_key: str | None = None,
    max_steps: int | None = None,
    working_dir: str | None = None,
    must_patch: bool = False,
    config_file: str = "trae_config.yaml",
    trajectory_file: str | None = None,
    console_type: str | None = "simple",
    agent_type: str | None = "trae_agent",
    # --- Add Docker Mode ---
    docker_image: str | None = None,
    docker_container_id: str | None = None,
    dockerfile_path: str | None = None,
    docker_image_file: str | None = None,
    docker_keep: bool = True,
):
    """
    Trae Agent 的主要运行函数，用于执行单次任务
    
    这是 trae 的核心功能，使用 Trae Agent 来执行用户指定的任务。支持多种运行模式，
    包括本地模式和 Docker 容器模式。
    
    Args:
        task: 要执行的任务描述字符串
        file_path: 包含任务描述的文件的路径（与 task 参数互斥）
        patch_path: 补丁文件路径
        provider: LLM 提供商（如 anthropic, deepseek, openai 等）
        model: 使用的具体模型名称
        model_base_url: 模型 API 的基础 URL
        api_key: API 密钥（也可以通过环境变量设置）
        max_steps: 最大执行步数限制
        working_dir: 代理的工作目录
        must_patch: 是否必须生成补丁
        config_file: 配置文件路径，默认 trae_config.yaml
        trajectory_file: 轨迹文件保存路径
        console_type: 控制台类型（simple 或 rich）
        agent_type: 代理类型（目前仅支持 trae_agent）
        docker_image: Docker 镜像名称（用于容器模式）
        docker_container_id: 现有 Docker 容器 ID（用于附加模式）
        dockerfile_path: Dockerfile 路径（用于构建模式）
        docker_image_file: 本地 Docker 镜像文件路径（用于加载模式）
        docker_keep: 任务完成后是否保留 Docker 容器
        
    Returns:
        None: 函数执行完成后退出程序
        
    Raises:
        SystemExit: 在配置错误或执行失败时退出程序
    """

    docker_config: dict[str, str | None] | None = None
    if (
        sum(
            [
                bool(docker_image),
                bool(docker_container_id),
                bool(dockerfile_path),
                bool(docker_image_file),
            ]
        )
        > 1
    ):
        console.print(
            "[red]Error: --docker-image, --docker-container-id, --dockerfile-path, and --docker-image-file are mutually exclusive.[/red]"
        )
        sys.exit(1)

    if dockerfile_path:
        docker_config = {"dockerfile_path": dockerfile_path}
        console.print(
            f"[blue]Docker mode enabled. Building from Dockerfile: {dockerfile_path}[/blue]"
        )
    elif docker_image_file:
        docker_config = {"docker_image_file": docker_image_file}
        console.print(
            f"[blue]Docker mode enabled. Loading from image file: {docker_image_file}[/blue]"
        )
    elif docker_container_id:
        docker_config = {"container_id": docker_container_id}
        console.print(
            f"[blue]Docker mode enabled. Attaching to container: {docker_container_id}[/blue]"
        )
    elif docker_image:
        docker_config = {"image": docker_image}
        console.print(f"[blue]Docker mode enabled. Using image: {docker_image}[/blue]")
    # --- ADDED END ---

    # Apply backward compatibility for config file
    config_file = resolve_config_file(config_file)

    if docker_config:
        check_msg = check_docker()
        if check_msg["cli"] and check_msg["daemon"] and check_msg["version"]:
            print("Docker is configured correctly.")
        else:
            print(f"Docker is configured incorrectly. {check_msg['error']}")
            sys.exit(1)
        if not (os.path.exists("trae_agent/dist") and os.path.exists("trae_agent/dist/_internal")):
            print("Building tools of Docker mode for the first use, waiting for a few seconds...")
            build_with_pyinstaller()
            print("Building finished.")

    if file_path:
        if task:
            console.print(
                "[red]Error: Cannot use both a task string and the --file argument.[/red]"
            )
            sys.exit(1)
        try:
            task = Path(file_path).read_text()
        except FileNotFoundError:
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            sys.exit(1)
    elif not task:
        console.print(
            "[red]Error: Must provide either a task string or use the --file argument.[/red]"
        )
        sys.exit(1)

    config = Config.create(
        config_file=config_file,
    ).resolve_config_values(
        provider=provider,
        model=model,
        model_base_url=model_base_url,
        api_key=api_key,
        max_steps=max_steps,
    )

    if not agent_type:
        console.print("[red]Error: agent_type is required.[/red]")
        sys.exit(1)

    # Create CLI Console
    console_mode = ConsoleMode.RUN
    if console_type:
        selected_console_type = (
            ConsoleType.SIMPLE if console_type.lower() == "simple" else ConsoleType.RICH
        )
    else:
        selected_console_type = ConsoleFactory.get_recommended_console_type(console_mode)

    cli_console = ConsoleFactory.create_console(
        console_type=selected_console_type, mode=console_mode
    )

    # For rich console in RUN mode, set the initial task
    if selected_console_type == ConsoleType.RICH and hasattr(cli_console, "set_initial_task"):
        cli_console.set_initial_task(task)

    # agent = Agent(agent_type, config, trajectory_file, cli_console)

    if docker_config is not None:
        docker_config["workspace_dir"] = working_dir  # now type-safe

    # Change working directory if specified
    if working_dir:
        try:
            Path(working_dir).mkdir(parents=True, exist_ok=True)
            # os.chdir(working_dir)
            console.print(f"[blue]Changed working directory to: {working_dir}[/blue]")
            working_dir = os.path.abspath(working_dir)
        except Exception as e:
            error_text = Text(f"Error changing directory: {e}", style="red")
            console.print(error_text)
            sys.exit(1)
    else:
        working_dir = os.getcwd()
        console.print(f"[blue]Using current directory as working directory: {working_dir}[/blue]")

    # Ensure working directory is an absolute path
    if not Path(working_dir).is_absolute():
        console.print(
            f"[red]Working directory must be an absolute path: {working_dir}, it should start with `/`[/red]"
        )
        sys.exit(1)

    agent = Agent(
        agent_type,
        config,
        trajectory_file,
        cli_console,
        docker_config=docker_config,
        docker_keep=docker_keep,
    )

    if not docker_config:
        try:
            os.chdir(working_dir)
        except Exception as e:
            error_text = Text(f"Error changing directory: {e}", style="red")
            console.print(error_text)
            sys.exit(1)

    try:
        task_args = {
            "project_path": working_dir,
            "issue": task,
            "must_patch": "true" if must_patch else "false",
            "patch_path": patch_path,
        }

        # Set up agent context for rich console if applicable
        if selected_console_type == ConsoleType.RICH and hasattr(cli_console, "set_agent_context"):
            cli_console.set_agent_context(agent, config.trae_agent, config_file, trajectory_file)

        # Agent will handle starting the appropriate console
        _ = asyncio.run(agent.run(task, task_args))

        console.print(f"\n[green]Trajectory saved to: {agent.trajectory_file}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Task execution interrupted by user[/yellow]")
        console.print(f"[blue]Partial trajectory saved to: {agent.trajectory_file}[/blue]")
        sys.exit(1)
    except Exception as e:
        try:
            from docker.errors import DockerException

            if isinstance(e, DockerException):
                error_text = Text(f"Docker Error: {e}", style="red")
                console.print(f"\n{error_text}")
                console.print(
                    "[yellow]Please ensure the Docker daemon is running and you have the necessary permissions.[/yellow]"
                )
            else:
                raise e
        except ImportError:
            error_text = Text(f"Unexpected error: {e}", style="red")
            console.print(f"\n{error_text}")
            console.print(traceback.format_exc())
        except Exception:
            error_text = Text(f"Unexpected error: {e}", style="red")
            console.print(f"\n{error_text}")
            console.print(traceback.format_exc())
        console.print(f"[blue]Trajectory saved to: {agent.trajectory_file}[/blue]")
        sys.exit(1)


@cli.command()
@click.option("--provider", "-p", help="LLM provider to use")
@click.option("--model", "-m", help="Specific model to use")
@click.option("--model-base-url", help="Base URL for the model API")
@click.option("--api-key", "-k", help="API key (or set via environment variable)")
@click.option(
    "--config-file",
    help="Path to configuration file",
    default="trae_config.yaml",
    envvar="TRAE_CONFIG_FILE",
)
@click.option("--max-steps", help="Maximum number of execution steps", type=int, default=20)
@click.option("--trajectory-file", "-t", help="Path to save trajectory file")
@click.option(
    "--console-type",
    "-ct",
    type=click.Choice(["simple", "rich"], case_sensitive=False),
    help="Type of console to use (simple or rich)",
)
@click.option(
    "--agent-type",
    "-at",
    type=click.Choice(["trae_agent"], case_sensitive=False),
    help="Type of agent to use (trae_agent)",
    default="trae_agent",
)
def interactive(
    provider: str | None = None,
    model: str | None = None,
    model_base_url: str | None = None,
    api_key: str | None = None,
    config_file: str = "trae_config.yaml",
    max_steps: int | None = None,
    trajectory_file: str | None = None,
    console_type: str | None = "simple",
    agent_type: str | None = "trae_agent",
):
    """
    启动与 Trae Agent 的交互式会话
    
    该函数创建一个交互式环境，用户可以连续输入多个任务，而无需每次重新启动程序。
    支持两种控制台模式：simple（简单文本）和 rich（富文本界面）。
    
    Args:
        provider: LLM 提供商（如 anthropic, deepseek, openai 等）
        model: 使用的具体模型名称
        model_base_url: 模型 API 的基础 URL
        api_key: API 密钥（也可以通过环境变量设置）
        config_file: 配置文件路径，默认 trae_config.yaml
        max_steps: 最大执行步数限制
        trajectory_file: 轨迹文件保存路径
        console_type: 控制台类型（simple 或 rich）
        agent_type: 代理类型（目前仅支持 trae_agent）
        
    Raises:
        SystemExit: 在配置错误时退出程序
    """
    # Apply backward compatibility for config file
    config_file = resolve_config_file(config_file)

    config = Config.create(
        config_file=config_file,
    ).resolve_config_values(
        provider=provider,
        model=model,
        model_base_url=model_base_url,
        api_key=api_key,
        max_steps=max_steps,
    )

    if config.trae_agent:
        trae_agent_config = config.trae_agent
    else:
        console.print("[red]Error: trae_agent configuration is required in the config file.[/red]")
        sys.exit(1)

    # Create CLI Console for interactive mode
    console_mode = ConsoleMode.INTERACTIVE
    if console_type:
        selected_console_type = (
            ConsoleType.SIMPLE if console_type.lower() == "simple" else ConsoleType.RICH
        )
    else:
        selected_console_type = ConsoleFactory.get_recommended_console_type(console_mode)

    cli_console = ConsoleFactory.create_console(
        console_type=selected_console_type,
        lakeview_config=config.lakeview,
        mode=console_mode,
    )

    if not agent_type:
        console.print("[red]Error: agent_type is required.[/red]")
        sys.exit(1)

    # Create agent
    agent = Agent(agent_type, config, trajectory_file, cli_console)

    # Get the actual trajectory file path (in case it was auto-generated)
    trajectory_file = agent.trajectory_file

    # For simple console, use traditional interactive loop
    if selected_console_type == ConsoleType.SIMPLE:
        asyncio.run(
            _run_simple_interactive_loop(
                agent, cli_console, trae_agent_config, config_file, trajectory_file
            )
        )
    else:
        # For rich console, start the textual app which handles interaction
        asyncio.run(
            _run_rich_interactive_loop(
                agent, cli_console, trae_agent_config, config_file, trajectory_file
            )
        )


async def _run_simple_interactive_loop(
    agent: Agent,
    cli_console: CLIConsole,
    trae_agent_config: TraeAgentConfig,
    config_file: str,
    trajectory_file: str | None,
):
    """
    运行简单控制台的交互式循环
    
    该函数处理简单文本控制台的交互逻辑，支持以下命令：
    - help: 显示帮助信息
    - status: 显示代理状态
    - clear: 清屏
    - exit/quit: 退出会话
    - 其他: 作为任务执行
    
    Args:
        agent: Trae Agent 实例
        cli_console: 命令行控制台实例
        trae_agent_config: Trae Agent 配置
        config_file: 配置文件路径
        trajectory_file: 轨迹文件路径
    """
    while True:
        try:
            task = cli_console.get_task_input()
            if task is None:
                console.print("[green]Goodbye![/green]")
                break

            if task.lower() == "help":
                console.print(
                    Panel(
                        """[bold]Available Commands:[/bold]

• Type any task description to execute it
• 'status' - Show agent status
• 'clear' - Clear the screen
• 'exit' or 'quit' - End the session""",
                        title="Help",
                        border_style="yellow",
                    )
                )
                continue

            working_dir = cli_console.get_working_dir_input()

            if task.lower() == "status":
                console.print(
                    Panel(
                        f"""[bold]Provider:[/bold] {agent.agent_config.model.model_provider.provider}
    [bold]Model:[/bold] {agent.agent_config.model.model}
    [bold]Available Tools:[/bold] {len(agent.agent.tools)}
    [bold]Config File:[/bold] {config_file}
    [bold]Working Directory:[/bold] {os.getcwd()}""",
                        title="Agent Status",
                        border_style="blue",
                    )
                )
                continue

            if task.lower() == "clear":
                console.clear()
                continue

            # Set up trajectory recording for this task
            console.print(f"[blue]Trajectory will be saved to: {trajectory_file}[/blue]")

            task_args = {
                "project_path": working_dir,
                "issue": task,
                "must_patch": "false",
            }

            # Execute the task
            console.print(f"\n[blue]Executing task: {task}[/blue]")

            # Start console and execute task
            console_task = asyncio.create_task(cli_console.start())
            execution_task = asyncio.create_task(agent.run(task, task_args))

            # Wait for execution to complete
            _ = await execution_task
            _ = await console_task

            console.print(f"\n[green]Trajectory saved to: {trajectory_file}[/green]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' or 'quit' to end the session[/yellow]")
        except EOFError:
            console.print("\n[green]Goodbye![/green]")
            break
        except Exception as e:
            error_text = Text(f"Error: {e}", style="red")
            console.print(error_text)


async def _run_rich_interactive_loop(
    agent: Agent,
    cli_console: CLIConsole,
    trae_agent_config: TraeAgentConfig,
    config_file: str,
    trajectory_file: str | None,
):
    """
    运行富文本控制台的交互式循环
    
    该函数启动富文本控制台的 UI 界面，由控制台自身处理交互逻辑。
    与简单控制台不同，富文本控制台提供更丰富的可视化体验。
    
    Args:
        agent: Trae Agent 实例
        cli_console: 命令行控制台实例
        trae_agent_config: Trae Agent 配置
        config_file: 配置文件路径
        trajectory_file: 轨迹文件路径
    """
    # Set up the agent in the rich console so it can handle task execution
    if hasattr(cli_console, "set_agent_context"):
        cli_console.set_agent_context(agent, trae_agent_config, config_file, trajectory_file)

    # Start the console UI - this will handle the entire interaction
    await cli_console.start()


@cli.command()
@click.option(
    "--config-file",
    help="Path to configuration file",
    default="trae_config.yaml",
    envvar="TRAE_CONFIG_FILE",
)
@click.option("--provider", "-p", help="LLM provider to use")
@click.option("--model", "-m", help="Specific model to use")
@click.option("--model-base-url", help="Base URL for the model API")
@click.option("--api-key", "-k", help="API key (or set via environment variable)")
@click.option("--max-steps", help="Maximum number of execution steps", type=int)
def show_config(
    config_file: str,
    provider: str | None,
    model: str | None,
    model_base_url: str | None,
    api_key: str | None,
    max_steps: int | None,
):
    """
    显示当前配置设置
    
    以表格形式展示当前加载的配置信息，包括：
    - 通用设置（默认提供商、最大步数等）
    - 提供商特定设置（模型、API 密钥、参数等）
    
    Args:
        config_file: 配置文件路径
        provider: 可选的 LLM 提供商覆盖
        model: 可选的模型覆盖
        model_base_url: 可选的模型基础 URL 覆盖
        api_key: 可选的 API 密钥覆盖
        max_steps: 可选的步数限制覆盖
        
    Raises:
        SystemExit: 在配置错误时退出程序
    """
    # Apply backward compatibility for config file
    config_file = resolve_config_file(config_file)

    config_path = Path(config_file)
    if not config_path.exists():
        console.print(
            Panel(
                f"""[yellow]No configuration file found at: {config_file}[/yellow]

Using default settings and environment variables.""",
                title="Configuration Status",
                border_style="yellow",
            )
        )

    config = Config.create(
        config_file=config_file,
    ).resolve_config_values(
        provider=provider,
        model=model,
        model_base_url=model_base_url,
        api_key=api_key,
        max_steps=max_steps,
    )

    if config.trae_agent:
        trae_agent_config = config.trae_agent
    else:
        console.print("[red]Error: trae_agent configuration is required in the config file.[/red]")
        sys.exit(1)

    # Display general settings
    general_table = Table(title="General Settings")
    general_table.add_column("Setting", style="cyan")
    general_table.add_column("Value", style="green")

    general_table.add_row(
        "Default Provider",
        str(trae_agent_config.model.model_provider.provider or "Not set"),
    )
    general_table.add_row("Max Steps", str(trae_agent_config.max_steps or "Not set"))

    console.print(general_table)

    # Display provider settings
    provider_config = trae_agent_config.model.model_provider
    provider_table = Table(title=f"{provider_config.provider.title()} Configuration")
    provider_table.add_column("Setting", style="cyan")
    provider_table.add_column("Value", style="green")

    provider_table.add_row("Model", trae_agent_config.model.model or "Not set")
    provider_table.add_row("Base URL", provider_config.base_url or "Not set")
    provider_table.add_row("API Version", provider_config.api_version or "Not set")
    provider_table.add_row(
        "API Key",
        (
            f"Set ({provider_config.api_key[:4]}...{provider_config.api_key[-4:]})"
            if provider_config.api_key
            else "Not set"
        ),
    )
    provider_table.add_row("Max Tokens", str(trae_agent_config.model.max_tokens))
    provider_table.add_row("Temperature", str(trae_agent_config.model.temperature))
    provider_table.add_row("Top P", str(trae_agent_config.model.top_p))

    if trae_agent_config.model.model_provider.provider == "anthropic":
        provider_table.add_row("Top K", str(trae_agent_config.model.top_k))

    console.print(provider_table)


@cli.command()
def tools():
    """
    显示可用工具及其描述
    
    以表格形式列出所有已注册的工具，包括工具名称和功能描述。
    如果工具加载失败，会显示错误信息。
    """
    from .tools import tools_registry

    tools_table = Table(title="Available Tools")
    tools_table.add_column("Tool Name", style="cyan")
    tools_table.add_column("Description", style="green")

    for tool_name in tools_registry:
        try:
            tool = tools_registry[tool_name]()
            tools_table.add_row(tool.name, tool.description)
        except Exception as e:
            tools_table.add_row(tool_name, f"[red]Error loading: {e}[/red]")

    console.print(tools_table)


def main():
    """
    CLI 的主要入口点
    
    这是命令行接口的启动函数，当直接运行模块时调用。
    """
    cli()


if __name__ == "__main__":
    main()
