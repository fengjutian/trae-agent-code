# Trae Agent

[![arXiv:2507.23370](https://img.shields.io/badge/TechReport-arXiv%3A2507.23370-b31a1b)](https://arxiv.org/abs/2507.23370)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pre-commit](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/pre-commit.yml)
[![Unit Tests](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml/badge.svg)](https://github.com/bytedance/trae-agent/actions/workflows/unit-test.yml)
[![Discord](https://img.shields.io/discord/1320998163615846420?label=Join%20Discord&color=7289DA)](https://discord.gg/VwaQ4ZBHvC)

**Trae Agent** 是一个基于 LLM 的通用软件工程任务智能体。它提供了一个强大的 CLI 接口，能够理解自然语言指令，并使用各种工具和 LLM 提供商执行复杂的软件工程工作流。

技术细节请参考[我们的技术报告](https://arxiv.org/abs/2507.23370)。

**项目状态：** 项目仍在积极开发中。如果您愿意帮助我们改进 Trae Agent，请参考 [docs/roadmap.md](docs/roadmap.md) 和 [CONTRIBUTING](CONTRIBUTING.md)。

**与其他 CLI 智能体的区别：** Trae Agent 提供了一个透明、模块化的架构，研究人员和开发人员可以轻松修改、扩展和分析，使其成为**研究 AI 智能体架构、进行消融研究和开发新型智能体能力**的理想平台。这种**研究友好型设计**使学术和开源社区能够在基础智能体框架上进行贡献和构建，促进 AI 智能体快速发展的创新。

## ✨ 特性

- 🌊 **Lakeview**: 为智能体步骤提供简短而简洁的摘要
- 🤖 **多 LLM 支持**: 支持 OpenAI、Anthropic、DeepSeek、Doubao、Azure、OpenRouter、Ollama 和 Google Gemini API
- 🛠️ **丰富的工具生态系统**: 文件编辑、bash 执行、顺序思考等
- 🎯 **交互模式**: 用于迭代开发的对话式界面
- 📊 **轨迹记录**: 详细记录所有智能体操作，用于调试和分析
- ⚙️ **灵活的配置**: 基于 YAML 的配置，支持环境变量
- 🚀 **简单安装**: 基于 pip 的简单安装

## 🚀 安装

### 要求
- UV (https://docs.astral.sh/uv/)
- 所选提供商的 API 密钥 (OpenAI、Anthropic、DeepSeek、Google Gemini、OpenRouter 等)

### 设置

```bash
git clone https://github.com/bytedance/trae-agent.git
cd trae-agent
uv sync --all-extras
source .venv/bin/activate
```

## ⚙️ 配置

### YAML 配置（推荐）

1. 复制示例配置文件：
   ```bash
   cp trae_config.yaml.example trae_config.yaml
   ```

2. 使用您的 API 凭据和偏好编辑 `trae_config.yaml`：

```yaml
agents:
  trae_agent:
    enable_lakeview: true
    model: trae_agent_model  # Trae Agent 的模型配置名称
    max_steps: 200  # 最大智能体步数
    tools:  # Trae Agent 使用的工具
      - bash
      - str_replace_based_edit_tool
      - sequentialthinking
      - task_done

model_providers:  # 模型提供商配置
  anthropic:
    api_key: your_anthropic_api_key
    provider: anthropic
  openai:
    api_key: your_openai_api_key
    provider: openai
  deepseek:
    api_key: your_deepseek_api_key
    provider: deepseek

models:
  trae_agent_model:
    model_provider: deepseek
    model: deepseek-chat
    max_tokens: 4096
    temperature: 0.5
```

**注意：** `trae_config.yaml` 文件被 git 忽略以保护您的 API 密钥。

### 使用 Base URL
在某些情况下，我们需要为 API 使用自定义 URL。只需在 `provider` 后添加 `base_url` 字段，以下配置为例：

```
openai:
    api_key: your_openrouter_api_key
    provider: openai
    base_url: https://openrouter.ai/api/v1
```
**注意：** 对于字段格式，仅使用空格。不允许使用制表符 (\t)。

### 环境变量（替代方案）

您也可以使用环境变量配置 API 密钥，并将它们存储在 .env 文件中：

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="your-openai-base-url"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export ANTHROPIC_BASE_URL="your-anthropic-base-url"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_BASE_URL="your-google-base-url"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export DOUBAO_API_KEY="your-doubao-api-key"
export DOUBAO_BASE_URL="https://ark.cn-beijing.volces.com/api/v3/"
```

### MCP 服务（可选）

要启用模型上下文协议（MCP）服务，请在配置中添加 `mcp_servers` 部分：

```yaml
mcp_servers:
  playwright:
    command: npx
    args:
      - "@playwright/mcp@0.0.27"
```

**配置优先级：** 命令行参数 > 配置文件 > 环境变量 > 默认值

**旧版 JSON 配置：** 如果使用较旧的 JSON 格式，请参阅 [docs/legacy_config.md](docs/legacy_config.md)。我们建议迁移到 YAML。

## 📖 使用

### 基本命令

```bash
# 简单任务执行
trae-cli run "Create a hello world Python script"

# 检查配置
trae-cli show-config

# 交互模式
trae-cli interactive
```

### 特定提供商示例

```bash
# OpenAI
trae-cli run "Fix the bug in main.py" --provider openai --model gpt-4o

# Anthropic
trae-cli run "Add unit tests" --provider anthropic --model claude-sonnet-4-20250514

# DeepSeek
trae-cli run "Optimize this algorithm" --provider deepseek --model deepseek-chat

# Google Gemini
trae-cli run "Optimize this algorithm" --provider google --model gemini-2.5-flash

# OpenRouter（访问多个提供商）
trae-cli run "Review this code" --provider openrouter --model "anthropic/claude-3-5-sonnet"
trae-cli run "Generate documentation" --provider openrouter --model "openai/gpt-4o"
trae-cli run "Analyze code quality" --provider openrouter --model "deepseek/deepseek-chat"

# Doubao
trae-cli run "Refactor the database module" --provider doubao --model doubao-seed-1.6

# Ollama（本地模型）
trae-cli run "Comment this code" --provider ollama --model qwen3
```

### 高级选项

```bash
# 自定义工作目录
trae-cli run "Add tests for utils module" --working-dir /path/to/project

# 保存执行轨迹
trae-cli run "Debug authentication" --trajectory-file debug_session.json

# 强制生成补丁
trae-cli run "Update API endpoints" --must-patch

# 自定义设置的交互模式
trae-cli interactive --provider openai --model gpt-4o --max-steps 30
```

## Docker 模式命令
### 准备
**重要：** 您需要确保 Docker 已在环境中配置好。

### 使用
```bash
# 指定 Docker 镜像在新容器中运行任务
trae-cli run "Add tests for utils module" --docker-image python:3.11

# 指定 Docker 镜像并在新容器中挂载目录
trae-cli run "write a script to print helloworld" --docker-image python:3.12 --working-dir test_workdir/

# 通过 ID 连接到现有的 Docker 容器（`--working-dir` 与 `--docker-container-id` 一起使用时无效）
trae-cli run "Update API endpoints" --docker-container-id 91998a56056c

# 指定 Dockerfile 的绝对路径来构建环境
trae-cli run "Debug authentication" --dockerfile-path test_workspace/Dockerfile

# 指定本地 Docker 镜像文件（tar 存档）以加载
trae-cli run "Fix the bug in main.py" --docker-image-file test_workspace/trae_agent_custom.tar

# 完成任务后删除 Docker 容器（默认保持）
trae-cli run "Add tests for utils module" --docker-image python:3.11 --docker-keep false
```

### 交互模式命令

在交互模式中，您可以使用：
- 输入任何任务描述来执行它
- `status` - 显示智能体信息
- `help` - 显示可用命令
- `clear` - 清屏
- `exit` 或 `quit` - 结束会话

## 🛠️ Advanced Features

### Available Tools

Trae Agent provides a comprehensive toolkit for software engineering tasks including file editing, bash execution, structured thinking, and task completion. For detailed information about all available tools and their capabilities, see [docs/tools.md](docs/tools.md).

### Trajectory Recording

Trae Agent automatically records detailed execution trajectories for debugging and analysis:

```bash
# Auto-generated trajectory file
trae-cli run "Debug the authentication module"
# Saves to: trajectories/trajectory_YYYYMMDD_HHMMSS.json

# Custom trajectory file
trae-cli run "Optimize database queries" --trajectory-file optimization_debug.json
```

Trajectory files contain LLM interactions, agent steps, tool usage, and execution metadata. For more details, see [docs/TRAJECTORY_RECORDING.md](docs/TRAJECTORY_RECORDING.md).

## 🔧 Development

### Contributing

For contribution guidelines, please refer to [CONTRIBUTING.md](CONTRIBUTING.md).

### Troubleshooting

**Import Errors:**
```bash
PYTHONPATH=. trae-cli run "your task"
```

**API Key Issues:**
```bash
# Verify API keys
echo $OPENAI_API_KEY
trae-cli show-config
```

**Command Not Found:**
```bash
uv run trae-cli run "your task"
```

**Permission Errors:**
```bash
chmod +x /path/to/your/project
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✍️ Citation

```bibtex
@article{traeresearchteam2025traeagent,
      title={Trae Agent: An LLM-based Agent for Software Engineering with Test-time Scaling},
      author={Trae Research Team and Pengfei Gao and Zhao Tian and Xiangxin Meng and Xinchen Wang and Ruida Hu and Yuanan Xiao and Yizhou Liu and Zhao Zhang and Junjie Chen and Cuiyun Gao and Yun Lin and Yingfei Xiong and Chao Peng and Xia Liu},
      year={2025},
      eprint={2507.23370},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2507.23370},
}
```

## 🙏 Acknowledgments

We thank Anthropic for building the [anthropic-quickstart](https://github.com/anthropics/anthropic-quickstarts) project that served as a valuable reference for the tool ecosystem.
