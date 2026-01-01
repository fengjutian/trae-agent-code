# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""DeepSeek provider configuration."""

import openai

from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.openai_compatible_base import (
    OpenAICompatibleClient,
    ProviderConfig,
)


class DeepSeekProvider(ProviderConfig):
    """DeepSeek provider configuration."""

    def create_client(
        self, api_key: str, base_url: str | None, api_version: str | None
    ) -> openai.OpenAI:
        """Create OpenAI client with DeepSeek base URL."""
        if base_url is None or base_url == "":
            base_url = "https://api.deepseek.com/v1"
        return openai.OpenAI(api_key=api_key, base_url=base_url)

    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        return "DeepSeek"

    def get_provider_name(self) -> str:
        """Get the provider name for trajectory recording."""
        return "deepseek"

    def get_extra_headers(self) -> dict[str, str]:
        """Get DeepSeek-specific headers."""
        return {}

    def supports_tool_calling(self, model_name: str) -> bool:
        """Check if the model supports tool calling."""
        # DeepSeek models that support tool calling
        tool_capable_patterns = [
            "deepseek-chat",
            "deepseek-coder",
        ]
        return any(pattern in model_name.lower() for pattern in tool_capable_patterns)


class DeepSeekClient(OpenAICompatibleClient):
    """DeepSeek client wrapper that maintains OpenAI compatibility."""

    def __init__(self, model_config: ModelConfig):
        if (
            model_config.model_provider.base_url is None
            or model_config.model_provider.base_url == ""
        ):
            model_config.model_provider.base_url = "https://api.deepseek.com/v1"
        super().__init__(model_config, DeepSeekProvider())