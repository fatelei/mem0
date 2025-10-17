from typing import Any, Callable, Optional

from mem0.configs.llms.base import BaseLlmConfig


class PydanticAIConfig(BaseLlmConfig):
    """
    Configuration class for PydanticAI-specific parameters.
    Inherits from BaseLlmConfig and adds PydanticAI-specific settings.
    """

    def __init__(
        self,
        # Base parameters
        model: Optional[str] = None,
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        max_tokens: int = 2000,
        top_p: float = 0.1,
        top_k: int = 1,
        enable_vision: bool = False,
        vision_details: Optional[str] = "auto",
        http_client_proxies: Optional[dict] = None,
        # PydanticAI-specific parameters
        agent_model: Optional[Any] = None,
        agent_deps_type: Optional[type] = None,
        agent_output_type: Optional[type] = None,
        agent_instructions: Optional[str] = None,
        response_callback: Optional[Callable[[Any, dict, dict], None]] = None,
    ):
        """
        Initialize PydanticAI configuration.

        Args:
            model: PydanticAI model string (e.g., 'openai:gpt-4o', 'anthropic:claude-3-5-sonnet'), defaults to None
            temperature: Controls randomness, defaults to 0.1
            api_key: API key for the underlying model provider, defaults to None
            max_tokens: Maximum tokens to generate, defaults to 2000
            top_p: Nucleus sampling parameter, defaults to 0.1
            top_k: Top-k sampling parameter, defaults to 1
            enable_vision: Enable vision capabilities, defaults to False
            vision_details: Vision detail level, defaults to "auto"
            http_client_proxies: HTTP client proxy settings, defaults to None
            agent_model: Pre-configured PydanticAI Agent instance, defaults to None
            agent_deps_type: Dependencies type for the agent, defaults to None
            agent_output_type: Output type for the agent, defaults to None
            agent_instructions: Instructions for the agent, defaults to None
            response_callback: Optional callback for monitoring LLM responses.
        """
        # Initialize base parameters
        super().__init__(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            enable_vision=enable_vision,
            vision_details=vision_details,
            http_client_proxies=http_client_proxies,
        )

        # PydanticAI-specific parameters
        self.agent_model = agent_model
        self.agent_deps_type = agent_deps_type
        self.agent_output_type = agent_output_type
        self.agent_instructions = agent_instructions

        # Response monitoring
        self.response_callback = response_callback