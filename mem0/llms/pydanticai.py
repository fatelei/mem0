import logging
from typing import Any, Dict, List, Optional, Union

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase

try:
    from pydantic_ai import Agent
except ImportError:
    raise ImportError("pydantic-ai is not installed. Please install it using `pip install pydantic-ai`")


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
        response_callback: Optional[Any] = None,
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


class PydanticAILLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, PydanticAIConfig, Dict]] = None):
        # Convert to PydanticAIConfig if needed
        if config is None:
            config = PydanticAIConfig()
        elif isinstance(config, dict):
            config = PydanticAIConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, PydanticAIConfig):
            # Convert BaseLlmConfig to PydanticAIConfig
            config = PydanticAIConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                http_client_proxies=config.http_client,
            )

        super().__init__(config)

        if not self.config.model and not getattr(self.config, 'agent_model', None):
            self.config.model = "openai:gpt-4o"

        # Initialize the PydanticAI agent
        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize the PydanticAI agent based on configuration."""
        if hasattr(self.config, 'agent_model') and self.config.agent_model:
            # Use pre-configured agent
            self.agent = self.config.agent_model
        else:
            # Create new agent with model string
            self.agent = Agent(
                self.config.model,
                deps_type=getattr(self.config, 'agent_deps_type', None),
                output_type=getattr(self.config, 'agent_output_type', None),
                instructions=getattr(self.config, 'agent_instructions', None),
            )

    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict]] = None, 
        tool_choice: str = "auto", 
        **kwargs
    ):
        """
        Generate a response using PydanticAI.

        Args:
            messages: List of message dicts containing 'role' and 'content'.
            tools: List of tools that the model can call. Defaults to None.
            tool_choice: Tool choice method. Defaults to "auto".
            **kwargs: Additional provider-specific parameters.

        Returns:
            str or dict: The generated response.
        """
        try:
            # Convert messages to a single prompt string for PydanticAI
            # PydanticAI works with single prompts, not message arrays
            prompt = self._messages_to_prompt(messages)
            
            # Prepare dependencies if specified
            deps = kwargs.pop('deps', None)
            
            # Run the agent
            if deps:
                result = self.agent.run_sync(prompt, deps=deps)
            else:
                result = self.agent.run_sync(prompt)
            
            # Handle response based on whether tools are used
            if tools:
                return self._parse_response_with_tools(result, tools)
            else:
                return result.output

        except Exception as e:
            logging.error(f"Error generating response with PydanticAI: {e}")
            raise

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert a list of messages to a single prompt string.
        
        Args:
            messages: List of message dicts
            
        Returns:
            str: Combined prompt string
        """
        if not messages:
            return ""
        
        # For now, just concatenate the content of all messages
        # In a more sophisticated implementation, we could format this better
        prompt_parts = []
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            if content:
                if role == 'system':
                    prompt_parts.append(f"System: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")
                else:  # user or any other role
                    prompt_parts.append(f"User: {content}")
        
        return "\n\n".join(prompt_parts)

    def _parse_response_with_tools(self, result, tools: Optional[List[Dict]]):
        """
        Parse the response when tools are provided.
        
        Args:
            result: The result from PydanticAI agent
            tools: The list of tools provided in the request
            
        Returns:
            dict: The processed response with tool calls if any
        """
        # For now, return the output as content
        # In a more sophisticated implementation, we would handle tool calls
        processed_response = {
            "content": result.output,
            "tool_calls": [],
        }
        
        # TODO: Implement tool call parsing when PydanticAI supports it
        # This would involve checking if the agent made any tool calls
        # and formatting them appropriately
        
        return processed_response

    def _get_common_params(self, **kwargs) -> Dict:
        """
        Get common parameters for PydanticAI.
        
        Returns:
            Dict: Common parameters dictionary.
        """
        params = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # Add provider-specific parameters from kwargs
        params.update(kwargs)

        return params