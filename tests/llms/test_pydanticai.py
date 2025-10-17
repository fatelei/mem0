from unittest.mock import Mock, patch

import pytest

from mem0.llms.pydanticai import PydanticAIConfig, PydanticAILLM


class TestPydanticAIConfig:
    """Test cases for PydanticAIConfig class."""

    def test_default_config(self):
        """Test default configuration initialization."""
        config = PydanticAIConfig()
        assert config.model is None
        assert config.temperature == 0.1
        assert config.max_tokens == 2000
        assert config.agent_model is None
        assert config.agent_deps_type is None
        assert config.agent_output_type is None
        assert config.agent_instructions is None

    def test_config_with_parameters(self):
        """Test configuration with custom parameters."""
        config = PydanticAIConfig(
            model="openai:gpt-4o",
            temperature=0.5,
            max_tokens=1000,
            agent_instructions="Be helpful"
        )
        assert config.model == "openai:gpt-4o"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert config.agent_instructions == "Be helpful"


class TestPydanticAILLM:
    """Test cases for PydanticAILLM class."""

    @patch('mem0.llms.pydanticai.Agent')
    def test_init_with_default_config(self, mock_agent):
        """Test initialization with default configuration."""
        mock_agent_instance = Mock()
        mock_agent.return_value = mock_agent_instance
        
        llm = PydanticAILLM()
        
        assert llm.config.model == "openai:gpt-4o"
        mock_agent.assert_called_once_with(
            "openai:gpt-4o",
            deps_type=None,
            output_type=None,
            instructions=None,
        )
        assert llm.agent == mock_agent_instance

    @patch('mem0.llms.pydanticai.Agent')
    def test_init_with_custom_config(self, mock_agent):
        """Test initialization with custom configuration."""
        mock_agent_instance = Mock()
        mock_agent.return_value = mock_agent_instance
        
        config = PydanticAIConfig(
            model="anthropic:claude-3-5-sonnet",
            temperature=0.7,
            agent_instructions="Be concise"
        )
        llm = PydanticAILLM(config)
        
        assert llm.config.model == "anthropic:claude-3-5-sonnet"
        mock_agent.assert_called_once_with(
            "anthropic:claude-3-5-sonnet",
            deps_type=None,
            output_type=None,
            instructions="Be concise",
        )

    @patch('mem0.llms.pydanticai.Agent')
    def test_init_with_preconfigured_agent(self, mock_agent):
        """Test initialization with pre-configured agent."""
        preconfigured_agent = Mock()
        
        config = PydanticAIConfig(agent_model=preconfigured_agent)
        llm = PydanticAILLM(config)
        
        # Should not create a new agent
        mock_agent.assert_not_called()
        assert llm.agent == preconfigured_agent

    @patch('mem0.llms.pydanticai.Agent')
    def test_init_with_dict_config(self, mock_agent):
        """Test initialization with dictionary configuration."""
        mock_agent_instance = Mock()
        mock_agent.return_value = mock_agent_instance
        
        config_dict = {
            "model": "openai:gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 1500
        }
        llm = PydanticAILLM(config_dict)
        
        assert llm.config.model == "openai:gpt-4o-mini"
        assert llm.config.temperature == 0.3
        assert llm.config.max_tokens == 1500

    @patch('mem0.llms.pydanticai.Agent')
    def test_generate_response_without_tools(self, mock_agent):
        """Test generating response without tools."""
        mock_agent_instance = Mock()
        mock_result = Mock()
        mock_result.output = "Hello, world!"
        mock_agent_instance.run_sync.return_value = mock_result
        mock_agent.return_value = mock_agent_instance
        
        llm = PydanticAILLM()
        messages = [{"role": "user", "content": "Say hello"}]
        
        response = llm.generate_response(messages)
        
        assert response == "Hello, world!"
        mock_agent_instance.run_sync.assert_called_once()
        
        # Check that the prompt was properly formatted
        call_args = mock_agent_instance.run_sync.call_args
        assert "User: Say hello" in call_args[0][0]

    @patch('mem0.llms.pydanticai.Agent')
    def test_generate_response_with_tools(self, mock_agent):
        """Test generating response with tools."""
        mock_agent_instance = Mock()
        mock_result = Mock()
        mock_result.output = "I'll help you with that"
        mock_agent_instance.run_sync.return_value = mock_result
        mock_agent.return_value = mock_agent_instance
        
        llm = PydanticAILLM()
        messages = [{"role": "user", "content": "Help me"}]
        tools = [{"name": "search", "description": "Search the web"}]
        
        response = llm.generate_response(messages, tools=tools)
        
        assert response["content"] == "I'll help you with that"
        assert response["tool_calls"] == []

    @patch('mem0.llms.pydanticai.Agent')
    def test_generate_response_with_deps(self, mock_agent):
        """Test generating response with dependencies."""
        mock_agent_instance = Mock()
        mock_result = Mock()
        mock_result.output = "Response with deps"
        mock_agent_instance.run_sync.return_value = mock_result
        mock_agent.return_value = mock_agent_instance
        
        llm = PydanticAILLM()
        messages = [{"role": "user", "content": "Test"}]
        deps = Mock()
        
        response = llm.generate_response(messages, deps=deps)
        
        assert response == "Response with deps"
        mock_agent_instance.run_sync.assert_called_once_with("User: Test", deps=deps)

    @patch('mem0.llms.pydanticai.Agent')
    def test_generate_response_with_multiple_messages(self, mock_agent):
        """Test generating response with multiple messages."""
        mock_agent_instance = Mock()
        mock_result = Mock()
        mock_result.output = "Multi-message response"
        mock_agent_instance.run_sync.return_value = mock_result
        mock_agent.return_value = mock_agent_instance
        
        llm = PydanticAILLM()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        response = llm.generate_response(messages)
        
        assert response == "Multi-message response"
        call_args = mock_agent_instance.run_sync.call_args
        prompt = call_args[0][0]
        assert "System: You are helpful" in prompt
        assert "User: Hello" in prompt
        assert "Assistant: Hi there!" in prompt
        assert "User: How are you?" in prompt

    @patch('mem0.llms.pydanticai.Agent')
    def test_generate_response_error_handling(self, mock_agent):
        """Test error handling in generate_response."""
        mock_agent_instance = Mock()
        mock_agent_instance.run_sync.side_effect = Exception("API Error")
        mock_agent.return_value = mock_agent_instance
        
        llm = PydanticAILLM()
        messages = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(Exception, match="API Error"):
            llm.generate_response(messages)

    def test_messages_to_prompt_empty(self):
        """Test converting empty messages to prompt."""
        llm = PydanticAILLM()
        prompt = llm._messages_to_prompt([])
        assert prompt == ""

    def test_messages_to_prompt_single_message(self):
        """Test converting single message to prompt."""
        llm = PydanticAILLM()
        messages = [{"role": "user", "content": "Hello"}]
        prompt = llm._messages_to_prompt(messages)
        assert prompt == "User: Hello"

    def test_get_common_params(self):
        """Test getting common parameters."""
        config = PydanticAIConfig(temperature=0.5, max_tokens=1000)
        llm = PydanticAILLM(config)
        
        params = llm._get_common_params(custom_param="value")
        
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 1000
        assert params["custom_param"] == "value"


@pytest.mark.skipif(True, reason="pydantic-ai not installed in test environment")
class TestPydanticAIIntegration:
    """Integration tests that require pydantic-ai to be installed."""

    def test_real_agent_creation(self):
        """Test creating a real PydanticAI agent."""
        try:
            config = PydanticAIConfig(model="openai:gpt-4o-mini")
            llm = PydanticAILLM(config)
            
            assert llm.agent is not None
            assert hasattr(llm.agent, 'run_sync')
            
        except ImportError:
            pytest.skip("pydantic-ai not installed")