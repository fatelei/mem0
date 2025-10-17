import os

from mem0 import Memory


def main():
    # Initialize mem0 Memory with PydanticAI LLM
    memory = Memory(
        llm_config={
            "provider": "pydanticai",
            "config": {
                "model": "openai:gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 1000
            }
        }
    )
    
    # Example conversation
    print("=== mem0 + PydanticAI Example ===\n")
    
    # Add memories
    user_id = "user123"
    
    print("Adding memories...")
    
    # Memory about user preferences
    result1 = memory.add(
        messages="I love hiking in the mountains and prefer trails with scenic views.",
        user_id=user_id,
        metadata={"category": "preferences"}
    )
    print(f"Memory 1 added: {result1}")
    
    # Memory about user background
    result2 = memory.add(
        messages="I work as a software engineer and have been coding for 10 years.",
        user_id=user_id,
        metadata={"category": "background"}
    )
    print(f"Memory 2 added: {result2}")
    
    # Memory about recent activity
    result3 = memory.add(
        messages="Last weekend I went camping with my family in Yosemite National Park.",
        user_id=user_id,
        metadata={"category": "activities"}
    )
    print(f"Memory 3 added: {result3}")
    
    print("\nSearching memories...")
    
    # Search memories
    search_results = memory.search(
        query="What are my outdoor hobbies?",
        user_id=user_id
    )
    print(f"Search results: {search_results}")
    
    # Get all memories for the user
    all_memories = memory.get_all(user_id=user_id)
    print(f"\nAll memories for {user_id}:")
    for i, mem in enumerate(all_memories, 1):
        print(f"{i}. {mem['memory']}")
    
    # Update a memory
    if all_memories:
        memory_id = all_memories[0]['id']
        print(f"\nUpdating memory {memory_id}...")
        memory.update(memory_id, data="I absolutely love mountain hiking on scenic trails!")
    
    # Delete a memory
    if len(all_memories) > 1:
        memory_id = all_memories[1]['id']
        print(f"Deleting memory {memory_id}...")
        memory.delete(memory_id)
    
    print("\nFinal memories:")
    final_memories = memory.get_all(user_id=user_id)
    for i, mem in enumerate(final_memories, 1):
        print(f"{i}. {mem['memory']}")


def advanced_example():
    """Advanced example using custom PydanticAI agent with structured output."""
    
    try:
        from pydantic import BaseModel, Field
        from pydantic_ai import Agent

        from mem0.llms.pydanticai import PydanticAIConfig, PydanticAILLM
    except ImportError:
        print("This example requires pydantic-ai to be installed.")
        print("Install it with: pip install pydantic-ai")
        return
    
    # Define structured output type
    class MemorySummary(BaseModel):
        """Structured summary of memories."""
        total_memories: int = Field(description="Total number of memories")
        categories: list[str] = Field(description="List of memory categories")
        key_insights: str = Field(description="Key insights about the user")
        
    # Create custom PydanticAI agent
    agent = Agent(
        "openai:gpt-4o-mini",
        output_type=MemorySummary,
        instructions="Analyze the user's memories and provide a structured summary."
    )
    
    # Configure mem0 with custom agent
    config = PydanticAIConfig(
        agent_model=agent,  # Use pre-configured agent
        agent_output_type=MemorySummary
    )
    
    llm = PydanticAILLM(config)
    
    # Initialize memory
    memory = Memory()
    
    # Add some memories
    user_id = "advanced_user"
    
    memories = [
        "I'm a vegetarian who loves Italian cuisine.",
        "I practice yoga every morning at 6 AM.",
        "I'm learning to play the guitar and can play basic chords.",
        "I prefer working remotely and enjoy flexible schedules."
    ]
    
    for mem in memories:
        memory.add(mem, user_id=user_id)
    
    # Get all memories and analyze with PydanticAI
    all_memories = memory.get_all(user_id=user_id)
    memory_text = "\n".join([mem['memory'] for mem in all_memories])
    
    # Use the PydanticAI agent to analyze memories
    result = llm.generate_response([
        {"role": "user", "content": f"Analyze these memories:\n\n{memory_text}"}
    ])
    
    print("=== Advanced PydanticAI Example ===")
    print(f"Structured analysis: {result}")


if __name__ == "__main__":
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable to run this example.")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
    else:
        main()
        print("\n" + "="*50 + "\n")
        advanced_example()