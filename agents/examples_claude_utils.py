"""
Example: Using CLAUDE Agent Utilities with LlamaIndex
Demonstrates best practices for production CLAUDE-based agents
"""
import asyncio
from claude_agent_utils import (
    ConversationMemory,
    AgentOrchestrator,
    PromptTemplate,
    StreamingResponseManager
)


# Example 1: Conversation Memory
def example_conversation_memory():
    """Demonstrate conversation memory management"""
    print("="*60)
    print("Example 1: Conversation Memory")
    print("="*60)
    
    # Initialize memory with 4K token limit
    memory = ConversationMemory(max_tokens=4000)
    
    # Simulate a conversation
    memory.add_message("user", "What is retrieval-augmented generation?")
    memory.add_message(
        "assistant",
        "Retrieval-Augmented Generation (RAG) is a technique that combines "
        "information retrieval with text generation. It retrieves relevant "
        "documents and uses them as context for generating responses."
    )
    
    memory.add_message("user", "How does it work with LlamaIndex?")
    memory.add_message(
        "assistant",
        "LlamaIndex provides tools to index your documents, retrieve relevant "
        "chunks based on queries, and use those chunks as context for LLM responses."
    )
    
    # Get context for next API call
    context = memory.get_context()
    print(f"\nContext messages: {len(context)}")
    print(f"Last message: {context[-1]['content'][:100]}...")
    
    # Get statistics
    stats = memory.get_stats()
    print(f"\nMemory Stats:")
    print(f"  Total messages: {stats['total_messages']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Utilization: {stats['utilization']:.1%}")
    
    print("\n✓ Conversation memory demonstrated")


# Example 2: Agent Orchestration
async def example_agent_orchestration():
    """Demonstrate multi-agent orchestration"""
    print("\n" + "="*60)
    print("Example 2: Agent Orchestration")
    print("="*60)
    
    orchestrator = AgentOrchestrator()
    
    # Define mock agent functions (in production, use real agents)
    def ingest_data(path):
        print(f"  → Ingesting data from: {path}")
        return {"documents": 100, "path": path}
    
    def create_index(documents):
        print(f"  → Creating index for {documents} documents")
        return {"index_id": "idx_12345", "doc_count": documents}
    
    def run_query(index_id, query):
        print(f"  → Querying index {index_id}: {query}")
        return {"answer": "RAG combines retrieval with generation", "sources": 3}
    
    # Define pipeline
    steps = [
        ("ingest", ingest_data, {"path": "./docs"}),
        ("index", create_index, {"documents": 100}),
        ("query", run_query, {"index_id": "idx_12345", "query": "What is RAG?"})
    ]
    
    # Run sequential pipeline
    print("\nRunning sequential pipeline:")
    results = await orchestrator.run_pipeline(steps, mode="sequential")
    
    print(f"\nPipeline Results:")
    for step_name, result in results.items():
        print(f"  {step_name}: {result}")
    
    # Check execution history
    history = orchestrator.get_execution_history()
    print(f"\nExecution History: {len(history)} steps completed")
    
    print("\n✓ Agent orchestration demonstrated")


# Example 3: Streaming Responses
async def example_streaming():
    """Demonstrate streaming response management"""
    print("\n" + "="*60)
    print("Example 3: Streaming Responses")
    print("="*60)
    
    manager = StreamingResponseManager()
    
    # Simulate streaming response
    async def mock_llm_stream():
        words = ["The", "answer", "is", "that", "RAG", "combines", "retrieval", "with", "generation"]
        for word in words:
            await asyncio.sleep(0.05)  # Simulate network delay
            yield word + " "
    
    print("\nStreaming response:")
    print("  ", end="")
    async for chunk in manager.stream_with_backpressure(mock_llm_stream()):
        print(chunk, end="", flush=True)
    print()
    
    print("\n✓ Streaming demonstrated")


# Example 4: Prompt Engineering
def example_prompt_engineering():
    """Demonstrate advanced prompt engineering"""
    print("\n" + "="*60)
    print("Example 4: Prompt Engineering")
    print("="*60)
    
    # Create template with few-shot examples
    template = PromptTemplate(
        "Extract key information from: {text}",
        examples=[
            {
                "input": "John works at Microsoft in Seattle",
                "output": "Person: John, Company: Microsoft, Location: Seattle"
            },
            {
                "input": "Tesla announced new Model Y in California",
                "output": "Company: Tesla, Product: Model Y, Location: California"
            }
        ]
    )
    
    # Format prompt
    prompt = template.format(text="Apple released iPhone 15 in Cupertino")
    print("\nGenerated Prompt:")
    print(prompt[:200] + "...")
    
    # Add chain-of-thought
    cot_prompt = template.add_chain_of_thought(prompt)
    print("\nWith Chain-of-Thought:")
    print("..." + cot_prompt[-150:])
    
    print("\n✓ Prompt engineering demonstrated")


# Example 5: Complete RAG Workflow
async def example_complete_rag_workflow():
    """Demonstrate complete RAG workflow with CLAUDE utilities"""
    print("\n" + "="*60)
    print("Example 5: Complete RAG Workflow")
    print("="*60)
    
    # Initialize components
    memory = ConversationMemory(max_tokens=4000)
    orchestrator = AgentOrchestrator()
    
    # User asks first question
    user_query = "What is retrieval-augmented generation?"
    memory.add_message("user", user_query)
    print(f"\nUser: {user_query}")
    
    # Process with RAG pipeline
    def process_rag(query):
        # Simulate RAG processing
        return {
            "answer": "RAG combines information retrieval with text generation for more accurate responses.",
            "sources": ["doc1.pdf", "doc2.pdf"]
        }
    
    steps = [
        ("rag", process_rag, {"query": user_query})
    ]
    
    results = await orchestrator.run_pipeline(steps)
    answer = results["rag"]["answer"]
    
    memory.add_message("assistant", answer)
    print(f"Assistant: {answer}")
    
    # Follow-up question (with memory)
    follow_up = "Can you explain more about the retrieval part?"
    memory.add_message("user", follow_up)
    print(f"\nUser: {follow_up}")
    
    # Get full context
    context = memory.get_context()
    print(f"\nContext for follow-up: {len(context)} messages")
    
    # Process follow-up (would use context in real implementation)
    follow_up_answer = "The retrieval part searches a knowledge base for relevant documents before generating a response."
    memory.add_message("assistant", follow_up_answer)
    print(f"Assistant: {follow_up_answer}")
    
    # Show conversation stats
    stats = memory.get_stats()
    print(f"\nConversation Stats:")
    print(f"  Messages: {stats['total_messages']}")
    print(f"  Tokens: {stats['total_tokens']}")
    print(f"  Utilization: {stats['utilization']:.1%}")
    
    print("\n✓ Complete RAG workflow demonstrated")


# Main execution
async def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("CLAUDE Agent Utilities - Examples")
    print("="*60)
    
    # Run examples
    example_conversation_memory()
    await example_agent_orchestration()
    await example_streaming()
    example_prompt_engineering()
    await example_complete_rag_workflow()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
    
    print("\n📚 Key Takeaways:")
    print("  1. Use ConversationMemory for multi-turn conversations")
    print("  2. Use AgentOrchestrator for complex workflows")
    print("  3. Use StreamingResponseManager for better UX")
    print("  4. Use PromptTemplate for consistent prompt engineering")
    print("  5. Combine all utilities for production-grade agents")
    
    print("\n💡 Next Steps:")
    print("  - Integrate these utilities into your agents")
    print("  - Customize prompt templates for your use case")
    print("  - Add error recovery to orchestration pipelines")
    print("  - Monitor token usage with conversation memory")
    print("  - Test streaming with real LLM APIs")


if __name__ == "__main__":
    asyncio.run(main())
