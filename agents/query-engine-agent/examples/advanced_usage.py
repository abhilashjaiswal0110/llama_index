"""
Advanced usage examples for Query Engine Agent
"""

from query_engine_agent.agent import QueryEngineAgent, QueryConfig

# Example 1: Hybrid search
def example_hybrid_search():
    """Combine vector and keyword search"""
    agent = QueryEngineAgent(index_path="./index")
    
    config = QueryConfig(
        mode="hybrid",
        top_k=5,
        similarity_threshold=0.75
    )
    
    response = agent.query(
        "Compare vector databases",
        config=config
    )
    
    print(f"Answer: {response.response}")
    print(f"Sources: {len(response.source_nodes)} chunks")


# Example 2: Sub-question decomposition
def example_sub_question():
    """Break complex queries into sub-questions"""
    agent = QueryEngineAgent(index_path="./index")
    
    response = agent.query(
        "What are the benefits and drawbacks of RAG versus fine-tuning?",
        config=QueryConfig(mode="sub-question")
    )
    
    print(f"Answer: {response.response}")


# Example 3: Streaming responses
def example_streaming():
    """Stream responses in real-time"""
    agent = QueryEngineAgent(index_path="./index")
    
    print("Streaming response...")
    for chunk in agent.query_stream("Explain transformers architecture"):
        print(chunk, end="", flush=True)
    print()


# Example 4: Custom prompt template
def example_custom_prompt():
    """Use custom prompt template"""
    agent = QueryEngineAgent(index_path="./index")
    
    # Custom template would be loaded from file in production
    response = agent.query(
        "What is machine learning?",
        config=QueryConfig(
            mode="similarity",
            top_k=3,
            temperature=0.7
        )
    )
    
    print(response.response)


if __name__ == "__main__":
    print("Query Engine Agent - Advanced Examples\n")
    
    # Run examples
    # example_hybrid_search()
    # example_sub_question()
    # example_streaming()
    # example_custom_prompt()
