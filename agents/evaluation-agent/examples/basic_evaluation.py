"""
Basic evaluation examples for Evaluation Agent
"""

from evaluation_agent.agent import EvaluationAgent
from llama_index.core import load_index_from_storage, StorageContext

# Example 1: Simple evaluation
def example_simple_evaluation():
    """Basic pipeline evaluation"""
    agent = EvaluationAgent()
    
    # Load pipeline
    storage = StorageContext.from_defaults(persist_dir="./pipeline")
    index = load_index_from_storage(storage)
    query_engine = index.as_query_engine()
    
    # Define test queries
    test_queries = [
        {"query": "What is RAG?"},
        {"query": "How does vector search work?"},
        {"query": "Explain embeddings"}
    ]
    
    # Evaluate
    results = agent.evaluate(
        query_engine,
        test_queries,
        metrics=["faithfulness", "relevance"]
    )
    
    print("Evaluation Results:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.3f}")


# Example 2: Generate test queries
def example_generate_tests():
    """Auto-generate test queries"""
    agent = EvaluationAgent()
    
    test_queries = agent.generate_test_queries(
        data_dir="./docs",
        num_queries=20
    )
    
    print(f"Generated {len(test_queries)} test queries")
    for i, query in enumerate(test_queries[:3], 1):
        print(f"\n{i}. {query['query']}")


# Example 3: Compare pipelines
def example_compare_pipelines():
    """Compare two pipeline versions"""
    agent = EvaluationAgent()
    
    # Load pipelines
    storage_v1 = StorageContext.from_defaults(persist_dir="./pipeline_v1")
    index_v1 = load_index_from_storage(storage_v1)
    
    storage_v2 = StorageContext.from_defaults(persist_dir="./pipeline_v2")
    index_v2 = load_index_from_storage(storage_v2)
    
    test_queries = [
        {"query": "What is RAG?"},
        {"query": "How does it work?"}
    ]
    
    comparison = agent.compare_pipelines(
        index_v1.as_query_engine(),
        index_v2.as_query_engine(),
        test_queries
    )
    
    print("Comparison Results:")
    for metric, values in comparison.items():
        improvement = ((values['b'] - values['a']) / values['a']) * 100
        print(f"  {metric}: {values['a']:.3f} -> {values['b']:.3f} ({improvement:+.1f}%)")


if __name__ == "__main__":
    print("Evaluation Agent - Examples\n")
    
    # Run examples
    # example_simple_evaluation()
    # example_generate_tests()
    # example_compare_pipelines()
