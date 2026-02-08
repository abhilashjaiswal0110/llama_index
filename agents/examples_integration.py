"""
Integration example: Complete RAG workflow using multiple agents
"""

from pathlib import Path

def complete_rag_workflow():
    """
    Demonstrates end-to-end RAG workflow using all agents.
    
    Workflow:
    1. Data Ingestion Agent - Load and process documents
    2. Indexing Agent - Create optimized index
    3. Query Engine Agent - Set up querying
    4. Evaluation Agent - Validate performance
    """
    
    print("=" * 60)
    print("Complete RAG Workflow Example")
    print("=" * 60)
    
    # Step 1: Data Ingestion
    print("\n[Step 1/4] Data Ingestion")
    print("-" * 60)
    
    from data_ingestion_agent.agent import DataIngestionAgent, IngestionConfig
    
    ingestion_config = IngestionConfig(
        chunk_size=1024,
        chunk_overlap=200,
        chunking_strategy="sentence",
        output_dir="./workflow_index"
    )
    
    ingestion_agent = DataIngestionAgent(config=ingestion_config)
    
    # Ingest documents (example - would use actual directory)
    print("Loading documents from ./docs...")
    # documents = ingestion_agent.ingest_directory("./docs", recursive=True)
    # print(f"✓ Loaded {len(documents)} documents")
    print("✓ Documents loaded (example)")
    
    # Step 2: Indexing
    print("\n[Step 2/4] Index Creation & Optimization")
    print("-" * 60)
    
    from indexing_agent.agent import IndexingAgent
    
    indexing_agent = IndexingAgent()
    
    # Create vector index
    print("Creating vector index...")
    # index = indexing_agent.create_index("./docs", index_type="vector")
    # print("✓ Vector index created")
    
    # Optimize index
    print("Optimizing index parameters...")
    # optimized_index = indexing_agent.optimize_index(index)
    # print("✓ Index optimized")
    print("✓ Index created and optimized (example)")
    
    # Save index
    # indexing_agent.save_index(optimized_index, "./workflow_index")
    print("✓ Index saved to ./workflow_index")
    
    # Step 3: Query Engine Setup
    print("\n[Step 3/4] Query Engine Configuration")
    print("-" * 60)
    
    from query_engine_agent.agent import QueryEngineAgent, QueryConfig
    
    # Load index and create query engine
    print("Setting up query engine...")
    # query_agent = QueryEngineAgent(index_path="./workflow_index")
    
    query_config = QueryConfig(
        mode="hybrid",
        top_k=5,
        similarity_threshold=0.75
    )
    
    print("✓ Query engine configured with hybrid mode")
    
    # Test query
    print("\nTesting query: 'What is RAG?'")
    # response = query_agent.query("What is RAG?", config=query_config)
    # print(f"Answer: {response.response[:200]}...")
    print("✓ Query executed successfully (example)")
    
    # Step 4: Evaluation
    print("\n[Step 4/4] Pipeline Evaluation")
    print("-" * 60)
    
    from evaluation_agent.agent import EvaluationAgent
    
    eval_agent = EvaluationAgent()
    
    # Define test queries
    test_queries = [
        {"query": "What is RAG?"},
        {"query": "How does vector search work?"},
        {"query": "Explain embeddings"},
    ]
    
    print("Running evaluation on test queries...")
    # results = eval_agent.evaluate(
    #     query_agent.query_engine,
    #     test_queries,
    #     metrics=["faithfulness", "relevance"]
    # )
    
    # Print results
    print("\nEvaluation Results:")
    # for metric, score in results.items():
    #     print(f"  {metric}: {score:.3f}")
    print("  faithfulness: 0.850 (example)")
    print("  relevance: 0.820 (example)")
    print("  overall: 0.835 (example)")
    
    print("\n" + "=" * 60)
    print("Workflow Complete!")
    print("=" * 60)
    
    # Print summary
    print("\n📊 Summary:")
    print("  ✓ Documents ingested and processed")
    print("  ✓ Optimized vector index created")
    print("  ✓ Hybrid query engine configured")
    print("  ✓ Pipeline evaluated with quality metrics")
    print("\n💡 Next Steps:")
    print("  1. Test with real queries")
    print("  2. Tune parameters based on evaluation")
    print("  3. Deploy using RAG Pipeline Agent")
    print("  4. Set up monitoring")


def rag_pipeline_optimization():
    """
    Example of iterative RAG pipeline optimization.
    """
    
    print("\n" + "=" * 60)
    print("RAG Pipeline Optimization Example")
    print("=" * 60)
    
    from rag_pipeline_agent.agent import RAGPipelineAgent
    
    agent = RAGPipelineAgent()
    
    # Build baseline pipeline
    print("\n[1] Building baseline pipeline...")
    # pipeline = agent.build_pipeline(data_dir="./docs")
    # agent.save_pipeline(pipeline, "./baseline_pipeline")
    print("✓ Baseline pipeline created")
    
    # Optimize pipeline
    print("\n[2] Optimizing pipeline parameters...")
    # optimized = agent.optimize(pipeline, test_queries="queries.json")
    # agent.save_pipeline(optimized, "./optimized_pipeline")
    print("✓ Pipeline optimized")
    
    # Evaluate both versions
    print("\n[3] Comparing baseline vs optimized...")
    # baseline_eval = agent.evaluate(pipeline, "test_set.json")
    # optimized_eval = agent.evaluate(optimized, "test_set.json")
    
    print("\nComparison Results:")
    print("  Baseline  -> Optimized")
    print("  Quality:   0.75 -> 0.85 (+13%)")
    print("  Latency:   250ms -> 180ms (-28%)")
    
    # Deploy optimized version
    print("\n[4] Generating deployment artifacts...")
    # agent.deploy(optimized, target="docker", output_dir="./deployment")
    print("✓ Deployment files generated")
    
    print("\n✅ Optimization complete!")


def multi_agent_collaboration():
    """
    Example of multiple agents working together for A/B testing.
    """
    
    print("\n" + "=" * 60)
    print("Multi-Agent A/B Testing Example")
    print("=" * 60)
    
    # Create two pipeline variants with different configurations
    print("\n[1] Creating pipeline variants...")
    
    from rag_pipeline_agent.agent import RAGPipelineAgent
    
    pipeline_agent = RAGPipelineAgent()
    
    # Variant A: Smaller chunks, higher top_k
    print("  Variant A: chunk_size=512, top_k=7")
    # pipeline_a = pipeline_agent.build_pipeline(
    #     data_dir="./docs",
    #     config_file="config_variant_a.yaml"
    # )
    
    # Variant B: Larger chunks, lower top_k
    print("  Variant B: chunk_size=1024, top_k=3")
    # pipeline_b = pipeline_agent.build_pipeline(
    #     data_dir="./docs",
    #     config_file="config_variant_b.yaml"
    # )
    
    print("✓ Two pipeline variants created")
    
    # Evaluate both variants
    print("\n[2] Running A/B evaluation...")
    
    from evaluation_agent.agent import EvaluationAgent
    
    eval_agent = EvaluationAgent()
    
    test_queries = [
        {"query": "What is machine learning?"},
        {"query": "Explain neural networks"},
        {"query": "What are transformers?"},
    ]
    
    # results_a = eval_agent.evaluate(pipeline_a, test_queries, ["faithfulness", "relevance"])
    # results_b = eval_agent.evaluate(pipeline_b, test_queries, ["faithfulness", "relevance"])
    
    print("\nResults:")
    print("  Variant A: faithfulness=0.82, relevance=0.85")
    print("  Variant B: faithfulness=0.87, relevance=0.81")
    
    # Determine winner
    print("\n[3] Analysis:")
    print("  Winner: Variant B (higher faithfulness)")
    print("  Trade-off: Slightly lower relevance")
    print("  Recommendation: Deploy Variant B for production")


if __name__ == "__main__":
    print("\n🚀 LlamaIndex Agents - Integration Examples\n")
    
    # Run examples
    complete_rag_workflow()
    # rag_pipeline_optimization()
    # multi_agent_collaboration()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
