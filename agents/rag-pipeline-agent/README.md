# RAG Pipeline Agent

Complete end-to-end RAG pipeline builder and orchestrator for LlamaIndex. Build, optimize, evaluate, and deploy production-ready RAG systems.

## Features

- 🔄 **End-to-end pipelines:** From data ingestion to deployment
- ⚙️ **Component orchestration:** Index, retriever, query engine, response synthesis
- 🎯 **Auto-optimization:** Automatic parameter tuning
- 📊 **Built-in evaluation:** Quality metrics and benchmarking
- 🚀 **Production templates:** Deployment-ready configurations
- 📈 **Performance monitoring:** Real-time pipeline health checks
- 🔧 **A/B testing:** Compare pipeline configurations

## Modes

| Mode | Description | Output |
|------|-------------|--------|
| `build` | Create complete RAG pipeline | Configured pipeline |
| `optimize` | Tune pipeline parameters | Optimized config |
| `evaluate` | Run evaluation metrics | Performance report |
| `deploy` | Generate deployment artifacts | Docker/K8s configs |
| `monitor` | Pipeline health monitoring | Metrics dashboard |
| `compare` | A/B test configurations | Comparison report |

## Installation

```bash
cd rag-pipeline-agent
pip install -r requirements.txt
cp .env.example .env
```

## Quick Start

```bash
# Build a complete RAG pipeline
python main.py build --data-dir ./docs --config pipeline.yaml

# Optimize existing pipeline
python main.py optimize --pipeline-path ./pipeline --test-queries queries.json

# Evaluate pipeline
python main.py evaluate --pipeline-path ./pipeline --test-set test_queries.json

# Generate deployment configs
python main.py deploy --pipeline-path ./pipeline --target docker

# Monitor running pipeline
python main.py monitor --endpoint http://localhost:8000
```

## Pipeline Configuration (pipeline.yaml)

```yaml
pipeline:
  name: "my_rag_system"
  description: "Production RAG for documentation"

data:
  source_dir: "./docs"
  file_types: ["pdf", "md", "txt"]
  
indexing:
  index_type: "vector"
  chunk_size: 1024
  chunk_overlap: 200
  embedding_model: "text-embedding-ada-002"
  
retrieval:
  top_k: 5
  similarity_threshold: 0.75
  retriever_mode: "hybrid"
  
generation:
  model: "gpt-4"
  temperature: 0.1
  max_tokens: 500
  
evaluation:
  metrics: ["faithfulness", "relevance", "coherence"]
  test_queries_file: "test_queries.json"
```

## Python API

### Build Complete Pipeline

```python
from rag_pipeline_agent import RAGPipelineAgent, PipelineConfig

# Define configuration
config = PipelineConfig(
    name="documentation_rag",
    data_dir="./docs",
    index_type="vector",
    chunk_size=1024,
    retriever_mode="hybrid"
)

# Build pipeline
agent = RAGPipelineAgent()
pipeline = agent.build_pipeline(config)

# Save pipeline
agent.save_pipeline(pipeline, "./my_pipeline")

# Use pipeline
response = pipeline.query("What is RAG?")
print(response.response)
```

### Optimize Pipeline

```python
from rag_pipeline_agent import RAGPipelineAgent

agent = RAGPipelineAgent()

# Load existing pipeline
pipeline = agent.load_pipeline("./my_pipeline")

# Optimize with test queries
optimized_pipeline = agent.optimize(
    pipeline,
    test_queries="queries.json",
    optimization_targets=["latency", "quality"],
    max_iterations=10
)

# Compare before/after
comparison = agent.compare_pipelines(pipeline, optimized_pipeline)
print(f"Quality improvement: {comparison['quality_delta']:.2%}")
print(f"Latency improvement: {comparison['latency_delta']:.2%}")
```

### Evaluate Pipeline

```python
from rag_pipeline_agent import RAGPipelineAgent, EvaluationConfig

agent = RAGPipelineAgent()
pipeline = agent.load_pipeline("./my_pipeline")

# Configure evaluation
eval_config = EvaluationConfig(
    metrics=["faithfulness", "relevance", "answer_correctness"],
    test_set="test_queries.json"
)

# Run evaluation
results = agent.evaluate(pipeline, eval_config)

print(f"Faithfulness: {results.metrics['faithfulness']:.3f}")
print(f"Relevance: {results.metrics['relevance']:.3f}")
print(f"Answer Correctness: {results.metrics['answer_correctness']:.3f}")
```

### Deploy Pipeline

```python
from rag_pipeline_agent import RAGPipelineAgent

agent = RAGPipelineAgent()
pipeline = agent.load_pipeline("./my_pipeline")

# Generate Docker deployment
agent.deploy(
    pipeline,
    target="docker",
    output_dir="./deployment",
    include_monitoring=True
)

# Generate Kubernetes deployment
agent.deploy(
    pipeline,
    target="kubernetes",
    output_dir="./k8s",
    replicas=3,
    auto_scaling=True
)

# Generate FastAPI service
agent.deploy(
    pipeline,
    target="fastapi",
    output_dir="./api",
    include_swagger=True
)
```

## Pipeline Components

### 1. Data Ingestion
```python
pipeline.configure_ingestion(
    sources=["./docs", "https://example.com/docs"],
    file_types=["pdf", "md"],
    preprocessing={
        "clean_text": True,
        "extract_metadata": True
    }
)
```

### 2. Indexing
```python
pipeline.configure_indexing(
    index_type="vector",
    embedding_model="text-embedding-ada-002",
    chunk_size=1024,
    chunk_overlap=200,
    storage_backend="local"  # or "pinecone", "weaviate"
)
```

### 3. Retrieval
```python
pipeline.configure_retrieval(
    retriever_mode="hybrid",
    top_k=5,
    similarity_threshold=0.75,
    enable_mmr=True,
    enable_reranking=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

### 4. Generation
```python
pipeline.configure_generation(
    llm_model="gpt-4",
    temperature=0.1,
    max_tokens=500,
    prompt_template="custom_template.txt",
    enable_streaming=True
)
```

### 5. Post-processing
```python
pipeline.configure_postprocessing(
    enable_citation=True,
    enable_fact_check=True,
    response_format="markdown"
)
```

## Monitoring & Observability

```python
# Enable monitoring
pipeline.enable_monitoring(
    metrics=["latency", "token_usage", "retrieval_quality"],
    export_to="prometheus",
    dashboard_port=9090
)

# Get real-time metrics
metrics = pipeline.get_metrics()
print(f"Avg latency: {metrics['avg_latency_ms']}ms")
print(f"Total queries: {metrics['total_queries']}")
print(f"Success rate: {metrics['success_rate']:.2%}")

# Set up alerts
pipeline.add_alert(
    metric="latency",
    threshold=1000,  # ms
    action="email",
    recipients=["ops@example.com"]
)
```

## A/B Testing

```python
# Create variant pipelines
pipeline_a = agent.build_pipeline(config_a)
pipeline_b = agent.build_pipeline(config_b)

# Run A/B test
test_results = agent.ab_test(
    pipelines={"baseline": pipeline_a, "variant": pipeline_b},
    test_queries="queries.json",
    traffic_split={"baseline": 0.5, "variant": 0.5},
    duration_minutes=60
)

# Analyze results
print(test_results.winner)  # "variant"
print(test_results.confidence)  # 0.95
print(test_results.improvement)  # {"quality": 0.15, "latency": -0.10}
```

## Production Deployment

### Docker
```bash
# Generate Docker deployment
python main.py deploy --target docker --output ./deploy

cd deploy
docker build -t my-rag-system .
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY my-rag-system
```

### Kubernetes
```bash
# Generate K8s manifests
python main.py deploy --target kubernetes --output ./k8s

kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

### FastAPI Service
```bash
# Generate FastAPI application
python main.py deploy --target fastapi --output ./api

cd api
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Examples

See `examples/` directory:
- `build_basic_pipeline.py` - Basic pipeline setup
- `optimize_pipeline.py` - Parameter optimization
- `evaluate_pipeline.py` - Evaluation workflow
- `deploy_production.py` - Production deployment
- `ab_testing.py` - A/B testing example
- `monitoring_setup.py` - Monitoring configuration

## Best Practices

1. **Start simple, iterate:**
   - Begin with default configurations
   - Optimize based on metrics
   - Gradually add complexity

2. **Evaluate continuously:**
   - Set up automated evaluation
   - Track metrics over time
   - Regression test on updates

3. **Monitor in production:**
   - Track latency and quality
   - Set up alerting
   - Review logs regularly

4. **Version control pipelines:**
   - Save configurations
   - Document changes
   - Enable rollbacks

## Troubleshooting

**Pipeline build fails:**
- Check data directory access
- Verify API keys
- Review log files

**Low evaluation scores:**
- Increase chunk size
- Add more data
- Tune retrieval parameters

**High latency:**
- Reduce top_k
- Enable caching
- Optimize index

**Deployment issues:**
- Check resource limits
- Verify network access
- Review container logs

## License

Part of LlamaIndex project.

## Version

Current version: 1.0.0
