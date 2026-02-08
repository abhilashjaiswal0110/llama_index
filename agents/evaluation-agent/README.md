# Evaluation Agent

Comprehensive RAG evaluation and quality metrics for LlamaIndex. Measure, compare, and improve your RAG systems with automated evaluation.

## Features

- 📊 **Multiple metrics:** Faithfulness, relevance, coherence, correctness
- 🎯 **Retrieval evaluation:** Hit rate, MRR, NDCG, Precision@K
- 🔄 **End-to-end testing:** Complete pipeline evaluation
- 📈 **A/B comparison:** Compare system variants
- 🤖 **Auto test generation:** Generate test queries from documents
- 📝 **Detailed reports:** HTML/PDF evaluation reports
- ✅ **CI/CD integration:** Regression testing support

## Metrics

### Retrieval Metrics
- **Hit Rate:** % of queries with relevant results
- **MRR (Mean Reciprocal Rank):** Average rank of first relevant result
- **NDCG:** Normalized Discounted Cumulative Gain
- **Precision@K:** Precision at top K results

### Generation Metrics
- **Faithfulness:** Answer consistency with source documents
- **Answer Relevance:** Relevance to query
- **Context Relevance:** Retrieved context quality
- **Answer Correctness:** Accuracy vs ground truth

### Semantic Metrics
- **BLEU:** N-gram overlap
- **ROUGE:** Recall-oriented overlap
- **BERTScore:** Semantic similarity

## Quick Start

```bash
# Evaluate pipeline
python main.py evaluate --pipeline-path ./pipeline --test-set queries.json

# Compare two systems
python main.py compare --pipeline-a ./pipeline_v1 --pipeline-b ./pipeline_v2 --test-set queries.json

# Generate test queries
python main.py generate-tests --data-dir ./docs --output test_queries.json --num-queries 100

# Batch evaluation
python main.py batch --config eval_config.yaml
```

## Python API

```python
from evaluation_agent import EvaluationAgent, EvaluationConfig

# Initialize evaluator
agent = EvaluationAgent()

# Load pipeline
from llama_index.core import load_index_from_storage, StorageContext
storage = StorageContext.from_defaults(persist_dir="./pipeline")
index = load_index_from_storage(storage)
query_engine = index.as_query_engine()

# Configure evaluation
config = EvaluationConfig(
    metrics=["faithfulness", "relevance", "answer_correctness"],
    test_queries="queries.json"
)

# Run evaluation
results = agent.evaluate(query_engine, config)

print(f"Faithfulness: {results.metrics['faithfulness']:.3f}")
print(f"Relevance: {results.metrics['relevance']:.3f}")
print(f"Answer Correctness: {results.metrics['answer_correctness']:.3f}")

# Generate report
agent.generate_report(results, output_file="report.html")
```

## Test Query Format (queries.json)

```json
[
  {
    "query": "What is RAG?",
    "expected_answer": "Retrieval-Augmented Generation...",
    "context": ["document_id_1", "document_id_2"]
  },
  {
    "query": "How does vector search work?",
    "expected_answer": "Vector search uses embeddings...",
    "context": ["document_id_3"]
  }
]
```

## Advanced Usage

### Custom Metrics

```python
from evaluation_agent import EvaluationAgent

agent = EvaluationAgent()

# Define custom metric
def custom_metric(response, expected, context):
    # Your evaluation logic
    return 0.85

agent.register_metric("custom_score", custom_metric)

# Use in evaluation
results = agent.evaluate(
    query_engine,
    test_queries,
    metrics=["faithfulness", "custom_score"]
)
```

### A/B Testing

```python
# Load two pipeline variants
pipeline_a = load_pipeline("./pipeline_v1")
pipeline_b = load_pipeline("./pipeline_v2")

# Compare
comparison = agent.compare(
    pipelines={"baseline": pipeline_a, "variant": pipeline_b},
    test_queries="queries.json",
    metrics=["faithfulness", "relevance", "latency"]
)

print(f"Winner: {comparison.winner}")
print(f"Improvement: {comparison.improvement:.2%}")
```

### Batch Evaluation

```python
# Evaluate multiple configurations
configs = [
    {"top_k": 3, "temperature": 0.1},
    {"top_k": 5, "temperature": 0.3},
    {"top_k": 7, "temperature": 0.5},
]

results = agent.batch_evaluate(
    pipeline_dir="./pipeline",
    configs=configs,
    test_queries="queries.json"
)

# Find best configuration
best_config = max(results, key=lambda x: x['overall_score'])
```

### Generate Test Dataset

```python
# Auto-generate test queries from documents
test_queries = agent.generate_test_queries(
    data_dir="./docs",
    num_queries=100,
    difficulty_levels=["easy", "medium", "hard"],
    query_types=["factual", "analytical", "comparative"]
)

# Save test queries
agent.save_test_queries(test_queries, "generated_tests.json")
```

## Regression Testing

```python
# Set up regression testing
agent.setup_regression_test(
    pipeline_path="./pipeline",
    test_queries="regression_tests.json",
    baseline_metrics_file="baseline.json"
)

# Run regression test
regression_results = agent.run_regression_test()

if regression_results.passed:
    print("✅ All tests passed")
else:
    print("❌ Regression detected:")
    for failure in regression_results.failures:
        print(f"  {failure}")
```

## CI/CD Integration

```bash
# In your CI pipeline
python main.py evaluate \
  --pipeline-path ./pipeline \
  --test-set tests/queries.json \
  --baseline-file tests/baseline.json \
  --threshold 0.80 \
  --fail-on-regression

# Exit code 0 if passed, 1 if failed
```

## Metrics Details

### Faithfulness
Measures if answer is supported by retrieved context. Uses LLM to verify claims.

```python
results = agent.evaluate(query_engine, metrics=["faithfulness"])
# Score: 0.0 - 1.0 (higher is better)
```

### Answer Relevance
Measures how well answer addresses the query.

```python
results = agent.evaluate(query_engine, metrics=["relevance"])
# Score: 0.0 - 1.0 (higher is better)
```

### Context Relevance
Measures quality of retrieved context.

```python
results = agent.evaluate(query_engine, metrics=["context_relevance"])
# Score: 0.0 - 1.0 (higher is better)
```

## Examples

See `examples/` directory:
- `basic_evaluation.py` - Simple evaluation
- `ab_testing.py` - Compare two systems
- `batch_evaluation.py` - Multiple configurations
- `generate_tests.py` - Auto-generate test queries
- `regression_testing.py` - CI/CD integration
- `custom_metrics.py` - Define custom metrics

## Best Practices

1. **Build comprehensive test sets:**
   - Cover different query types
   - Include edge cases
   - Mix difficulty levels

2. **Track metrics over time:**
   - Save baseline metrics
   - Monitor trends
   - Detect regressions

3. **Use multiple metrics:**
   - Single metric can be misleading
   - Combine retrieval + generation metrics
   - Consider latency and cost

4. **Automate evaluation:**
   - Integrate in CI/CD
   - Run on every change
   - Set quality thresholds

## Troubleshooting

**Low faithfulness scores:**
- Improve retrieval (increase top_k)
- Enhance prompts
- Use better embedding model

**Low relevance scores:**
- Check query understanding
- Improve retrieval
- Refine prompt templates

**Inconsistent metrics:**
- Increase test set size
- Use more diverse queries
- Check for bias in test data

## License

Part of LlamaIndex project. Version 1.0.0
