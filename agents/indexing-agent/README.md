# Indexing Agent

Advanced indexing strategies and optimization for LlamaIndex. Choose, create, and optimize the right index for your use case.

## Features

- 📊 **Multiple index types:** Vector, Tree, Keyword, List, Knowledge Graph
- 🎯 **Intelligent selection:** Auto-recommend best index type
- ⚡ **Performance optimization:** Chunk size, embedding model tuning
- 🔄 **Index composition:** Combine multiple index types
- 📈 **Benchmarking:** Compare index performance
- 💾 **Storage backends:** Local, Pinecone, Weaviate, Chroma

## Index Types

| Type | Best For | Retrieval Speed | Memory |
|------|----------|----------------|--------|
| `vector` | General-purpose semantic search | Fast | Medium |
| `tree` | Summarization, hierarchical data | Medium | Low |
| `keyword` | Exact match, keyword search | Very Fast | Low |
| `list` | Small datasets, simple retrieval | Fast | Very Low |
| `document-summary` | Document-level retrieval | Fast | Low |
| `knowledge-graph` | Relationship queries | Medium | High |

## Quick Start

```bash
# Create vector index (default)
python main.py create --data-dir ./docs --index-type vector

# Create with optimization
python main.py create --data-dir ./docs --index-type vector --optimize

# Compare index types
python main.py compare --data-dir ./docs --types vector,tree,keyword

# Benchmark performance
python main.py benchmark --index-path ./index --test-queries queries.json
```

## Python API

```python
from indexing_agent import IndexingAgent

agent = IndexingAgent()

# Create vector index
index = agent.create_index(
    data_dir="./docs",
    index_type="vector",
    chunk_size=1024,
    embedding_model="text-embedding-ada-002"
)

# Optimize index
optimized_index = agent.optimize_index(index, test_queries="queries.json")

# Compare index types
results = agent.compare_indexes(
    data_dir="./docs",
    index_types=["vector", "tree", "keyword"]
)

# Get recommendations
recommendation = agent.recommend_index_type(data_dir="./docs")
print(f"Recommended: {recommendation['type']}")
```

## Configuration

```yaml
index:
  type: "vector"
  chunk_size: 1024
  chunk_overlap: 200
  embedding_model: "text-embedding-ada-002"
  
storage:
  backend: "local"  # local, pinecone, weaviate, chroma
  persist_dir: "./index"
  
optimization:
  auto_tune: true
  target_metric: "retrieval_quality"  # retrieval_quality, speed, memory
```

## License

Part of LlamaIndex project. Version 1.0.0
