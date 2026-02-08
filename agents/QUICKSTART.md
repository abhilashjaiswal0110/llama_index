# Quick Start Guide - LlamaIndex Agents

Get started with LlamaIndex specialized agents in 5 minutes.

## Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install core dependencies
pip install llama-index llama-index-core click rich pyyaml python-dotenv
```

## Choose Your Use Case

### 1. Loading Data → Use Data Ingestion Agent

```bash
cd data-ingestion-agent

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Process documents
python main.py pdf --path document.pdf
python main.py directory --path ./docs --recursive
```

### 2. Querying Data → Use Query Engine Agent

```bash
cd query-engine-agent

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env

# Query your index
python main.py query --mode similarity --query "What is RAG?" --index-path ../data-ingestion-agent/index
```

### 3. Building Complete RAG → Use RAG Pipeline Agent

```bash
cd rag-pipeline-agent

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env

# Build end-to-end pipeline
python main.py build --data-dir ./docs --output ./my_pipeline

# Evaluate pipeline
python main.py evaluate --pipeline-path ./my_pipeline --test-set queries.json
```

### 4. Optimizing Indexes → Use Indexing Agent

```bash
cd indexing-agent

# Install dependencies
pip install -r requirements.txt

# Create optimized index
python main.py create --data-dir ./docs --index-type vector --optimize

# Compare index types
python main.py compare --data-dir ./docs --types vector,tree,keyword
```

### 5. Measuring Quality → Use Evaluation Agent

```bash
cd evaluation-agent

# Install dependencies
pip install -r requirements.txt

# Evaluate your RAG system
python main.py evaluate --pipeline-path ./pipeline --test-set test_queries.json

# Generate test queries
python main.py generate-tests --data-dir ./docs --num-queries 50
```

## Complete Workflow Example

```bash
# 1. Ingest data
cd data-ingestion-agent
python main.py directory --path ../sample_docs --output ../workflow_index

# 2. Query the index
cd ../query-engine-agent
python main.py query --mode hybrid --query "What is machine learning?" --index-path ../workflow_index

# 3. Evaluate quality
cd ../evaluation-agent
python main.py evaluate --pipeline-path ../workflow_index --test-set test_queries.json
```

## Python API Quick Start

```python
# Complete RAG in a few lines
from data_ingestion_agent.agent import DataIngestionAgent
from llama_index.core import VectorStoreIndex

# 1. Ingest
agent = DataIngestionAgent()
documents = agent.ingest_directory("./docs")

# 2. Index
index = VectorStoreIndex.from_documents(documents)

# 3. Query
query_engine = index.as_query_engine()
response = query_engine.query("What is RAG?")
print(response.response)
```

## Common Commands Cheat Sheet

### Data Ingestion
```bash
# Single PDF
python main.py pdf --path doc.pdf

# Directory of files
python main.py directory --path ./docs --recursive

# Web scraping
python main.py web --url https://example.com --depth 2

# Database
python main.py database --connection "postgresql://..." --query "SELECT * FROM docs"
```

### Query Engine
```bash
# Similarity search
python main.py query --mode similarity --query "your question"

# Hybrid search
python main.py query --mode hybrid --query "your question" --top-k 5

# Sub-question decomposition
python main.py query --mode sub-question --query "complex question"
```

### RAG Pipeline
```bash
# Build pipeline
python main.py build --data-dir ./docs

# Optimize
python main.py optimize --pipeline-path ./pipeline --test-queries queries.json

# Deploy
python main.py deploy --pipeline-path ./pipeline --target docker
```

### Evaluation
```bash
# Evaluate
python main.py evaluate --pipeline-path ./pipeline --test-set queries.json

# Compare
python main.py compare --pipeline-a ./v1 --pipeline-b ./v2 --test-set queries.json

# Generate tests
python main.py generate-tests --data-dir ./docs --output tests.json
```

## Test Queries Format

Create `test_queries.json`:
```json
[
  {
    "query": "What is RAG?",
    "expected_answer": "Retrieval-Augmented Generation..."
  },
  {
    "query": "How does vector search work?",
    "expected_answer": "Vector search uses embeddings..."
  }
]
```

## Configuration

Each agent uses:
- `.env` for secrets (API keys)
- `config.yaml` for settings
- CLI arguments for overrides

Example `.env`:
```bash
OPENAI_API_KEY=sk-...
```

Example `config.yaml`:
```yaml
chunk_size: 1024
chunk_overlap: 200
top_k: 3
```

## Troubleshooting

**ImportError: No module named 'llama_index'**
```bash
pip install llama-index llama-index-core
```

**API Key Error**
```bash
# Make sure .env file has your key
echo "OPENAI_API_KEY=sk-your-key" > .env
```

**Index not found**
```bash
# Check the path
ls -la ./index

# Rebuild if needed
python data-ingestion-agent/main.py directory --path ./docs --output ./index
```

## Next Steps

1. Read individual agent READMEs for detailed documentation
2. Check `examples/` directories for advanced usage
3. Customize `config.yaml` for your use case
4. Set up evaluation with your test queries
5. Deploy to production using RAG Pipeline Agent

## Support

- **Documentation:** See README.md in each agent directory
- **Examples:** Check examples/ folders
- **Issues:** GitHub Issues
- **Integration:** See examples_integration.py

## Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai)
- [Main README](./README.md) - Detailed agent comparison
- Individual agent READMEs for deep dives
