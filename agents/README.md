# LlamaIndex Specialized Agents

This directory contains production-ready specialized agents for common LlamaIndex use cases. Each agent provides focused functionality for specific stages of the RAG (Retrieval-Augmented Generation) pipeline.

## Overview

These agents are designed to simplify and accelerate LlamaIndex development by providing:

- **Pre-built solutions** for common RAG patterns
- **Production-ready code** with error handling and logging
- **Flexible configuration** for different use cases
- **CLI interfaces** for easy integration
- **Comprehensive documentation** and examples

## Quick Start

```bash
# Navigate to any agent directory
cd data-ingestion-agent/

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Add your API keys to .env

# Run the agent
python main.py --help
```

## Available Agents

### 📥 Data Ingestion Agent
**Purpose:** Load and process data from multiple sources into LlamaIndex

**Modes:**
- `pdf` - Process PDF documents with metadata extraction
- `web` - Scrape and index web content
- `api` - Ingest data from REST APIs
- `database` - Connect to SQL/NoSQL databases
- `directory` - Batch process file directories

**Key Features:**
- Multi-format document loading (PDF, DOCX, TXT, MD, HTML, JSON)
- Automatic metadata extraction and enrichment
- Chunking strategies (sentence, semantic, fixed-size)
- Deduplication and cleaning
- Progress tracking and error recovery

**Use Cases:**
- Building knowledge bases from document collections
- Ingesting enterprise documentation
- Creating searchable archives

[View Documentation →](./data-ingestion-agent/README.md)

---

### 🔍 Query Engine Agent
**Purpose:** Advanced querying with multiple retrieval strategies

**Modes:**
- `similarity` - Vector similarity search
- `hybrid` - Combine vector and keyword search
- `sub-question` - Break complex queries into sub-questions
- `fusion` - Query fusion with multiple strategies
- `tree` - Tree-based query decomposition

**Key Features:**
- Multiple retrieval modes (top-k, similarity threshold, MMR)
- Custom prompt templates
- Query transformation and expansion
- Response synthesis strategies
- Streaming support

**Use Cases:**
- Complex question answering
- Multi-hop reasoning
- Comparative analysis

[View Documentation →](./query-engine-agent/README.md)

---

### 🔄 RAG Pipeline Agent
**Purpose:** End-to-end RAG pipeline orchestration and management

**Modes:**
- `build` - Create complete RAG pipeline from scratch
- `optimize` - Tune pipeline parameters
- `evaluate` - Run evaluation metrics
- `deploy` - Generate deployment artifacts
- `monitor` - Pipeline health monitoring

**Key Features:**
- Complete RAG workflow automation
- Component selection and configuration
- Parameter tuning and optimization
- Built-in evaluation framework
- Production deployment templates
- Performance monitoring

**Use Cases:**
- Rapid RAG prototyping
- Pipeline experimentation
- Production deployment preparation
- A/B testing different configurations

[View Documentation →](./rag-pipeline-agent/README.md)

---

### 📊 Indexing Agent
**Purpose:** Create and optimize indexes for different use cases

**Index Types:**
- `vector` - Vector store index (default)
- `tree` - Tree-based summarization index
- `keyword` - Keyword table index
- `list` - Simple list index
- `document-summary` - Document summary index
- `knowledge-graph` - Knowledge graph index
- `multi-modal` - Multi-modal index

**Key Features:**
- Intelligent index type selection
- Embedding model comparison
- Chunk size optimization
- Index composition strategies
- Storage backend configuration
- Performance benchmarking

**Use Cases:**
- Choosing the right index for your data
- Optimizing retrieval performance
- Combining multiple index types
- Large-scale document indexing

[View Documentation →](./indexing-agent/README.md)

---

### 📈 Evaluation Agent
**Purpose:** Comprehensive RAG evaluation and quality metrics

**Evaluation Types:**
- `retrieval` - Retrieval quality metrics (MRR, NDCG, Hit Rate)
- `generation` - Generation quality (faithfulness, relevance)
- `end-to-end` - Complete pipeline evaluation
- `compare` - A/B comparison between systems
- `batch` - Batch evaluation with test datasets

**Metrics:**
- **Retrieval:** Hit Rate, MRR, NDCG, Precision@K
- **Generation:** Faithfulness, Answer Relevance, Context Relevance
- **Semantic:** BLEU, ROUGE, BERTScore
- **Custom:** Define your own evaluation criteria

**Key Features:**
- Multiple evaluation frameworks
- Automated test dataset generation
- Detailed metric reports
- Visualization and dashboards
- Regression testing
- CI/CD integration

**Use Cases:**
- Measuring RAG system quality
- Comparing different approaches
- Regression testing
- Continuous improvement

[View Documentation →](./evaluation-agent/README.md)

---

## Agent Comparison

| Agent | Purpose | Input | Output | Use When |
|-------|---------|-------|--------|----------|
| **Data Ingestion** | Load data | Raw files/APIs | Indexed documents | Starting a new project |
| **Query Engine** | Query data | Questions | Answers | Need advanced retrieval |
| **RAG Pipeline** | Full pipeline | Requirements | Complete system | Building end-to-end RAG |
| **Indexing** | Optimize indexes | Documents | Optimized index | Performance matters |
| **Evaluation** | Measure quality | Test queries | Metrics report | Validating system |

## When to Use Which Agent

### Starting a New Project
1. **Data Ingestion Agent** - Load your data
2. **Indexing Agent** - Choose and create optimal index
3. **Query Engine Agent** - Set up querying
4. **Evaluation Agent** - Validate performance

### Have Existing RAG System
- **RAG Pipeline Agent** - Rebuild with best practices
- **Evaluation Agent** - Measure current performance
- **Indexing Agent** - Optimize retrieval

### Production Deployment
1. **RAG Pipeline Agent** - Generate deployment config
2. **Evaluation Agent** - Set up monitoring
3. **Query Engine Agent** - Configure production querying

### Improving Performance
1. **Evaluation Agent** - Identify bottlenecks
2. **Indexing Agent** - Optimize index
3. **Query Engine Agent** - Tune retrieval
4. **Evaluation Agent** - Validate improvements

## Integration Examples

### Sequential Pipeline
```bash
# 1. Ingest data
cd data-ingestion-agent
python main.py --mode directory --path /path/to/docs

# 2. Build index
cd ../indexing-agent
python main.py --mode vector --optimize

# 3. Create pipeline
cd ../rag-pipeline-agent
python main.py --mode build --config config.yaml

# 4. Evaluate
cd ../evaluation-agent
python main.py --mode end-to-end --test-file queries.json
```

### Python Integration
```python
from llama_index.agents.data_ingestion import DataIngestionAgent
from llama_index.agents.indexing import IndexingAgent
from llama_index.agents.query_engine import QueryEngineAgent
from llama_index.agents.evaluation import EvaluationAgent

# Ingest data
ingestion = DataIngestionAgent()
documents = ingestion.ingest(source="./docs", mode="directory")

# Create optimized index
indexing = IndexingAgent()
index = indexing.create_index(documents, index_type="vector", optimize=True)

# Set up query engine
query_engine = QueryEngineAgent(index)
response = query_engine.query("What is RAG?", mode="hybrid")

# Evaluate
evaluator = EvaluationAgent()
metrics = evaluator.evaluate(query_engine, test_queries="test_set.json")
print(f"Faithfulness: {metrics['faithfulness']:.2f}")
```

## Architecture

All agents follow a consistent architecture:

```
agent-name/
├── README.md              # Comprehensive documentation
├── main.py               # CLI interface
├── agent.py              # Core agent logic
├── config.py             # Configuration management
├── utils.py              # Helper functions
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
├── config.yaml           # Default configuration
├── examples/             # Example scripts
│   ├── basic_usage.py
│   ├── advanced_usage.py
│   └── integration.py
└── tests/                # Unit tests
    ├── test_agent.py
    └── test_utils.py
```

## Best Practices

### Configuration Management
- Use `.env` for secrets (API keys, credentials)
- Use `config.yaml` for application settings
- Override via CLI arguments for flexibility

### Error Handling
- All agents implement comprehensive error handling
- Graceful degradation when possible
- Detailed error messages with recovery suggestions

### Logging
- Structured logging to files and console
- Configurable log levels
- Performance metrics tracking

### Type Safety
- Full type hints for all functions
- Pydantic models for configuration validation
- Runtime type checking where critical

## Requirements

### Core Dependencies
```
llama-index>=0.10.0
python-dotenv>=1.0.0
pyyaml>=6.0
click>=8.1.0
rich>=13.0.0
pydantic>=2.0.0
```

### Optional Dependencies
```
# For PDF processing
pypdf>=3.0.0

# For web scraping
beautifulsoup4>=4.12.0
requests>=2.31.0

# For evaluation
ragas>=0.1.0
bert-score>=0.3.13

# For monitoring
prometheus-client>=0.19.0
```

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run specific agent tests
pytest data-ingestion-agent/tests/

# Run with coverage
pytest --cov=agents
```

### Contributing
1. Follow the existing agent structure
2. Include comprehensive documentation
3. Add example usage scripts
4. Include unit tests
5. Update this README with new agents

## Support

- **Documentation:** Each agent has detailed README
- **Examples:** Check `examples/` in each agent directory
- **Issues:** Report issues in the main repository
- **Discussions:** Use GitHub Discussions for questions

## License

These agents are part of the LlamaIndex project and follow the same license.

## Version History

- **v1.0.0** (2024-02) - Initial release with 5 specialized agents
  - Data Ingestion Agent
  - Query Engine Agent
  - RAG Pipeline Agent
  - Indexing Agent
  - Evaluation Agent
