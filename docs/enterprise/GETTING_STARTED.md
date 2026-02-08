# Getting Started with LlamaIndex

> **📍 You are here:** Complete guide to getting started with LlamaIndex for enterprise use.

## 🚀 Quick Start

LlamaIndex is a data framework for your LLM application. This guide will help you set up LlamaIndex locally and start building powerful LLM applications.

## 📋 Prerequisites

Before you begin, ensure you have the following:

- **Python 3.9+** - Required for running LlamaIndex
- **pip** or **uv** - Python package manager
- **API Keys** - For your chosen LLM provider (OpenAI, Anthropic, etc.)
- **Git** - For version control and repository management

### Checking Prerequisites

```bash
# Check Python version
python --version  # Should be 3.9 or higher

# Check pip
pip --version

# Check git
git --version
```

## 🔧 Installation

### Option 1: Starter Package (Recommended for Beginners)

The starter package includes core LlamaIndex and common integrations:

```bash
pip install llama-index
```

### Option 2: Core + Custom Integrations (Recommended for Production)

For more control, install core and only the integrations you need:

```bash
# Install core
pip install llama-index-core

# Install specific integrations
pip install llama-index-llms-openai
pip install llama-index-embeddings-huggingface
pip install llama-index-vector-stores-chroma
```

### Option 3: Using uv (Fast Alternative)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install LlamaIndex with uv
uv pip install llama-index
```

## 🔑 API Key Configuration

### OpenAI

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or in Python:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### Other Providers

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key"

# Replicate
export REPLICATE_API_TOKEN="your-token"

# Cohere
export COHERE_API_KEY="your-key"
```

## 🎯 Your First LlamaIndex Application

### 1. Create a Simple Vector Store Index

```python
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Set your API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Load documents from a directory
documents = SimpleDirectoryReader("data").load_data()

# Create an index
index = VectorStoreIndex.from_documents(documents)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic of these documents?")
print(response)
```

### 2. Persist and Load Index

```python
# Save the index
index.storage_context.persist(persist_dir="./storage")

# Load the index later
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

### 3. Using Non-OpenAI LLMs

```python
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure LLM
Settings.llm = Anthropic(model="claude-3-sonnet-20240229", api_key="your-key")

# Configure embeddings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Now create your index
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
```

## 📁 Project Structure

Recommended structure for your LlamaIndex project:

```
my-llama-project/
├── data/                  # Your source documents
│   ├── docs/
│   └── pdfs/
├── storage/              # Persisted indices
├── src/
│   ├── __init__.py
│   ├── indexing.py      # Index creation logic
│   ├── querying.py      # Query logic
│   └── config.py        # Configuration
├── notebooks/           # Jupyter notebooks for exploration
├── tests/              # Unit tests
├── .env               # Environment variables (API keys)
├── .gitignore        # Don't commit API keys!
├── requirements.txt  # Dependencies
└── README.md        # Project documentation
```

## 🎓 Next Steps

### Learn Core Concepts

1. **[LOCAL_SETUP.md](./LOCAL_SETUP.md)** - Set up your development environment
2. **[USAGE_GUIDE.md](./USAGE_GUIDE.md)** - Comprehensive usage guide
3. **[EXAMPLES.md](./EXAMPLES.md)** - Practical examples
4. **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Understand the architecture

### Explore Use Cases

- **Data Ingestion**: Connect to APIs, databases, PDFs, and more
- **Semantic Search**: Build powerful search engines
- **Question Answering**: Create chatbots and Q&A systems
- **RAG Applications**: Retrieval-Augmented Generation
- **Agents**: Build autonomous AI agents

### Try the Agents

Navigate to the `/agents` directory to find specialized agent skills for common tasks:

- **Data Ingestion Agent** - Automate data loading and processing
- **Query Engine Agent** - Advanced query configurations
- **RAG Pipeline Agent** - Build complete RAG systems
- **Evaluation Agent** - Evaluate your RAG performance

## 📚 Resources

### Official Documentation
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [API Reference](https://docs.llamaindex.ai/en/stable/api_reference/)
- [LlamaHub](https://llamahub.ai/) - 300+ integrations

### Community
- [Discord](https://discord.gg/dGcwcsnxhU) - Join the community
- [Twitter/X](https://x.com/llama_index) - Latest updates
- [GitHub](https://github.com/run-llama/llama_index) - Source code and issues

### Learning Resources
- [Official Examples](https://docs.llamaindex.ai/en/stable/examples/)
- [Tutorials](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/)
- [YouTube Channel](https://www.youtube.com/@LlamaIndex)

## ⚠️ Common Issues

### Installation Problems

**Issue**: `ModuleNotFoundError: No module named 'llama_index'`

**Solution**:
```bash
pip install --upgrade llama-index
```

**Issue**: Conflicting dependencies

**Solution**:
```bash
# Use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install llama-index
```

### API Key Issues

**Issue**: `AuthenticationError: Invalid API key`

**Solution**: Verify your API key is correctly set and has not expired.

### Performance Issues

**Issue**: Slow indexing or querying

**Solution**:
- Use smaller chunk sizes
- Consider using a vector database (Chroma, Pinecone, etc.)
- Use local embeddings instead of API-based ones

## 🔒 Security Best Practices

1. **Never commit API keys** - Use environment variables or `.env` files
2. **Add `.env` to `.gitignore`**
3. **Rotate keys regularly**
4. **Use different keys for dev/prod**
5. **Monitor API usage** to detect anomalies

## 💡 Tips for Success

1. **Start small** - Begin with a simple example and expand
2. **Use the right chunk size** - Experiment with different values
3. **Choose the right LLM** - Consider cost, speed, and quality
4. **Test your queries** - Validate responses match expectations
5. **Monitor costs** - Track API usage to avoid surprises

## 🤝 Getting Help

If you're stuck:

1. **Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)** for common solutions
2. **Search existing [GitHub Issues](https://github.com/run-llama/llama_index/issues)**
3. **Ask on [Discord](https://discord.gg/dGcwcsnxhU)**
4. **Review [examples](./EXAMPLES.md)** for similar use cases

---

**Ready to dive deeper?** Continue to [LOCAL_SETUP.md](./LOCAL_SETUP.md) for development environment setup! 🚀
