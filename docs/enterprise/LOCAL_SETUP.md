# Local Development Setup

> **📍 You are here:** Guide to setting up a professional development environment for LlamaIndex.

## 🎯 Overview

This guide will help you set up a complete local development environment for working with LlamaIndex, following enterprise best practices.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.9 or higher
- **RAM**: 8GB (16GB recommended for large datasets)
- **Disk Space**: 10GB free space
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)

### Recommended Requirements
- **Python**: 3.10 or 3.11
- **RAM**: 16GB or more
- **Disk Space**: 50GB+ for embeddings and indices
- **GPU**: NVIDIA GPU with CUDA support (optional, for local embeddings)

## 🔧 Development Environment Setup

### 1. Install Python

#### macOS (using Homebrew)
```bash
brew install python@3.11
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

#### Windows
Download from [python.org](https://www.python.org/downloads/) and install.

### 2. Set Up Virtual Environment

Always use a virtual environment to isolate dependencies:

```bash
# Navigate to your project directory
cd my-llama-project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Verify activation (you should see (venv) in your prompt)
which python  # Should point to venv/bin/python
```

### 3. Install LlamaIndex and Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Option A: Install starter package
pip install llama-index

# Option B: Install core with specific integrations
pip install llama-index-core
pip install llama-index-llms-openai llama-index-llms-anthropic
pip install llama-index-embeddings-huggingface
pip install llama-index-vector-stores-chroma
pip install llama-index-readers-file
```

### 4. Install Development Tools

```bash
# Code quality tools
pip install black isort flake8 mypy

# Testing tools
pip install pytest pytest-asyncio pytest-cov

# Jupyter for experimentation
pip install jupyter notebook ipykernel

# Monitoring and debugging
pip install rich tenacity

# Create requirements files
pip freeze > requirements.txt
```

## 📁 Project Structure

Create a well-organized project structure:

```bash
my-llama-project/
├── .env                    # Environment variables (DON'T COMMIT!)
├── .env.example           # Template for environment variables
├── .gitignore            # Git ignore file
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup (if distributing)
├── pyproject.toml       # Modern Python project config
├── data/                # Source data (add to .gitignore if sensitive)
│   ├── raw/
│   ├── processed/
│   └── README.md
├── storage/             # Persisted indices (add to .gitignore)
│   └── .gitkeep
├── src/                 # Source code
│   ├── __init__.py
│   ├── config.py       # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py  # Data loading utilities
│   │   └── processors.py
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── builders.py # Index building
│   │   └── storage.py  # Storage management
│   ├── querying/
│   │   ├── __init__.py
│   │   ├── engines.py  # Query engines
│   │   └── prompts.py  # Custom prompts
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/              # Test files
│   ├── __init__.py
│   ├── conftest.py    # Pytest configuration
│   ├── unit/
│   └── integration/
├── notebooks/         # Jupyter notebooks
│   ├── 01_exploration.ipynb
│   ├── 02_experiments.ipynb
│   └── README.md
├── scripts/          # Utility scripts
│   ├── setup.sh
│   └── index_data.py
└── docs/            # Additional documentation
    └── architecture.md
```

## 🔐 Environment Configuration

### Create .env File

```bash
# .env (DO NOT COMMIT THIS FILE!)
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Vector Store
CHROMA_PERSIST_DIR=./storage/chroma

# Application Settings
LOG_LEVEL=INFO
CHUNK_SIZE=1024
CHUNK_OVERLAP=20
```

### Create .env.example

```bash
# .env.example (SAFE TO COMMIT)
# Copy this to .env and fill in your values

# OpenAI
OPENAI_API_KEY=your_openai_key_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key_here

# Vector Store
CHROMA_PERSIST_DIR=./storage/chroma

# Application Settings
LOG_LEVEL=INFO
CHUNK_SIZE=1024
CHUNK_OVERLAP=20
```

### Update .gitignore

```bash
# Add to .gitignore
.env
storage/
data/
*.pyc
__pycache__/
venv/
.venv/
*.egg-info/
.pytest_cache/
.coverage
.ipynb_checkpoints/
```

## 🧪 Configuration Management

Create a config.py file:

```python
# src/config.py
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration."""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    STORAGE_DIR = PROJECT_ROOT / "storage"
    
    # Indexing Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1024"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "20"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls):
        """Validate configuration."""
        if not cls.OPENAI_API_KEY and not cls.ANTHROPIC_API_KEY:
            raise ValueError("At least one LLM API key must be set")
        
        # Create directories
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.STORAGE_DIR.mkdir(exist_ok=True)

# Validate on import
Config.validate()
```

## 🧪 Testing Setup

### Create pytest Configuration

```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory with sample data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create sample file
    (data_dir / "sample.txt").write_text("This is a test document.")
    
    return data_dir

@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        "chunk_size": 512,
        "chunk_overlap": 50,
    }
```

### Example Test

```python
# tests/unit/test_indexing.py
import pytest
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

def test_create_index(sample_data_dir):
    """Test index creation."""
    documents = SimpleDirectoryReader(str(sample_data_dir)).load_data()
    assert len(documents) > 0
    
    index = VectorStoreIndex.from_documents(documents)
    assert index is not None
```

## 🚀 Running Your Application

### Create a Main Entry Point

```python
# src/main.py
import logging
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from src.config import Config

# Setup logging
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

def main():
    """Main application entry point."""
    logger.info("Starting LlamaIndex application...")
    
    # Load documents
    logger.info(f"Loading documents from {Config.DATA_DIR}")
    documents = SimpleDirectoryReader(str(Config.DATA_DIR / "raw")).load_data()
    logger.info(f"Loaded {len(documents)} documents")
    
    # Create index
    logger.info("Creating index...")
    index = VectorStoreIndex.from_documents(documents)
    
    # Persist index
    logger.info(f"Persisting index to {Config.STORAGE_DIR}")
    index.storage_context.persist(persist_dir=str(Config.STORAGE_DIR))
    
    # Query
    query_engine = index.as_query_engine()
    response = query_engine.query("What are the main topics?")
    logger.info(f"Response: {response}")

if __name__ == "__main__":
    main()
```

### Run Your Application

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the application
python src/main.py

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🛠️ Development Tools

### Code Formatting

```bash
# Format code with black
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Or JupyterLab
jupyter lab
```

## 🐳 Docker Setup (Optional)

Create a Dockerfile for containerized development:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run application
CMD ["python", "src/main.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    volumes:
      - .:/app
      - ./storage:/app/storage
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    command: python src/main.py
```

## 🔍 Troubleshooting

### Virtual Environment Issues

**Problem**: `command not found: python`

**Solution**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate
```

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or reinstall LlamaIndex
pip install --upgrade llama-index
```

### API Key Errors

**Problem**: API key not found

**Solution**:
1. Verify .env file exists
2. Check .env is in project root
3. Verify load_dotenv() is called before accessing keys

## 📚 Next Steps

1. **[USAGE_GUIDE.md](./USAGE_GUIDE.md)** - Learn how to use LlamaIndex features
2. **[EXAMPLES.md](./EXAMPLES.md)** - See practical examples
3. **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Understand the architecture

---

**Environment set up?** Move on to the [Usage Guide](./USAGE_GUIDE.md)! 🎉
