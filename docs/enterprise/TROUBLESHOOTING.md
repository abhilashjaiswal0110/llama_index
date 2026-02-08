# Troubleshooting Guide

> **📍 You are here:** Solutions to common issues when working with LlamaIndex.

## 🔍 Quick Diagnosis

### Check Your Setup

```bash
# Check Python version
python --version

# Check installed packages
pip list | grep llama

# Check API key (should not be empty)
echo $OPENAI_API_KEY

# Test import
python -c "import llama_index; print('Success!')"
```

## 🚨 Common Issues

### Installation Issues

#### Issue: `ModuleNotFoundError: No module named 'llama_index'`

**Cause**: LlamaIndex not installed or virtual environment not activated.

**Solutions**:

```bash
# Solution 1: Install LlamaIndex
pip install llama-index

# Solution 2: Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Solution 3: Reinstall in case of corruption
pip uninstall llama-index
pip install llama-index
```

#### Issue: Dependency Conflicts

**Cause**: Incompatible package versions.

**Solutions**:

```bash
# Solution 1: Use a fresh virtual environment
python -m venv new_venv
source new_venv/bin/activate
pip install llama-index

# Solution 2: Use uv for faster, better dependency resolution
pip install uv
uv pip install llama-index

# Solution 3: Install specific versions
pip install llama-index-core==0.14.13
```

#### Issue: `error: Microsoft Visual C++ 14.0 or greater is required` (Windows)

**Cause**: Missing C++ compiler for building certain dependencies.

**Solution**:

1. Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Or use pre-compiled wheels: `pip install --only-binary :all: package-name`

### API Key Issues

#### Issue: `AuthenticationError: Invalid API key`

**Cause**: API key not set, incorrect, or expired.

**Solutions**:

```bash
# Solution 1: Set environment variable
export OPENAI_API_KEY="sk-your-key-here"

# Solution 2: Use .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Solution 3: Set in Python code
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"

# Verify key is set
python -c "import os; print('Key set:', bool(os.getenv('OPENAI_API_KEY')))"
```

#### Issue: `openai.error.RateLimitError`

**Cause**: Exceeded API rate limits.

**Solutions**:

1. **Wait and retry**: Implement exponential backoff
2. **Upgrade plan**: Increase rate limits
3. **Reduce requests**: Batch queries or cache results
4. **Use different provider**: Switch to Anthropic, etc.

```python
# Example: Add retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def query_with_retry(query_engine, query):
    return query_engine.query(query)
```

### Data Loading Issues

#### Issue: `ValueError: No documents found`

**Cause**: Data directory empty or files not in supported format.

**Solutions**:

```python
# Solution 1: Check directory exists and has files
from pathlib import Path
data_dir = Path("./data")
print(f"Directory exists: {data_dir.exists()}")
print(f"Files: {list(data_dir.glob('*'))}")

# Solution 2: Specify file extensions
from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader(
    "./data",
    required_exts=[".txt", ".pdf", ".md"]
).load_data()

# Solution 3: Use specific readers
from llama_index.readers.file import PDFReader
reader = PDFReader()
documents = reader.load_data(file=Path("./data/file.pdf"))
```

#### Issue: PDF Loading Fails

**Cause**: Missing PDF parsing libraries.

**Solutions**:

```bash
# Install PDF dependencies
pip install pypdf
# or
pip install pdfminer.six

# For better results with complex PDFs
pip install llama-index-readers-llama-parse
```

### Indexing Issues

#### Issue: `OutOfMemoryError` during indexing

**Cause**: Dataset too large for available RAM.

**Solutions**:

```python
# Solution 1: Use smaller chunks
from llama_index.core import Settings
Settings.chunk_size = 512  # Reduce from default 1024

# Solution 2: Process in batches
from llama_index.core import VectorStoreIndex

def index_in_batches(documents, batch_size=100):
    index = None
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        if index is None:
            index = VectorStoreIndex.from_documents(batch)
        else:
            index.insert_nodes(batch)
    return index

# Solution 3: Use a persistent vector store
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

chroma_client = chromadb.PersistentClient(path="./storage/chroma")
chroma_collection = chroma_client.create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
```

#### Issue: Slow indexing

**Cause**: Using API-based embeddings or large dataset.

**Solutions**:

```python
# Solution 1: Use local embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Solution 2: Enable batch processing
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    embed_batch_size=10
)

# Solution 3: Use GPU if available
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cuda"  # or "mps" for Mac
)
```

### Query Issues

#### Issue: Poor quality responses

**Cause**: Various factors affecting retrieval or generation.

**Solutions**:

```python
# Solution 1: Adjust similarity top_k
query_engine = index.as_query_engine(similarity_top_k=5)

# Solution 2: Use different retrieval mode
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(
    response_mode="tree_summarize"  # or "compact", "refine"
)

# Solution 3: Add custom prompts
from llama_index.core import PromptTemplate

qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

query_engine = index.as_query_engine(text_qa_template=qa_prompt)

# Solution 4: Enable verbose mode to debug
query_engine = index.as_query_engine(verbose=True)
```

#### Issue: Hallucinated or incorrect answers

**Cause**: LLM generating information not in the documents.

**Solutions**:

```python
# Solution 1: Strengthen system prompt
from llama_index.core import PromptTemplate

strict_qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "You must answer based ONLY on the context above. "
    "If the answer is not in the context, say 'I don't have enough information to answer this.'\n"
    "Query: {query_str}\n"
    "Answer: "
)

# Solution 2: Increase context relevance
query_engine = index.as_query_engine(
    similarity_top_k=10,  # Get more context
    similarity_cutoff=0.7  # Filter low-relevance chunks
)

# Solution 3: Use citation mode
from llama_index.core.response_synthesizers import ResponseMode
query_engine = index.as_query_engine(
    response_mode=ResponseMode.COMPACT,
    verbose=True
)
```

### Storage Issues

#### Issue: `PermissionError` when persisting

**Cause**: Insufficient permissions to write to storage directory.

**Solutions**:

```bash
# Solution 1: Check permissions
ls -la storage/

# Solution 2: Create directory with proper permissions
mkdir -p storage
chmod 755 storage

# Solution 3: Use different directory
```

```python
# In Python
import os
storage_dir = "./storage"
os.makedirs(storage_dir, exist_ok=True)
index.storage_context.persist(persist_dir=storage_dir)
```

#### Issue: Cannot load persisted index

**Cause**: Index format changed or corruption.

**Solutions**:

```python
# Solution 1: Verify storage directory structure
from pathlib import Path
storage_path = Path("./storage")
print("Files:", list(storage_path.glob("*")))
# Should see: default__vector_store.json, docstore.json, index_store.json

# Solution 2: Rebuild index if corrupted
# Delete storage and recreate
import shutil
shutil.rmtree("./storage", ignore_errors=True)
# Then rebuild from source documents

# Solution 3: Use version-specific loading
from llama_index.core import load_index_from_storage, StorageContext
try:
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
except Exception as e:
    print(f"Error loading index: {e}")
    # Rebuild index
```

### Performance Issues

#### Issue: High memory usage

**Solutions**:

```python
# Solution 1: Clear cache periodically
import gc
gc.collect()

# Solution 2: Use streaming for large responses
query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("your query")
for text in response.response_gen:
    print(text, end="", flush=True)

# Solution 3: Limit node size
from llama_index.core import Settings
Settings.chunk_size = 512
Settings.chunk_overlap = 50
```

#### Issue: Slow query response time

**Solutions**:

```python
# Solution 1: Reduce similarity_top_k
query_engine = index.as_query_engine(similarity_top_k=3)

# Solution 2: Use faster LLM
from llama_index.llms.openai import OpenAI
Settings.llm = OpenAI(model="gpt-3.5-turbo")  # Faster than GPT-4

# Solution 3: Implement caching
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_query(query_str):
    return query_engine.query(query_str)
```

### Integration Issues

#### Issue: Vector store connection problems

**Solutions**:

```python
# Solution 1: Verify vector store is running
# For Chroma:
import chromadb
try:
    client = chromadb.HttpClient(host="localhost", port=8000)
    client.heartbeat()
    print("Chroma is running")
except Exception as e:
    print(f"Chroma not accessible: {e}")

# Solution 2: Use persistent client instead
client = chromadb.PersistentClient(path="./chroma_db")

# Solution 3: Check firewall/network settings
```

#### Issue: Custom reader not working

**Solutions**:

```python
# Solution 1: Verify reader implementation
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

class CustomReader(BaseReader):
    def load_data(self, file_path):
        # Ensure returns List[Document]
        with open(file_path) as f:
            text = f.read()
        return [Document(text=text)]

# Solution 2: Check reader registration
from llama_index.core import SimpleDirectoryReader
SimpleDirectoryReader.supported_suffix.add(".custom")

# Solution 3: Use file_extractor parameter
reader = SimpleDirectoryReader(
    "./data",
    file_extractor={".custom": CustomReader()}
)
```

## 🔧 Debugging Tips

### Enable Debug Logging

```python
import logging
import sys

# Set up detailed logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable LlamaIndex debug logging
logging.getLogger('llama_index').setLevel(logging.DEBUG)
```

### Inspect Retrieved Context

```python
# Get detailed response with source nodes
query_engine = index.as_query_engine(response_mode="compact")
response = query_engine.query("your query")

# Inspect source nodes
print("Number of source nodes:", len(response.source_nodes))
for i, node in enumerate(response.source_nodes):
    print(f"\nNode {i}:")
    print(f"Score: {node.score}")
    print(f"Text: {node.text[:200]}...")  # First 200 chars
```

### Test Individual Components

```python
# Test document loading
from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader("./data").load_data()
print(f"Loaded {len(documents)} documents")

# Test embeddings
from llama_index.core import Settings
embed_model = Settings.embed_model
test_embedding = embed_model.get_text_embedding("test text")
print(f"Embedding dimension: {len(test_embedding)}")

# Test LLM
from llama_index.core import Settings
llm = Settings.llm
response = llm.complete("Say hello")
print(f"LLM response: {response}")
```

## 📚 Additional Resources

- **[LlamaIndex Documentation](https://docs.llamaindex.ai/)** - Official docs
- **[Discord Community](https://discord.gg/dGcwcsnxhU)** - Get help from community
- **[GitHub Issues](https://github.com/run-llama/llama_index/issues)** - Report bugs
- **[Stack Overflow](https://stackoverflow.com/questions/tagged/llama-index)** - Q&A

## 🆘 Still Having Issues?

If you can't find a solution:

1. **Search GitHub Issues**: Someone may have had the same problem
2. **Ask on Discord**: Community is very responsive
3. **Create a minimal reproducible example**: Helps others help you
4. **Check version compatibility**: Ensure all packages are compatible

---

**Issue resolved?** Return to your [local setup](./LOCAL_SETUP.md) or continue with [usage examples](./EXAMPLES.md)!
