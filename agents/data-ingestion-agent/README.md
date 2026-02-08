# Data Ingestion Agent

AI-powered data loading and processing agent for LlamaIndex. Handles multiple data sources, formats, and provides intelligent chunking, metadata extraction, and document preprocessing.

## Features

- 📄 **Multi-format support:** PDF, DOCX, TXT, MD, HTML, JSON, CSV
- 🌐 **Web scraping:** Intelligent web content extraction
- 🔌 **API integration:** REST API data ingestion
- 💾 **Database connectors:** SQL and NoSQL support
- 🧹 **Smart preprocessing:** Deduplication, cleaning, normalization
- 📊 **Metadata extraction:** Automatic metadata enrichment
- ⚡ **Batch processing:** Efficient directory scanning
- 🔄 **Error recovery:** Retry logic and checkpoint support

## Modes

| Mode | Description | Input | Output |
|------|-------------|-------|--------|
| `pdf` | Process PDF documents | PDF file(s) | Indexed documents |
| `web` | Scrape web content | URL(s) | Web pages as documents |
| `api` | Ingest from REST API | API endpoint | API data as documents |
| `database` | Connect to database | Connection string | Database records |
| `directory` | Batch process directory | Directory path | All supported files |
| `csv` | Process CSV/Excel | CSV/XLSX file | Structured data |
| `json` | Process JSON data | JSON file(s) | JSON records |

## Installation

```bash
# Navigate to agent directory
cd data-ingestion-agent

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Required: OPENAI_API_KEY or other LLM provider key
# Optional: Database credentials, API keys
```

## Quick Start

```bash
# Process a single PDF
python main.py pdf --path document.pdf

# Process entire directory
python main.py directory --path ./docs --recursive

# Scrape web content
python main.py web --url https://example.com --depth 2

# Process with custom chunking
python main.py pdf --path doc.pdf --chunk-size 512 --chunk-overlap 50

# Save to specific index
python main.py directory --path ./docs --output-dir ./index --index-name my_docs
```

## Usage Examples

### Basic PDF Processing
```bash
python main.py \
  --mode pdf \
  --path ./documents/handbook.pdf \
  --chunk-size 1024 \
  --chunk-overlap 200
```

### Web Scraping
```bash
python main.py \
  --mode web \
  --url https://docs.example.com \
  --depth 3 \
  --max-pages 100 \
  --include-pattern "*/docs/*"
```

### Database Ingestion
```bash
python main.py \
  --mode database \
  --connection "postgresql://user:pass@localhost/dbname" \
  --query "SELECT * FROM documents WHERE active=true" \
  --text-columns "title,content" \
  --metadata-columns "id,created_at,author"
```

### API Ingestion
```bash
python main.py \
  --mode api \
  --endpoint https://api.example.com/v1/articles \
  --headers "Authorization: Bearer TOKEN" \
  --text-field "content" \
  --metadata-fields "id,title,date"
```

### Directory Processing with Filters
```bash
python main.py \
  --mode directory \
  --path ./knowledge_base \
  --recursive \
  --extensions "pdf,docx,md" \
  --exclude "archive,old" \
  --output-dir ./index
```

## Configuration

### Command Line Options

```
General Options:
  --mode              Ingestion mode (pdf, web, api, database, directory)
  --path              File or directory path
  --output-dir        Output directory for index (default: ./index)
  --index-name        Name for the index (default: auto-generated)
  --verbose           Enable verbose logging
  --log-file          Log file path (default: ingestion.log)

Chunking Options:
  --chunk-size        Chunk size in tokens (default: 1024)
  --chunk-overlap     Overlap between chunks (default: 200)
  --chunking-strategy Strategy: fixed, sentence, semantic (default: sentence)

Processing Options:
  --clean-text        Remove extra whitespace and formatting
  --extract-metadata  Extract metadata from documents (default: true)
  --deduplicate       Remove duplicate chunks (default: false)
  --min-chunk-size    Minimum chunk size to keep (default: 100)

Mode-Specific Options:
  PDF:
    --extract-images    Extract and describe images
    --ocr               Use OCR for scanned PDFs
    
  Web:
    --url               Starting URL
    --depth             Crawl depth (default: 1)
    --max-pages         Maximum pages to crawl (default: 50)
    --include-pattern   URL pattern to include
    --exclude-pattern   URL pattern to exclude
    
  API:
    --endpoint          API endpoint URL
    --headers           HTTP headers (JSON format)
    --params            Query parameters (JSON format)
    --text-field        Field containing text content
    --metadata-fields   Comma-separated metadata fields
    
  Database:
    --connection        Database connection string
    --query             SQL query to execute
    --text-columns      Columns containing text (comma-separated)
    --metadata-columns  Columns for metadata (comma-separated)
    
  Directory:
    --recursive         Recurse into subdirectories
    --extensions        File extensions to include (comma-separated)
    --exclude           Patterns to exclude
```

### Configuration File (config.yaml)

```yaml
# Data Ingestion Agent Configuration

# General Settings
general:
  output_dir: "./index"
  log_level: "INFO"
  log_file: "ingestion.log"
  
# Chunking Configuration
chunking:
  chunk_size: 1024
  chunk_overlap: 200
  strategy: "sentence"  # fixed, sentence, semantic
  min_chunk_size: 100
  
# Text Processing
processing:
  clean_text: true
  extract_metadata: true
  deduplicate: false
  normalize_whitespace: true
  remove_special_chars: false
  
# PDF Settings
pdf:
  extract_images: false
  ocr_enabled: false
  extract_tables: true
  
# Web Scraping Settings
web:
  depth: 1
  max_pages: 50
  timeout: 30
  user_agent: "LlamaIndex-DataIngestion/1.0"
  respect_robots_txt: true
  
# API Settings
api:
  timeout: 30
  retry_attempts: 3
  retry_delay: 5
  pagination: true
  
# Database Settings
database:
  batch_size: 1000
  timeout: 300
  
# Performance
performance:
  batch_size: 32
  num_workers: 4
  show_progress: true
```

## Python API

### Basic Usage

```python
from data_ingestion_agent import DataIngestionAgent

# Initialize agent
agent = DataIngestionAgent(
    chunk_size=1024,
    chunk_overlap=200,
    output_dir="./index"
)

# Ingest PDF
documents = agent.ingest_pdf("document.pdf")

# Ingest directory
documents = agent.ingest_directory(
    "./docs",
    recursive=True,
    extensions=["pdf", "md", "txt"]
)

# Create index
index = agent.create_index(documents, index_name="my_docs")

# Save index
agent.save_index(index, "./index")
```

### Advanced Usage

```python
from data_ingestion_agent import DataIngestionAgent, ChunkingConfig

# Custom configuration
config = ChunkingConfig(
    chunk_size=512,
    chunk_overlap=50,
    strategy="semantic",
    min_chunk_size=100
)

agent = DataIngestionAgent(config=config)

# Web scraping with filters
documents = agent.ingest_web(
    url="https://docs.example.com",
    depth=2,
    max_pages=100,
    include_pattern="*/docs/*",
    exclude_pattern="*/old/*"
)

# API ingestion
documents = agent.ingest_api(
    endpoint="https://api.example.com/articles",
    headers={"Authorization": "Bearer TOKEN"},
    text_field="content",
    metadata_fields=["id", "title", "date"],
    pagination=True
)

# Database ingestion
documents = agent.ingest_database(
    connection="postgresql://localhost/mydb",
    query="SELECT * FROM articles WHERE active=true",
    text_columns=["title", "content"],
    metadata_columns=["id", "created_at", "author"]
)

# Process with callbacks
def on_document_processed(doc):
    print(f"Processed: {doc.metadata.get('filename')}")

agent.on_document_processed = on_document_processed
documents = agent.ingest_directory("./docs")
```

### Custom Loaders

```python
from llama_index.core import Document
from data_ingestion_agent import DataIngestionAgent

agent = DataIngestionAgent()

# Register custom loader
@agent.register_loader("custom")
def custom_loader(file_path: str) -> list[Document]:
    # Your custom loading logic
    content = load_custom_format(file_path)
    return [Document(text=content, metadata={"source": file_path})]

# Use custom loader
documents = agent.ingest(
    path="data.custom",
    loader_type="custom"
)
```

## Pipeline Integration

### With Query Engine

```python
from data_ingestion_agent import DataIngestionAgent
from llama_index.core import VectorStoreIndex

# Ingest data
agent = DataIngestionAgent()
documents = agent.ingest_directory("./docs")

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is RAG?")
```

### With Custom Processing

```python
from data_ingestion_agent import DataIngestionAgent

agent = DataIngestionAgent()

# Add preprocessing step
def preprocess(doc):
    # Custom preprocessing
    doc.text = doc.text.upper()  # Example
    return doc

agent.add_preprocessor(preprocess)

documents = agent.ingest_pdf("document.pdf")
```

## Examples

See the `examples/` directory for more usage examples:

- `basic_usage.py` - Simple PDF and directory processing
- `web_scraping.py` - Advanced web scraping
- `api_ingestion.py` - REST API integration
- `database_ingestion.py` - Database connections
- `custom_loaders.py` - Creating custom loaders
- `pipeline_integration.py` - Integration with RAG pipelines

## Best Practices

### Chunking Strategy Selection

- **Fixed-size:** Simple, predictable, good for uniform content
- **Sentence-based:** Better semantic boundaries, preserves meaning
- **Semantic:** Best quality, but slower, uses embeddings

```python
# For technical documentation
agent = DataIngestionAgent(
    chunk_size=1024,
    chunking_strategy="sentence"
)

# For code or structured data
agent = DataIngestionAgent(
    chunk_size=512,
    chunking_strategy="fixed"
)

# For maximum quality (slower)
agent = DataIngestionAgent(
    chunk_size=256,
    chunking_strategy="semantic"
)
```

### Metadata Enrichment

```python
# Extract and enrich metadata
agent = DataIngestionAgent(extract_metadata=True)

# Add custom metadata
def add_custom_metadata(doc):
    doc.metadata["processed_date"] = datetime.now()
    doc.metadata["version"] = "1.0"
    return doc

agent.add_preprocessor(add_custom_metadata)
```

### Error Handling

```python
# Enable checkpointing for large ingestions
agent = DataIngestionAgent(
    checkpoint_enabled=True,
    checkpoint_file="ingestion.checkpoint"
)

# Resume from checkpoint
if agent.has_checkpoint():
    agent.resume_from_checkpoint()
```

### Performance Optimization

```python
# Use batch processing
agent = DataIngestionAgent(
    batch_size=32,
    num_workers=4  # Parallel processing
)

# Disable unnecessary features
agent = DataIngestionAgent(
    extract_metadata=False,  # Skip if not needed
    deduplicate=False,       # Skip for unique content
)
```

## Troubleshooting

### Common Issues

**Issue:** Out of memory with large PDFs
```python
# Solution: Use streaming mode
agent = DataIngestionAgent(streaming=True, batch_size=16)
```

**Issue:** Slow processing
```python
# Solution: Increase parallelism
agent = DataIngestionAgent(num_workers=8)
```

**Issue:** Web scraping blocked
```python
# Solution: Add delays and custom user agent
agent = DataIngestionAgent()
documents = agent.ingest_web(
    url="https://example.com",
    delay=2,  # seconds between requests
    user_agent="Mozilla/5.0..."
)
```

## Logging

The agent provides detailed logging:

```python
import logging

# Set log level
logging.basicConfig(level=logging.INFO)

# View progress
agent = DataIngestionAgent(verbose=True, show_progress=True)
```

Log output includes:
- Document processing status
- Chunking statistics
- Error messages and warnings
- Performance metrics

## Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_pdf_ingestion.py

# Run with coverage
pytest --cov=data_ingestion_agent tests/
```

## Performance Metrics

The agent tracks:
- Documents processed
- Total chunks created
- Processing time
- Average chunk size
- Memory usage

Access metrics:
```python
agent = DataIngestionAgent()
documents = agent.ingest_directory("./docs")

metrics = agent.get_metrics()
print(f"Processed: {metrics['documents_processed']}")
print(f"Chunks: {metrics['total_chunks']}")
print(f"Time: {metrics['processing_time']:.2f}s")
```

## License

Part of the LlamaIndex project. See main repository for license details.

## Support

- Issues: Report in main LlamaIndex repository
- Documentation: See examples/ directory
- Discussions: GitHub Discussions

## Version

Current version: 1.0.0
