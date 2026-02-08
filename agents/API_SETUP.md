# API Server Setup and Usage

## Installation

### Option 1: Quick Setup (For Testing)

```bash
# Navigate to agents directory
cd /home/runner/work/llama_index/llama_index/agents

# Install Python dependencies
pip install fastapi uvicorn python-multipart pydantic

# Add agents to Python path
export PYTHONPATH="${PYTHONPATH}:/home/runner/work/llama_index/llama_index/agents"
```

### Option 2: Full Installation

```bash
# Install llama-index and agent dependencies
pip install llama-index llama-index-core

# Install each agent's requirements
cd data-ingestion-agent && pip install -r requirements.txt && cd ..
cd query-engine-agent && pip install -r requirements.txt && cd ..
cd rag-pipeline-agent && pip install -r requirements.txt && cd ..
cd indexing-agent && pip install -r requirements.txt && cd ..
cd evaluation-agent && pip install -r requirements.txt && cd ..

# Install API server dependencies
pip install fastapi uvicorn python-multipart

# Add agents to Python path
export PYTHONPATH="${PYTHONPATH}:/home/runner/work/llama_index/llama_index/agents"
```

## Running the API Server

### Start the Server

```bash
cd /home/runner/work/llama_index/llama_index/agents
python api_server.py
```

Server will start at `http://localhost:8000`

### Verify Server is Running

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "running",
  "version": "1.0.0",
  "agents": {
    "data_ingestion": "available",
    "query_engine": "available",
    "rag_pipeline": "available",
    "indexing": "available",
    "evaluation": "available"
  }
}
```

## Using the Web UI

1. Ensure API server is running
2. Open `web_ui.html` in a web browser:
   ```bash
   # On Linux
   xdg-open web_ui.html
   
   # On Mac
   open web_ui.html
   
   # On Windows
   start web_ui.html
   ```
3. The UI will automatically connect to `http://localhost:8000`

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Data Ingestion
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "directory",
    "source_path": "/path/to/data",
    "chunk_size": 1024,
    "output_dir": "/tmp/index"
  }'
```

### Query Index
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "index_path": "/tmp/index",
    "mode": "similarity",
    "top_k": 3
  }'
```

### Build Pipeline
```bash
curl -X POST http://localhost:8000/api/v1/pipeline/build \
  -H "Content-Type: application/json" \
  -d '{
    "data_dir": "/path/to/data",
    "output_dir": "/tmp/pipeline"
  }'
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Troubleshooting

### ModuleNotFoundError

If you get `ModuleNotFoundError: No module named 'data_ingestion_agent'`:

```bash
# Add agents to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or modify the import in api_server.py to use relative imports
```

### Port Already in Use

If port 8000 is already in use:

```bash
# Change port in api_server.py or run with custom port
uvicorn api_server:app --host 0.0.0.0 --port 8001
```

### CORS Errors

If you get CORS errors in the web UI:
1. Ensure API server is running
2. Check the API_BASE URL in web_ui.html matches your server
3. CORS is already configured in api_server.py for all origins

## Production Deployment

### Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app/
RUN pip install -r requirements.txt
ENV PYTHONPATH=/app
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t llamaindex-agents-api .
docker run -p 8000:8000 llamaindex-agents-api
```

### Cloud Deployment

See PRODUCTION_CERTIFICATION.md for deployment options on AWS, GCP, Azure.

## Testing

Run the test suite:
```bash
# Install test dependencies
pip install -r tests/requirements.txt

# Run all tests
python run_tests.py

# Or use pytest directly
pytest tests/ -v
```

## Support

For issues or questions:
1. Check TESTING_GUIDE.md for detailed testing procedures
2. Review PRODUCTION_CERTIFICATION.md for architecture details
3. See individual agent READMEs for agent-specific documentation
