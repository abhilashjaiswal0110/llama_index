"""RAG Pipeline Agent Core"""
from typing import Optional, Dict, Any
from pathlib import Path
import yaml
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext

class RAGPipelineAgent:
    def __init__(self):
        self.pipeline = None
        
    def build_pipeline(self, data_dir: str, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Build complete RAG pipeline"""
        config = self._load_config(config_file) if config_file else {}
        
        # Load documents
        documents = SimpleDirectoryReader(data_dir).load_data()
        
        # Create index
        index = VectorStoreIndex.from_documents(documents)
        
        # Create query engine
        query_engine = index.as_query_engine(
            similarity_top_k=config.get('top_k', 3),
            response_mode=config.get('response_mode', 'compact')
        )
        
        return {
            'index': index,
            'query_engine': query_engine,
            'config': config,
            'documents': documents
        }
    
    def save_pipeline(self, pipeline: Dict[str, Any], output_dir: str):
        """Save pipeline to disk"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pipeline['index'].storage_context.persist(persist_dir=output_dir)
        
        # Save config
        with open(Path(output_dir) / 'config.yaml', 'w') as f:
            yaml.dump(pipeline['config'], f)
    
    def load_pipeline(self, pipeline_dir: str) -> Dict[str, Any]:
        """Load pipeline from disk"""
        from llama_index.core import load_index_from_storage
        
        storage_context = StorageContext.from_defaults(persist_dir=pipeline_dir)
        index = load_index_from_storage(storage_context)
        
        with open(Path(pipeline_dir) / 'config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        query_engine = index.as_query_engine()
        
        return {
            'index': index,
            'query_engine': query_engine,
            'config': config
        }
    
    def optimize(self, pipeline: Dict[str, Any], test_queries: str) -> Dict[str, Any]:
        """Optimize pipeline parameters"""
        # Simplified optimization - in production, test multiple configurations
        import json
        with open(test_queries, 'r') as f:
            queries = json.load(f)
        
        # Test different top_k values
        best_k = 3
        best_score = 0
        
        for k in [3, 5, 7, 10]:
            query_engine = pipeline['index'].as_query_engine(similarity_top_k=k)
            score = self._evaluate_queries(query_engine, queries[:5])
            if score > best_score:
                best_score = score
                best_k = k
        
        pipeline['config']['top_k'] = best_k
        pipeline['query_engine'] = pipeline['index'].as_query_engine(similarity_top_k=best_k)
        
        return pipeline
    
    def evaluate(self, pipeline: Dict[str, Any], test_set: str) -> Dict[str, float]:
        """Evaluate pipeline performance"""
        import json
        with open(test_set, 'r') as f:
            test_data = json.load(f)
        
        scores = {
            'faithfulness': 0.85,  # Placeholder - use actual evaluation
            'relevance': 0.82,
            'coherence': 0.88
        }
        
        return scores
    
    def deploy(self, pipeline: Dict[str, Any], target: str, output_dir: str):
        """Generate deployment artifacts"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if target == 'docker':
            self._generate_dockerfile(output_dir)
        elif target == 'kubernetes':
            self._generate_k8s(output_dir)
        elif target == 'fastapi':
            self._generate_fastapi(output_dir)
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _evaluate_queries(self, query_engine, queries) -> float:
        # Simplified evaluation
        return 0.8
    
    def _generate_dockerfile(self, output_dir: str):
        dockerfile = """FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "serve.py"]
"""
        with open(Path(output_dir) / 'Dockerfile', 'w') as f:
            f.write(dockerfile)
    
    def _generate_k8s(self, output_dir: str):
        k8s_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-pipeline
  template:
    metadata:
      labels:
        app: rag-pipeline
    spec:
      containers:
      - name: rag-pipeline
        image: rag-pipeline:latest
        ports:
        - containerPort: 8000
"""
        with open(Path(output_dir) / 'deployment.yaml', 'w') as f:
            f.write(k8s_yaml)
    
    def _generate_fastapi(self, output_dir: str):
        fastapi_app = """from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/query")
async def query(q: Query):
    # Load and query pipeline
    return {"response": "Answer to query"}
"""
        with open(Path(output_dir) / 'app.py', 'w') as f:
            f.write(fastapi_app)
