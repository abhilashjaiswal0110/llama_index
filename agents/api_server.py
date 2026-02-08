"""
FastAPI Server for LlamaIndex Specialized Agents
Provides REST API endpoints for all 5 agents
"""
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
import tempfile
import sys
import os
import json
import logging

# Add agents to path
sys.path.insert(0, str(Path(__file__).parent))

from data_ingestion_agent.agent import DataIngestionAgent, IngestionConfig
from query_engine_agent.agent import QueryEngineAgent, QueryConfig
from rag_pipeline_agent.agent import RAGPipelineAgent
from indexing_agent.agent import IndexingAgent
from evaluation_agent.agent import EvaluationAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LlamaIndex Agents API",
    description="REST API for LlamaIndex Specialized Agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class IngestionRequest(BaseModel):
    """Request model for data ingestion"""
    source_type: str = Field(..., description="Type of source: directory, pdf, web, api, database")
    source_path: str = Field(..., description="Path or URL to the data source")
    chunk_size: int = Field(default=1024, ge=128, le=4096)
    chunk_overlap: int = Field(default=200, ge=0, le=1024)
    chunking_strategy: str = Field(default="fixed", pattern="^(fixed|sentence|semantic)$")
    recursive: bool = Field(default=False, description="Recursively process directories")
    output_dir: Optional[str] = Field(default=None, description="Output directory for index")


class QueryRequest(BaseModel):
    """Request model for query engine"""
    query: str = Field(..., min_length=1, max_length=2000)
    index_path: str = Field(..., description="Path to the index")
    mode: str = Field(default="similarity", pattern="^(similarity|hybrid|sub-question|fusion|tree|router)$")
    top_k: int = Field(default=3, ge=1, le=50)
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stream: bool = Field(default=False, description="Stream the response")


class PipelineBuildRequest(BaseModel):
    """Request model for building RAG pipeline"""
    data_dir: str = Field(..., description="Directory containing data")
    config_file: Optional[str] = Field(default=None)
    output_dir: str = Field(..., description="Output directory for pipeline")


class PipelineOptimizeRequest(BaseModel):
    """Request model for pipeline optimization"""
    pipeline_path: str = Field(..., description="Path to existing pipeline")
    test_queries_file: str = Field(..., description="Path to test queries JSON file")


class EvaluationRequest(BaseModel):
    """Request model for evaluation"""
    pipeline_path: str = Field(..., description="Path to pipeline to evaluate")
    test_queries: List[Dict[str, str]] = Field(..., description="List of test queries")
    metrics: List[str] = Field(default=["faithfulness", "relevance"], description="Metrics to evaluate")


class IndexCreationRequest(BaseModel):
    """Request model for index creation"""
    data_dir: str = Field(..., description="Directory containing data")
    index_type: str = Field(default="vector", pattern="^(vector|tree|keyword|list|document-summary|knowledge-graph)$")
    output_dir: str = Field(..., description="Output directory for index")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    agents: Dict[str, str]


# Global agent instances
ingestion_agent = None
rag_pipeline_agent = None
indexing_agent = None
evaluation_agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup"""
    global ingestion_agent, rag_pipeline_agent, indexing_agent, evaluation_agent
    
    logger.info("Initializing agents...")
    try:
        ingestion_agent = DataIngestionAgent()
        rag_pipeline_agent = RAGPipelineAgent()
        indexing_agent = IndexingAgent()
        evaluation_agent = EvaluationAgent()
        logger.info("All agents initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API information"""
    return {
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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return await root()


# ==================== Data Ingestion Endpoints ====================

@app.post("/api/v1/ingest", tags=["Data Ingestion"])
async def ingest_data(request: IngestionRequest):
    """
    Ingest data from various sources
    
    Supports: directory, pdf, web, api, database
    """
    try:
        logger.info(f"Ingesting data from {request.source_type}: {request.source_path}")
        
        # Configure ingestion
        config = IngestionConfig(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            chunking_strategy=request.chunking_strategy,
            output_dir=request.output_dir
        )
        agent = DataIngestionAgent(config=config)
        
        # Ingest based on source type
        if request.source_type == "directory":
            documents = agent.ingest_directory(request.source_path, recursive=request.recursive)
        elif request.source_type == "pdf":
            documents = agent.ingest_pdf(request.source_path)
        elif request.source_type == "web":
            documents = agent.ingest_web(request.source_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source type: {request.source_type}")
        
        # Create index if output_dir specified
        index_path = None
        if request.output_dir:
            index = agent.create_index(documents)
            agent.save_index(index, request.output_dir)
            index_path = request.output_dir
        
        # Get metrics
        metrics = agent.get_metrics()
        
        return {
            "status": "success",
            "documents_processed": len(documents),
            "index_path": index_path,
            "metrics": metrics
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Source not found: {str(e)}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/api/v1/ingest/metrics", tags=["Data Ingestion"])
async def get_ingestion_metrics():
    """Get metrics from last ingestion operation"""
    try:
        metrics = ingestion_agent.get_metrics()
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Query Engine Endpoints ====================

@app.post("/api/v1/query", tags=["Query Engine"])
async def query_index(request: QueryRequest):
    """
    Query an existing index
    
    Supports multiple query modes and streaming
    """
    try:
        logger.info(f"Querying index: {request.index_path}")
        
        # Check if index exists
        if not Path(request.index_path).exists():
            raise HTTPException(status_code=404, detail=f"Index not found: {request.index_path}")
        
        # Configure query
        config = QueryConfig(
            mode=request.mode,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        # Create query engine
        query_agent = QueryEngineAgent(index_path=request.index_path, config=config)
        
        # Execute query
        if request.stream:
            # Streaming response
            def generate():
                for chunk in query_agent.query_stream(request.query, config=config):
                    yield json.dumps({"chunk": str(chunk)}) + "\n"
            
            return StreamingResponse(generate(), media_type="application/x-ndjson")
        else:
            # Regular response
            response = query_agent.query(request.query, config=config)
            return {
                "status": "success",
                "query": request.query,
                "response": str(response),
                "source_nodes": [{"text": node.get_content()[:200]} for node in response.source_nodes] if hasattr(response, 'source_nodes') else []
            }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ==================== RAG Pipeline Endpoints ====================

@app.post("/api/v1/pipeline/build", tags=["RAG Pipeline"])
async def build_pipeline(request: PipelineBuildRequest, background_tasks: BackgroundTasks):
    """Build a complete RAG pipeline"""
    try:
        logger.info(f"Building pipeline from {request.data_dir}")
        
        # Check if data directory exists
        if not Path(request.data_dir).exists():
            raise HTTPException(status_code=404, detail=f"Data directory not found: {request.data_dir}")
        
        # Build pipeline
        pipeline = rag_pipeline_agent.build_pipeline(
            data_dir=request.data_dir,
            config_file=request.config_file
        )
        
        # Save pipeline
        rag_pipeline_agent.save_pipeline(pipeline, request.output_dir)
        
        return {
            "status": "success",
            "pipeline_path": request.output_dir,
            "documents_count": len(pipeline.get('documents', [])),
            "config": pipeline.get('config', {})
        }
        
    except Exception as e:
        logger.error(f"Pipeline build failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline build failed: {str(e)}")


@app.post("/api/v1/pipeline/optimize", tags=["RAG Pipeline"])
async def optimize_pipeline(request: PipelineOptimizeRequest):
    """Optimize an existing pipeline"""
    try:
        logger.info(f"Optimizing pipeline: {request.pipeline_path}")
        
        # Load pipeline
        pipeline = rag_pipeline_agent.load_pipeline(request.pipeline_path)
        
        # Optimize
        optimized = rag_pipeline_agent.optimize(pipeline, request.test_queries_file)
        
        # Save optimized pipeline
        rag_pipeline_agent.save_pipeline(optimized, request.pipeline_path)
        
        return {
            "status": "success",
            "pipeline_path": request.pipeline_path,
            "optimized_config": optimized.get('config', {})
        }
        
    except Exception as e:
        logger.error(f"Pipeline optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.post("/api/v1/pipeline/evaluate", tags=["RAG Pipeline"])
async def evaluate_pipeline(request: EvaluationRequest):
    """Evaluate a pipeline"""
    try:
        logger.info(f"Evaluating pipeline: {request.pipeline_path}")
        
        # Load pipeline
        pipeline = rag_pipeline_agent.load_pipeline(request.pipeline_path)
        
        # Evaluate
        query_engine = pipeline['query_engine']
        results = evaluation_agent.evaluate(
            query_engine=query_engine,
            test_queries=request.test_queries,
            metrics=request.metrics
        )
        
        return {
            "status": "success",
            "pipeline_path": request.pipeline_path,
            "metrics": results
        }
        
    except Exception as e:
        logger.error(f"Pipeline evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# ==================== Indexing Endpoints ====================

@app.post("/api/v1/index/create", tags=["Indexing"])
async def create_index(request: IndexCreationRequest):
    """Create an optimized index"""
    try:
        logger.info(f"Creating {request.index_type} index from {request.data_dir}")
        
        # Check if data directory exists
        if not Path(request.data_dir).exists():
            raise HTTPException(status_code=404, detail=f"Data directory not found: {request.data_dir}")
        
        # Create index
        index = indexing_agent.create_index(
            data_dir=request.data_dir,
            index_type=request.index_type
        )
        
        # Save index
        indexing_agent.save_index(index, request.output_dir)
        
        return {
            "status": "success",
            "index_type": request.index_type,
            "index_path": request.output_dir
        }
        
    except Exception as e:
        logger.error(f"Index creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Index creation failed: {str(e)}")


@app.get("/api/v1/index/recommend", tags=["Indexing"])
async def recommend_index_type(data_dir: str):
    """Get index type recommendation"""
    try:
        recommendation = indexing_agent.recommend_index_type(data_dir)
        return {
            "status": "success",
            "recommendation": recommendation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Evaluation Endpoints ====================

@app.post("/api/v1/evaluate", tags=["Evaluation"])
async def evaluate_system(request: EvaluationRequest):
    """Evaluate a RAG system"""
    try:
        logger.info(f"Evaluating system at {request.pipeline_path}")
        
        # Load pipeline
        pipeline = rag_pipeline_agent.load_pipeline(request.pipeline_path)
        query_engine = pipeline['query_engine']
        
        # Evaluate
        results = evaluation_agent.evaluate(
            query_engine=query_engine,
            test_queries=request.test_queries,
            metrics=request.metrics
        )
        
        return {
            "status": "success",
            "metrics": results,
            "test_queries_count": len(request.test_queries)
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/api/v1/evaluate/generate-tests", tags=["Evaluation"])
async def generate_test_queries(data_dir: str, num_queries: int = 10):
    """Generate test queries from data"""
    try:
        queries = evaluation_agent.generate_test_queries(data_dir, num_queries)
        return {
            "status": "success",
            "queries": queries,
            "count": len(queries)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "Resource not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
