"""Query Engine Agent Core Implementation"""
from typing import Optional
from pydantic import BaseModel, Field
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

class QueryConfig(BaseModel):
    mode: str = Field(default="similarity")
    top_k: int = Field(default=3)
    similarity_threshold: float = Field(default=0.7)
    streaming: bool = Field(default=False)
    temperature: float = Field(default=0.1)

class QueryEngineAgent:
    def __init__(self, index_path: str, config: Optional[QueryConfig] = None):
        self.config = config or QueryConfig()
        self.index = self._load_index(index_path)
        self.query_engine = None
    
    def _load_index(self, path: str) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(persist_dir=path)
        return load_index_from_storage(storage_context)
    
    def query(self, query_text: str, config: Optional[QueryConfig] = None):
        cfg = config or self.config
        
        if cfg.mode == "similarity":
            query_engine = self.index.as_query_engine(
                similarity_top_k=cfg.top_k,
                response_mode="compact"
            )
        elif cfg.mode == "hybrid":
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.core.retrievers import VectorIndexRetriever
            retriever = VectorIndexRetriever(index=self.index, similarity_top_k=cfg.top_k)
            query_engine = RetrieverQueryEngine(retriever=retriever)
        elif cfg.mode == "sub-question":
            from llama_index.core.tools import QueryEngineTool, ToolMetadata
            
            # Wrap query engine in QueryEngineTool for SubQuestionQueryEngine
            base_query_engine = self.index.as_query_engine(similarity_top_k=cfg.top_k)
            query_engine_tool = QueryEngineTool(
                query_engine=base_query_engine,
                metadata=ToolMetadata(
                    name="base_query_engine",
                    description="Query engine for the indexed documents"
                )
            )
            query_engine = SubQuestionQueryEngine.from_defaults(
                query_engine_tools=[query_engine_tool]
            )
        else:
            query_engine = self.index.as_query_engine(similarity_top_k=cfg.top_k)
        
        return query_engine.query(query_text)
    
    def query_stream(self, query_text: str, config: Optional[QueryConfig] = None):
        cfg = config or self.config
        query_engine = self.index.as_query_engine(
            similarity_top_k=cfg.top_k,
            streaming=True
        )
        response = query_engine.query(query_text)
        return response.response_gen
