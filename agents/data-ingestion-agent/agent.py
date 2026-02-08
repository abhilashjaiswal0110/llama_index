"""
Data Ingestion Agent Core Implementation

Enterprise-grade data ingestion with support for multiple sources.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time

from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    TokenTextSplitter
)
from llama_index.readers.file import PDFReader
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class IngestionConfig(BaseModel):
    """Configuration for data ingestion"""
    
    # General settings
    output_dir: str = Field(default="./index", description="Output directory for index")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Chunking settings
    chunk_size: int = Field(default=1024, description="Chunk size in tokens")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    chunking_strategy: str = Field(default="sentence", description="Chunking strategy")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size")
    
    # Processing settings
    clean_text: bool = Field(default=True, description="Clean text during processing")
    extract_metadata: bool = Field(default=True, description="Extract metadata")
    deduplicate: bool = Field(default=False, description="Remove duplicate chunks")
    
    # Performance settings
    batch_size: int = Field(default=32, description="Batch size for processing")
    num_workers: int = Field(default=4, description="Number of parallel workers")
    show_progress: bool = Field(default=True, description="Show progress bar")
    
    # PDF settings
    extract_images: bool = Field(default=False, description="Extract images from PDFs")
    ocr_enabled: bool = Field(default=False, description="Use OCR for scanned PDFs")
    
    # Web settings
    web_depth: int = Field(default=1, description="Web crawl depth")
    web_max_pages: int = Field(default=50, description="Maximum pages to crawl")
    web_timeout: int = Field(default=30, description="Web request timeout")
    
    # API settings
    api_timeout: int = Field(default=30, description="API request timeout")
    api_retry_attempts: int = Field(default=3, description="API retry attempts")
    
    # Database settings
    db_batch_size: int = Field(default=1000, description="Database batch size")
    db_timeout: int = Field(default=300, description="Database timeout")


class DataIngestionAgent:
    """
    Enterprise-grade data ingestion agent for LlamaIndex.
    
    Supports multiple data sources, intelligent chunking, and metadata extraction.
    """
    
    def __init__(self, config: Optional[IngestionConfig] = None):
        """Initialize the data ingestion agent
        
        Args:
            config: Configuration for ingestion. Uses defaults if None.
        """
        self.config = config or IngestionConfig()
        self._metrics: Dict[str, Any] = {
            'files_processed': 0,
            'total_chunks': 0,
            'processing_time': 0.0,
            'errors': []
        }
        self._preprocessors: List[Callable] = []
        self._start_time: Optional[float] = None
        
        # Setup output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data Ingestion Agent initialized with config: {self.config}")
    
    def ingest_pdf(
        self,
        file_path: str,
        extract_images: bool = False,
        ocr_enabled: bool = False
    ) -> List[Document]:
        """
        Ingest PDF document(s)
        
        Args:
            file_path: Path to PDF file or directory
            extract_images: Extract and describe images
            ocr_enabled: Use OCR for scanned PDFs
            
        Returns:
            List of Document objects
        """
        self._start_timer()
        logger.info(f"Ingesting PDF: {file_path}")
        
        try:
            path = Path(file_path)
            
            if path.is_file():
                # Single PDF file
                reader = PDFReader()
                documents = reader.load_data(file_path)
            else:
                # Directory of PDFs
                reader = SimpleDirectoryReader(
                    input_dir=str(path),
                    required_exts=[".pdf"],
                    recursive=False
                )
                documents = reader.load_data()
            
            # Process documents
            documents = self._process_documents(documents)
            
            # Update metrics
            self._metrics['files_processed'] += 1
            self._metrics['total_chunks'] += len(documents)
            self._metrics['processing_time'] = time.time() - self._start_time
            
            logger.info(f"Successfully ingested {len(documents)} documents from PDF")
            return documents
            
        except Exception as e:
            logger.error(f"Error ingesting PDF: {e}", exc_info=True)
            self._metrics['errors'].append(str(e))
            raise
    
    def ingest_directory(
        self,
        directory_path: str,
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Ingest entire directory of documents
        
        Args:
            directory_path: Path to directory
            recursive: Recurse into subdirectories
            extensions: List of file extensions to include
            exclude_patterns: Patterns to exclude
            
        Returns:
            List of Document objects
        """
        self._start_timer()
        logger.info(f"Ingesting directory: {directory_path}")
        
        try:
            # Normalize extensions to match Path.suffix format (e.g., ".pdf")
            normalized_exts = None
            if extensions:
                normalized_exts = [
                    ext if isinstance(ext, str) and ext.startswith(".") else f".{ext}"
                    for ext in extensions
                ]
            
            reader = SimpleDirectoryReader(
                input_dir=directory_path,
                recursive=recursive,
                required_exts=normalized_exts,
                exclude_hidden=True,
                exclude=exclude_patterns
            )
            
            documents = reader.load_data(show_progress=self.config.show_progress)
            
            # Process documents
            documents = self._process_documents(documents)
            
            # Update metrics
            self._metrics['files_processed'] += len(documents)
            self._metrics['total_chunks'] += len(documents)
            self._metrics['processing_time'] = time.time() - self._start_time
            
            logger.info(f"Successfully ingested {len(documents)} documents from directory")
            return documents
            
        except Exception as e:
            logger.error(f"Error ingesting directory: {e}", exc_info=True)
            self._metrics['errors'].append(str(e))
            raise
    
    def ingest_web(
        self,
        url: str,
        depth: int = 1,
        max_pages: int = 50,
        include_pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None
    ) -> List[Document]:
        """
        Scrape and ingest web content
        
        Args:
            url: Starting URL
            depth: Crawl depth
            max_pages: Maximum pages to crawl
            include_pattern: URL pattern to include
            exclude_pattern: URL pattern to exclude
            
        Returns:
            List of Document objects
        """
        self._start_timer()
        logger.info(f"Scraping web content from: {url}")
        
        try:
            # Import web readers (optional dependency)
            from llama_index.readers.web import SimpleWebPageReader
            
            reader = SimpleWebPageReader(html_to_text=True)
            documents = reader.load_data([url])
            
            # Process documents
            documents = self._process_documents(documents)
            
            # Update metrics
            self._metrics['files_processed'] += 1
            self._metrics['total_chunks'] += len(documents)
            self._metrics['processing_time'] = time.time() - self._start_time
            
            logger.info(f"Successfully scraped {len(documents)} pages")
            return documents
            
        except ImportError:
            logger.error("Web scraping requires: pip install llama-index-readers-web")
            raise
        except Exception as e:
            logger.error(f"Error scraping web: {e}", exc_info=True)
            self._metrics['errors'].append(str(e))
            raise
    
    def ingest_api(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        text_field: str = "content",
        metadata_fields: Optional[List[str]] = None,
        pagination: bool = False
    ) -> List[Document]:
        """
        Ingest data from REST API
        
        Args:
            endpoint: API endpoint URL
            headers: HTTP headers
            params: Query parameters
            text_field: Field containing text content
            metadata_fields: Fields to include as metadata
            pagination: Enable pagination support
            
        Returns:
            List of Document objects
        """
        self._start_timer()
        logger.info(f"Ingesting from API: {endpoint}")
        
        try:
            import requests
            
            response = requests.get(
                endpoint,
                headers=headers or {},
                params=params or {},
                timeout=self.config.api_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Convert API response to documents
            documents = []
            items = data if isinstance(data, list) else data.get('results', [data])
            
            for item in items:
                text = item.get(text_field, "")
                metadata = {
                    field: item.get(field)
                    for field in (metadata_fields or [])
                    if field in item
                }
                metadata['source'] = endpoint
                metadata['ingested_at'] = datetime.now().isoformat()
                
                documents.append(Document(text=text, metadata=metadata))
            
            # Process documents
            documents = self._process_documents(documents)
            
            # Update metrics
            self._metrics['files_processed'] += 1
            self._metrics['total_chunks'] += len(documents)
            self._metrics['processing_time'] = time.time() - self._start_time
            
            logger.info(f"Successfully ingested {len(documents)} records from API")
            return documents
            
        except Exception as e:
            logger.error(f"Error ingesting from API: {e}", exc_info=True)
            self._metrics['errors'].append(str(e))
            raise
    
    def ingest_database(
        self,
        connection: str,
        query: str,
        text_columns: List[str],
        metadata_columns: Optional[List[str]] = None,
        batch_size: int = 1000
    ) -> List[Document]:
        """
        Ingest data from database
        
        Args:
            connection: Database connection string
            query: SQL query to execute
            text_columns: Columns containing text content
            metadata_columns: Columns to include as metadata
            batch_size: Batch size for fetching
            
        Returns:
            List of Document objects
        """
        self._start_timer()
        logger.info(f"Ingesting from database with query: {query[:100]}...")
        
        try:
            from llama_index.readers.database import DatabaseReader
            
            reader = DatabaseReader(uri=connection)
            load_kwargs: Dict[str, Any] = {}
            if text_columns:
                load_kwargs["text_columns"] = text_columns
            if metadata_columns:
                load_kwargs["metadata_columns"] = metadata_columns
            documents = reader.load_data(query=query, **load_kwargs)
            
            # Process documents
            documents = self._process_documents(documents)
            
            # Update metrics
            self._metrics['files_processed'] += 1
            self._metrics['total_chunks'] += len(documents)
            self._metrics['processing_time'] = time.time() - self._start_time
            
            logger.info(f"Successfully ingested {len(documents)} records from database")
            return documents
            
        except ImportError:
            logger.error("Database ingestion requires: pip install llama-index-readers-database sqlalchemy")
            raise
        except Exception as e:
            logger.error(f"Error ingesting from database: {e}", exc_info=True)
            self._metrics['errors'].append(str(e))
            raise
    
    def create_index(
        self,
        documents: List[Document],
        index_name: Optional[str] = None
    ) -> VectorStoreIndex:
        """
        Create vector store index from documents
        
        Args:
            documents: List of documents
            index_name: Name for the index
            
        Returns:
            VectorStoreIndex
        """
        logger.info(f"Creating index from {len(documents)} documents")
        
        try:
            index = VectorStoreIndex.from_documents(
                documents,
                show_progress=self.config.show_progress
            )
            
            logger.info("Index created successfully")
            return index
            
        except Exception as e:
            logger.error(f"Error creating index: {e}", exc_info=True)
            raise
    
    def save_index(self, index: VectorStoreIndex, output_path: Optional[str] = None):
        """
        Save index to disk
        
        Args:
            index: VectorStoreIndex to save
            output_path: Output path (uses config.output_dir if None)
        """
        output_path = output_path or self.config.output_dir
        logger.info(f"Saving index to: {output_path}")
        
        try:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            index.storage_context.persist(persist_dir=output_path)
            logger.info("Index saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}", exc_info=True)
            raise
    
    def add_preprocessor(self, preprocessor: Callable[[Document], Document]):
        """Add a preprocessing function for documents"""
        self._preprocessors.append(preprocessor)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get ingestion metrics"""
        return self._metrics.copy()
    
    def _process_documents(self, documents: List[Document]) -> List[Document]:
        """Apply processing steps to documents"""
        
        # Apply custom preprocessors
        for preprocessor in self._preprocessors:
            documents = [preprocessor(doc) for doc in documents]
        
        # Clean text if enabled
        if self.config.clean_text:
            documents = self._clean_documents(documents)
        
        # Chunk documents
        documents = self._chunk_documents(documents)
        
        return documents
    
    def _clean_documents(self, documents: List[Document]) -> List[Document]:
        """Clean document text"""
        import re
        
        for doc in documents:
            # Remove extra whitespace
            doc.text = re.sub(r'\s+', ' ', doc.text)
            doc.text = doc.text.strip()
        
        return documents
    
    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents based on strategy"""
        
        if self.config.chunking_strategy == "sentence":
            splitter = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        elif self.config.chunking_strategy == "semantic":
            from llama_index.core.embeddings import resolve_embed_model
            embed_model = resolve_embed_model("local")
            splitter = SemanticSplitterNodeParser(
                embed_model=embed_model,
                buffer_size=1
            )
        else:  # fixed
            splitter = TokenTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        
        # Split documents into nodes then back to documents
        nodes = splitter.get_nodes_from_documents(documents)
        
        # Convert nodes back to documents
        chunked_docs = []
        for node in nodes:
            doc = Document(
                text=node.get_content(),
                metadata=node.metadata
            )
            chunked_docs.append(doc)
        
        return chunked_docs
    
    def _start_timer(self):
        """Start processing timer"""
        self._start_time = time.time()
