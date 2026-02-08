"""
Unit tests for Data Ingestion Agent
Tests all ingestion modes and functionality
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ingestion_agent.agent import DataIngestionAgent, IngestionConfig


class TestDataIngestionAgent:
    """Test suite for Data Ingestion Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance for testing"""
        config = IngestionConfig(
            chunk_size=512,
            chunk_overlap=100,
            chunking_strategy="fixed",
            output_dir=None
        )
        return DataIngestionAgent(config=config)
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent is not None
        assert agent.config.chunk_size == 512
        assert agent.config.chunk_overlap == 100
        assert agent.config.chunking_strategy == "fixed"
    
    def test_agent_initialization_default_config(self):
        """Test agent initializes with default config"""
        agent = DataIngestionAgent()
        assert agent is not None
        assert agent.config is not None
        assert agent.config.chunk_size == 1024  # default
    
    @patch('data_ingestion_agent.agent.SimpleDirectoryReader')
    def test_ingest_directory(self, mock_reader, agent, temp_dir):
        """Test directory ingestion"""
        # Create test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Test content for ingestion")
        
        # Mock reader
        mock_doc = Mock()
        mock_doc.text = "Test content"
        mock_doc.metadata = {}
        mock_reader.return_value.load_data.return_value = [mock_doc]
        
        # Test ingestion
        docs = agent.ingest_directory(temp_dir, recursive=False)
        
        assert docs is not None
        assert len(docs) > 0
        mock_reader.assert_called_once()
    
    def test_get_metrics(self, agent):
        """Test metrics retrieval"""
        metrics = agent.get_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert 'documents_processed' in metrics
        assert 'total_chunks' in metrics
        assert 'processing_time' in metrics
    
    def test_add_preprocessor(self, agent):
        """Test adding custom preprocessor"""
        def custom_preprocessor(doc):
            doc.text = doc.text.upper()
            return doc
        
        agent.add_preprocessor(custom_preprocessor)
        assert len(agent.preprocessors) > 0
    
    @patch('data_ingestion_agent.agent.VectorStoreIndex')
    def test_create_index(self, mock_index, agent):
        """Test index creation from documents"""
        # Mock documents
        mock_doc = Mock()
        mock_doc.text = "Test document"
        mock_doc.metadata = {}
        
        mock_index.from_documents.return_value = Mock()
        
        # Create index
        index = agent.create_index([mock_doc])
        
        assert index is not None
        mock_index.from_documents.assert_called_once()
    
    @patch('data_ingestion_agent.agent.VectorStoreIndex')
    def test_save_index(self, mock_index, agent, temp_dir):
        """Test saving index to disk"""
        # Mock index with storage_context
        mock_idx = Mock()
        mock_storage_context = Mock()
        mock_idx.storage_context = mock_storage_context
        
        # Save index
        output_path = str(Path(temp_dir) / "index")
        agent.save_index(mock_idx, output_path)
        
        # Verify persist was called
        mock_storage_context.persist.assert_called_once()


class TestIngestionConfig:
    """Test IngestionConfig validation"""
    
    def test_valid_config(self):
        """Test valid configuration"""
        config = IngestionConfig(
            chunk_size=1024,
            chunk_overlap=200,
            chunking_strategy="sentence"
        )
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 200
        assert config.chunking_strategy == "sentence"
    
    def test_default_values(self):
        """Test default configuration values"""
        config = IngestionConfig()
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 200
        assert config.chunking_strategy == "fixed"


def test_agent_api_compatibility():
    """Test that agent exposes expected API methods"""
    agent = DataIngestionAgent()
    
    # Verify required methods exist
    assert hasattr(agent, 'ingest_directory')
    assert hasattr(agent, 'ingest_pdf')
    assert hasattr(agent, 'ingest_web')
    assert hasattr(agent, 'ingest_api')
    assert hasattr(agent, 'ingest_database')
    assert hasattr(agent, 'create_index')
    assert hasattr(agent, 'save_index')
    assert hasattr(agent, 'get_metrics')
    
    # Verify methods are callable
    assert callable(agent.ingest_directory)
    assert callable(agent.create_index)
    assert callable(agent.get_metrics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
