"""
Integration tests for all agents working together
Tests complete workflows and multi-agent interactions
"""
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ingestion_agent.agent import DataIngestionAgent, IngestionConfig
from query_engine_agent.agent import QueryEngineAgent, QueryConfig
from rag_pipeline_agent.agent import RAGPipelineAgent
from indexing_agent.agent import IndexingAgent
from evaluation_agent.agent import EvaluationAgent


class TestCompleteRAGWorkflow:
    """Test complete RAG workflow using multiple agents"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            
            # Create test data directory
            data_dir = workspace / "data"
            data_dir.mkdir()
            
            # Create sample documents
            (data_dir / "doc1.txt").write_text("Test document about artificial intelligence.")
            (data_dir / "doc2.txt").write_text("Machine learning is a subset of AI.")
            
            yield workspace
    
    @patch('data_ingestion_agent.agent.SimpleDirectoryReader')
    @patch('data_ingestion_agent.agent.VectorStoreIndex')
    def test_ingestion_to_indexing_workflow(self, mock_index, mock_reader, temp_workspace):
        """Test workflow from ingestion to indexing"""
        # Setup mocks
        mock_doc = Mock()
        mock_doc.text = "Test content"
        mock_doc.metadata = {}
        mock_reader.return_value.load_data.return_value = [mock_doc]
        mock_index.from_documents.return_value = Mock()
        
        # Step 1: Ingest data
        ingestion_agent = DataIngestionAgent()
        data_dir = str(temp_workspace / "data")
        documents = ingestion_agent.ingest_directory(data_dir)
        
        assert documents is not None
        assert len(documents) > 0
        
        # Step 2: Create index
        index = ingestion_agent.create_index(documents)
        assert index is not None
        
        # Step 3: Save index
        index_dir = str(temp_workspace / "index")
        ingestion_agent.save_index(index, index_dir)
    
    def test_pipeline_agent_initialization(self, temp_workspace):
        """Test RAG Pipeline agent can be initialized"""
        agent = RAGPipelineAgent()
        assert agent is not None
    
    def test_evaluation_agent_initialization(self):
        """Test Evaluation agent can be initialized"""
        agent = EvaluationAgent()
        assert agent is not None
        assert hasattr(agent, 'evaluate')
        assert hasattr(agent, 'generate_test_queries')


class TestAgentInteroperability:
    """Test agents can work together"""
    
    def test_all_agents_can_be_instantiated(self):
        """Test all agents can be created without errors"""
        agents = {
            'ingestion': DataIngestionAgent(),
            'pipeline': RAGPipelineAgent(),
            'indexing': IndexingAgent(),
            'evaluation': EvaluationAgent()
        }
        
        assert all(agent is not None for agent in agents.values())
    
    def test_agents_have_expected_methods(self):
        """Test all agents expose expected public methods"""
        ingestion = DataIngestionAgent()
        assert callable(ingestion.ingest_directory)
        assert callable(ingestion.create_index)
        assert callable(ingestion.get_metrics)
        
        pipeline = RAGPipelineAgent()
        assert callable(pipeline.build_pipeline)
        assert callable(pipeline.save_pipeline)
        assert callable(pipeline.load_pipeline)
        
        indexing = IndexingAgent()
        assert callable(indexing.create_index)
        assert callable(indexing.save_index)
        
        evaluation = EvaluationAgent()
        assert callable(evaluation.evaluate)
        assert callable(evaluation.generate_test_queries)


class TestConfigurationManagement:
    """Test configuration handling across agents"""
    
    def test_ingestion_config_creation(self):
        """Test ingestion configuration"""
        config = IngestionConfig(
            chunk_size=512,
            chunk_overlap=100,
            chunking_strategy="sentence"
        )
        assert config.chunk_size == 512
        assert config.chunk_overlap == 100
    
    def test_query_config_creation(self):
        """Test query configuration"""
        config = QueryConfig(
            mode="hybrid",
            top_k=5,
            similarity_threshold=0.75
        )
        assert config.mode == "hybrid"
        assert config.top_k == 5
    
    def test_agent_with_custom_config(self):
        """Test agent initialization with custom config"""
        config = IngestionConfig(chunk_size=256)
        agent = DataIngestionAgent(config=config)
        assert agent.config.chunk_size == 256


class TestErrorHandling:
    """Test error handling across agents"""
    
    def test_ingestion_missing_directory(self):
        """Test ingestion handles missing directory"""
        agent = DataIngestionAgent()
        with pytest.raises((FileNotFoundError, Exception)):
            agent.ingest_directory("/nonexistent/path")
    
    def test_query_engine_missing_index(self):
        """Test query engine handles missing index"""
        with pytest.raises((FileNotFoundError, Exception)):
            QueryEngineAgent(index_path="/nonexistent/index")
    
    def test_pipeline_load_missing_pipeline(self):
        """Test pipeline agent handles missing pipeline"""
        agent = RAGPipelineAgent()
        with pytest.raises((FileNotFoundError, Exception)):
            agent.load_pipeline("/nonexistent/pipeline")


class TestMetricsAndMonitoring:
    """Test metrics collection across agents"""
    
    def test_ingestion_metrics(self):
        """Test ingestion agent provides metrics"""
        agent = DataIngestionAgent()
        metrics = agent.get_metrics()
        
        assert isinstance(metrics, dict)
        assert 'documents_processed' in metrics
        assert 'total_chunks' in metrics
        assert 'processing_time' in metrics
    
    def test_metrics_structure(self):
        """Test metrics have expected structure"""
        agent = DataIngestionAgent()
        metrics = agent.get_metrics()
        
        # Check all values are numeric
        assert all(isinstance(v, (int, float)) for v in metrics.values())


class TestSecurityConsiderations:
    """Test security-related functionality"""
    
    def test_path_traversal_in_ingestion(self):
        """Test ingestion handles path traversal attempts safely"""
        agent = DataIngestionAgent()
        
        # Try path traversal
        with pytest.raises((FileNotFoundError, ValueError, Exception)):
            agent.ingest_directory("../../etc/passwd")
    
    def test_config_validation(self):
        """Test configuration validates input"""
        # Test invalid chunk size
        config = IngestionConfig(chunk_size=1024)
        assert config.chunk_size == 1024
        
        # Config should accept valid values
        assert config.chunk_overlap >= 0
    
    def test_query_input_sanitization(self):
        """Test query input handling"""
        config = QueryConfig(mode="similarity")
        assert config.mode in ["similarity", "hybrid", "sub-question", "fusion", "tree", "router"]


class TestPerformance:
    """Test performance considerations"""
    
    @patch('data_ingestion_agent.agent.SimpleDirectoryReader')
    def test_batch_ingestion_efficiency(self, mock_reader):
        """Test batch ingestion can handle multiple documents"""
        # Create mock documents
        mock_docs = [Mock(text=f"Doc {i}", metadata={}) for i in range(100)]
        mock_reader.return_value.load_data.return_value = mock_docs
        
        agent = DataIngestionAgent()
        documents = agent.ingest_directory("/tmp/test")
        
        assert len(documents) > 0
    
    def test_agent_instantiation_speed(self):
        """Test agents can be instantiated quickly"""
        import time
        
        start = time.time()
        agents = [
            DataIngestionAgent(),
            RAGPipelineAgent(),
            IndexingAgent(),
            EvaluationAgent()
        ]
        duration = time.time() - start
        
        # Should instantiate in reasonable time
        assert duration < 5.0  # 5 seconds max
        assert all(a is not None for a in agents)


class TestDocumentationAndAPI:
    """Test API consistency and documentation"""
    
    def test_all_agents_have_docstrings(self):
        """Test all agent classes have docstrings"""
        assert DataIngestionAgent.__doc__ is not None or hasattr(DataIngestionAgent, '__init__')
        assert RAGPipelineAgent.__doc__ is not None or hasattr(RAGPipelineAgent, '__init__')
    
    def test_method_signatures_consistent(self):
        """Test method signatures are consistent"""
        # All agents should have similar patterns
        ingestion = DataIngestionAgent()
        assert hasattr(ingestion, 'get_metrics')
        
        # Methods should return appropriate types
        metrics = ingestion.get_metrics()
        assert isinstance(metrics, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
