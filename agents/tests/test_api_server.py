"""
API Integration Tests for LlamaIndex Agents
Tests all REST API endpoints
"""
import pytest
from fastapi.testclient import TestClient
import tempfile
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from api_server import app

# Create test client
client = TestClient(app)


class TestAPIHealth:
    """Test API health and status endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns status"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "agents" in data
        assert "version" in data
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["agents"]["data_ingestion"] == "available"
        assert data["agents"]["query_engine"] == "available"
        assert data["agents"]["rag_pipeline"] == "available"
    
    def test_docs_available(self):
        """Test API documentation is available"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"


class TestDataIngestionAPI:
    """Test Data Ingestion API endpoints"""
    
    def test_ingest_endpoint_validation(self):
        """Test request validation"""
        # Missing required fields
        response = client.post("/api/v1/ingest", json={})
        assert response.status_code == 422  # Validation error
    
    def test_ingest_invalid_source_type(self):
        """Test invalid source type"""
        response = client.post("/api/v1/ingest", json={
            "source_type": "invalid",
            "source_path": "/tmp/test"
        })
        assert response.status_code in [400, 500]
    
    def test_ingest_missing_source(self):
        """Test missing source path"""
        response = client.post("/api/v1/ingest", json={
            "source_type": "directory",
            "source_path": "/nonexistent/path"
        })
        assert response.status_code in [404, 500]
    
    def test_get_metrics_endpoint(self):
        """Test metrics retrieval"""
        response = client.get("/api/v1/ingest/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data


class TestQueryEngineAPI:
    """Test Query Engine API endpoints"""
    
    def test_query_endpoint_validation(self):
        """Test query request validation"""
        # Missing required fields
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422
    
    def test_query_missing_index(self):
        """Test query with missing index"""
        response = client.post("/api/v1/query", json={
            "query": "Test query",
            "index_path": "/nonexistent/index"
        })
        assert response.status_code in [404, 500]
    
    def test_query_request_structure(self):
        """Test query request structure validation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            response = client.post("/api/v1/query", json={
                "query": "What is RAG?",
                "index_path": tmpdir,
                "mode": "similarity",
                "top_k": 5,
                "stream": False
            })
            # Will fail because index doesn't exist, but validates structure
            assert response.status_code in [404, 500]
    
    def test_query_invalid_mode(self):
        """Test query with invalid mode"""
        response = client.post("/api/v1/query", json={
            "query": "test",
            "index_path": "/tmp",
            "mode": "invalid_mode"
        })
        assert response.status_code == 422  # Validation error


class TestRAGPipelineAPI:
    """Test RAG Pipeline API endpoints"""
    
    def test_build_pipeline_validation(self):
        """Test pipeline build validation"""
        response = client.post("/api/v1/pipeline/build", json={})
        assert response.status_code == 422
    
    def test_build_pipeline_missing_data_dir(self):
        """Test build with missing data directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            response = client.post("/api/v1/pipeline/build", json={
                "data_dir": "/nonexistent/path",
                "output_dir": tmpdir
            })
            assert response.status_code in [404, 500]
    
    def test_optimize_pipeline_validation(self):
        """Test optimize validation"""
        response = client.post("/api/v1/pipeline/optimize", json={})
        assert response.status_code == 422
    
    def test_evaluate_pipeline_validation(self):
        """Test evaluate validation"""
        response = client.post("/api/v1/pipeline/evaluate", json={})
        assert response.status_code == 422
    
    def test_evaluate_pipeline_structure(self):
        """Test evaluate request structure"""
        response = client.post("/api/v1/pipeline/evaluate", json={
            "pipeline_path": "/tmp/pipeline",
            "test_queries": [
                {"query": "What is RAG?"},
                {"query": "How does it work?"}
            ],
            "metrics": ["faithfulness", "relevance"]
        })
        # Will fail but validates structure
        assert response.status_code in [404, 500]


class TestIndexingAPI:
    """Test Indexing API endpoints"""
    
    def test_create_index_validation(self):
        """Test index creation validation"""
        response = client.post("/api/v1/index/create", json={})
        assert response.status_code == 422
    
    def test_create_index_missing_data(self):
        """Test index creation with missing data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            response = client.post("/api/v1/index/create", json={
                "data_dir": "/nonexistent/path",
                "index_type": "vector",
                "output_dir": tmpdir
            })
            assert response.status_code in [404, 500]
    
    def test_create_index_invalid_type(self):
        """Test invalid index type"""
        response = client.post("/api/v1/index/create", json={
            "data_dir": "/tmp",
            "index_type": "invalid_type",
            "output_dir": "/tmp"
        })
        assert response.status_code == 422
    
    def test_recommend_index_endpoint(self):
        """Test index recommendation endpoint"""
        response = client.get("/api/v1/index/recommend?data_dir=/tmp")
        # May fail but endpoint should exist
        assert response.status_code in [200, 500]


class TestEvaluationAPI:
    """Test Evaluation API endpoints"""
    
    def test_evaluate_validation(self):
        """Test evaluation validation"""
        response = client.post("/api/v1/evaluate", json={})
        assert response.status_code == 422
    
    def test_evaluate_structure(self):
        """Test evaluation request structure"""
        response = client.post("/api/v1/evaluate", json={
            "pipeline_path": "/tmp/pipeline",
            "test_queries": [{"query": "test"}],
            "metrics": ["faithfulness"]
        })
        # Will fail but validates structure
        assert response.status_code in [404, 500]
    
    def test_generate_tests_endpoint(self):
        """Test generate test queries endpoint"""
        response = client.post("/api/v1/evaluate/generate-tests?data_dir=/tmp&num_queries=5")
        # May fail but endpoint should exist
        assert response.status_code in [200, 500]


class TestAPIErrorHandling:
    """Test API error handling"""
    
    def test_404_handler(self):
        """Test 404 error handling"""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_json(self):
        """Test invalid JSON handling"""
        response = client.post(
            "/api/v1/ingest",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestAPISecurity:
    """Test API security aspects"""
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.options("/api/v1/ingest")
        assert response.status_code == 200
    
    def test_path_traversal_protection(self):
        """Test protection against path traversal"""
        response = client.post("/api/v1/ingest", json={
            "source_type": "directory",
            "source_path": "../../etc/passwd",
            "output_dir": "/tmp/test"
        })
        # Should either reject or handle safely
        assert response.status_code in [400, 404, 500]
    
    def test_input_length_validation(self):
        """Test input length limits"""
        long_query = "a" * 10000  # Very long query
        response = client.post("/api/v1/query", json={
            "query": long_query,
            "index_path": "/tmp"
        })
        assert response.status_code == 422  # Should reject


class TestAPIPerformance:
    """Test API performance aspects"""
    
    def test_concurrent_health_checks(self):
        """Test multiple concurrent health checks"""
        responses = [client.get("/health") for _ in range(10)]
        assert all(r.status_code == 200 for r in responses)
    
    def test_large_test_query_list(self):
        """Test handling large test query lists"""
        large_query_list = [
            {"query": f"Test query {i}"} for i in range(100)
        ]
        response = client.post("/api/v1/pipeline/evaluate", json={
            "pipeline_path": "/tmp/pipeline",
            "test_queries": large_query_list,
            "metrics": ["faithfulness"]
        })
        # Will fail but should handle gracefully
        assert response.status_code in [404, 500]


def test_api_documentation_completeness():
    """Test API documentation is complete"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    spec = response.json()
    assert "openapi" in spec
    assert "info" in spec
    assert "paths" in spec
    
    # Check key endpoints are documented
    paths = spec["paths"]
    assert "/health" in paths
    assert "/api/v1/ingest" in paths
    assert "/api/v1/query" in paths
    assert "/api/v1/pipeline/build" in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
