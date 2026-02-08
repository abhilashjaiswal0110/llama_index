"""
Tests for CLAUDE Agent Utilities
"""
import pytest
import asyncio
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_agent_utils import (
    Message,
    ConversationMemory,
    AgentOrchestrator,
    StreamingResponseManager,
    PromptTemplate
)


class TestConversationMemory:
    """Test conversation memory management"""
    
    def test_add_message(self):
        """Test adding messages to memory"""
        memory = ConversationMemory(max_tokens=1000)
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there!")
        
        assert len(memory.messages) == 2
        assert memory.messages[0].role == "user"
        assert memory.messages[1].role == "assistant"
        
    def test_context_retrieval(self):
        """Test getting context for API calls"""
        memory = ConversationMemory()
        memory.add_message("user", "What is RAG?")
        memory.add_message("assistant", "RAG stands for Retrieval-Augmented Generation")
        
        context = memory.get_context(include_summary=False)
        
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert "RAG" in context[0]["content"]
        
    def test_context_truncation(self):
        """Test automatic context truncation"""
        memory = ConversationMemory(max_tokens=100, summary_threshold=80)
        
        # Add many messages
        for i in range(10):
            memory.add_message("user", f"Message {i}" * 10, tokens=20)
            
        # Should have triggered truncation
        assert memory.total_tokens <= memory.max_tokens
        
    def test_get_stats(self):
        """Test statistics retrieval"""
        memory = ConversationMemory()
        memory.add_message("user", "Test", tokens=10)
        
        stats = memory.get_stats()
        
        assert stats["total_messages"] == 1
        assert stats["total_tokens"] == 10
        assert "utilization" in stats
        
    def test_clear_memory(self):
        """Test clearing conversation"""
        memory = ConversationMemory()
        memory.add_message("user", "Test")
        memory.clear()
        
        assert len(memory.messages) == 0
        assert memory.total_tokens == 0


class TestAgentOrchestrator:
    """Test agent orchestration"""
    
    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        """Test sequential agent execution"""
        orchestrator = AgentOrchestrator()
        
        # Define simple test functions
        def step1(**kwargs):
            return {"result": "step1_done"}
            
        def step2(**kwargs):
            return {"result": "step2_done"}
            
        steps = [
            ("step1", step1, {}),
            ("step2", step2, {})
        ]
        
        results = await orchestrator.run_pipeline(steps, mode="sequential")
        
        assert "step1" in results
        assert "step2" in results
        assert results["step1"]["result"] == "step1_done"
        
    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel agent execution"""
        orchestrator = AgentOrchestrator()
        
        async def async_step1(**kwargs):
            await asyncio.sleep(0.1)
            return {"result": "async_step1"}
            
        async def async_step2(**kwargs):
            await asyncio.sleep(0.1)
            return {"result": "async_step2"}
            
        steps = [
            ("step1", async_step1, {}),
            ("step2", async_step2, {})
        ]
        
        results = await orchestrator.run_pipeline(steps, mode="parallel")
        
        assert "step1" in results
        assert "step2" in results
        
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in pipeline"""
        orchestrator = AgentOrchestrator()
        
        def failing_step(**kwargs):
            raise ValueError("Test error")
            
        def normal_step(**kwargs):
            return {"result": "success"}
            
        steps = [
            ("failing", failing_step, {}),
            ("normal", normal_step, {})
        ]
        
        results = await orchestrator.run_pipeline(steps, mode="sequential")
        
        assert "error" in results["failing"]
        assert "normal" not in results  # Should stop on error
        
    def test_execution_history(self):
        """Test execution history tracking"""
        orchestrator = AgentOrchestrator()
        
        # History should be empty initially
        history = orchestrator.get_execution_history()
        assert len(history) == 0
        
        # Clear should work
        orchestrator.clear_history()
        assert len(orchestrator.get_execution_history()) == 0


class TestStreamingResponseManager:
    """Test streaming response management"""
    
    @pytest.mark.asyncio
    async def test_basic_streaming(self):
        """Test basic streaming functionality"""
        manager = StreamingResponseManager()
        
        async def simple_generator():
            for i in range(5):
                yield f"chunk_{i}"
                
        chunks = []
        async for chunk in manager.stream_with_backpressure(simple_generator()):
            chunks.append(chunk)
            
        assert len(chunks) == 5
        assert chunks[0] == "chunk_0"
        
    @pytest.mark.asyncio
    async def test_chunk_processing(self):
        """Test chunk processing during streaming"""
        manager = StreamingResponseManager()
        
        async def simple_generator():
            for i in range(3):
                yield f"data"
                
        def process_chunk(chunk):
            return chunk.upper()
            
        chunks = []
        async for chunk in manager.stream_with_backpressure(
            simple_generator(),
            process_chunk=process_chunk
        ):
            chunks.append(chunk)
            
        assert all(c == "DATA" for c in chunks)
        
    @pytest.mark.asyncio
    async def test_cancellation(self):
        """Test streaming cancellation"""
        manager = StreamingResponseManager()
        
        async def long_generator():
            for i in range(100):
                yield f"chunk_{i}"
                await asyncio.sleep(0.01)
                
        # Cancel after first chunk
        chunk_count = 0
        async for chunk in manager.stream_with_backpressure(long_generator()):
            chunk_count += 1
            if chunk_count == 1:
                manager.cancel()
                
        assert chunk_count == 1


class TestPromptTemplate:
    """Test prompt template functionality"""
    
    def test_basic_formatting(self):
        """Test basic template formatting"""
        template = PromptTemplate("Hello {name}, how are you?")
        result = template.format(name="Alice")
        
        assert "Alice" in result
        assert "how are you" in result
        
    def test_few_shot_examples(self):
        """Test few-shot example inclusion"""
        template = PromptTemplate(
            "Translate: {text}",
            examples=[
                {"input": "Hello", "output": "Hola"},
                {"input": "Goodbye", "output": "Adiós"}
            ]
        )
        
        result = template.format(text="Thank you")
        
        assert "Example 1" in result
        assert "Hola" in result
        assert "Adiós" in result
        
    def test_add_example(self):
        """Test adding examples dynamically"""
        template = PromptTemplate("Query: {query}")
        template.add_example("What is AI?", "AI is artificial intelligence")
        
        result = template.format(query="What is ML?")
        
        assert "artificial intelligence" in result
        
    def test_chain_of_thought(self):
        """Test chain-of-thought addition"""
        template = PromptTemplate("Solve: {problem}")
        
        base_prompt = template.format(problem="2+2")
        cot_prompt = template.add_chain_of_thought(base_prompt)
        
        assert "step-by-step" in cot_prompt
        assert "2+2" in cot_prompt


class TestMessage:
    """Test Message dataclass"""
    
    def test_message_creation(self):
        """Test creating a message"""
        msg = Message(role="user", content="Hello", tokens=5)
        
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tokens == 5
        assert isinstance(msg.timestamp, datetime)
        
    def test_message_metadata(self):
        """Test message metadata"""
        msg = Message(
            role="assistant",
            content="Response",
            metadata={"model": "claude-3", "temperature": 0.7}
        )
        
        assert msg.metadata["model"] == "claude-3"
        assert msg.metadata["temperature"] == 0.7


def test_imports():
    """Test that all exports are available"""
    from claude_agent_utils import (
        Message,
        ConversationMemory,
        AgentOrchestrator,
        StreamingResponseManager,
        PromptTemplate
    )
    
    # All imports should work
    assert Message is not None
    assert ConversationMemory is not None
    assert AgentOrchestrator is not None
    assert StreamingResponseManager is not None
    assert PromptTemplate is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
