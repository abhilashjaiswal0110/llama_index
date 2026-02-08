"""
CLAUDE Agent Utilities
Best practices for building production-grade CLAUDE-based agents
"""
import logging
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a message in conversation history"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """
    Manages conversation history with intelligent context window management.
    
    Features:
    - Automatic truncation when approaching token limits
    - Priority-based message retention
    - Conversation summarization for long interactions
    """
    
    def __init__(self, max_tokens: int = 4000, summary_threshold: int = 3000):
        """
        Initialize conversation memory.
        
        Args:
            max_tokens: Maximum tokens to keep in context
            summary_threshold: When to trigger summarization
        """
        self.max_tokens = max_tokens
        self.summary_threshold = summary_threshold
        self.messages: List[Message] = []
        self.total_tokens = 0
        self.summary: Optional[str] = None
        
    def add_message(self, role: str, content: str, tokens: Optional[int] = None, **metadata):
        """
        Add a message to conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            tokens: Token count (estimated if None)
            metadata: Additional metadata
        """
        if tokens is None:
            # Rough estimation: ~4 chars per token
            tokens = len(content) // 4
            
        message = Message(
            role=role,
            content=content,
            tokens=tokens,
            metadata=metadata
        )
        
        self.messages.append(message)
        self.total_tokens += tokens
        
        # Manage context window
        if self.total_tokens > self.summary_threshold:
            self._manage_context()
            
    def _manage_context(self):
        """Manage context window by truncating or summarizing old messages"""
        if self.total_tokens <= self.max_tokens:
            return
            
        # Keep most recent messages, summarize older ones
        messages_to_keep = []
        tokens_kept = 0
        
        # Work backwards from most recent
        for msg in reversed(self.messages):
            if tokens_kept + msg.tokens <= self.max_tokens * 0.8:  # Keep 80% for recent
                messages_to_keep.insert(0, msg)
                tokens_kept += msg.tokens
            else:
                break
                
        # Create summary of removed messages
        removed_messages = [m for m in self.messages if m not in messages_to_keep]
        if removed_messages and not self.summary:
            self.summary = self._create_summary(removed_messages)
            
        self.messages = messages_to_keep
        self.total_tokens = tokens_kept
        
    def _create_summary(self, messages: List[Message]) -> str:
        """Create a summary of conversation messages"""
        # Simple summarization - in production, use LLM
        summary_parts = []
        for msg in messages:
            summary_parts.append(f"{msg.role}: {msg.content[:100]}...")
        return "\n".join(summary_parts)
        
    def get_context(self, include_summary: bool = True) -> List[Dict[str, str]]:
        """
        Get conversation context for next request.
        
        Args:
            include_summary: Include conversation summary if available
            
        Returns:
            List of message dicts with role and content
        """
        context = []
        
        if include_summary and self.summary:
            context.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.summary}"
            })
            
        for msg in self.messages:
            context.append({
                "role": msg.role,
                "content": msg.content
            })
            
        return context
        
    def clear(self):
        """Clear conversation history"""
        self.messages = []
        self.total_tokens = 0
        self.summary = None
        
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        return {
            "total_messages": len(self.messages),
            "total_tokens": self.total_tokens,
            "has_summary": self.summary is not None,
            "utilization": self.total_tokens / self.max_tokens
        }


class AgentOrchestrator:
    """
    Orchestrates multiple agents in complex workflows.
    
    Supports:
    - Sequential execution
    - Parallel execution
    - Conditional branching
    - Error recovery
    """
    
    def __init__(self):
        self.execution_history: List[Dict[str, Any]] = []
        
    async def run_pipeline(
        self,
        steps: List[Tuple[str, Callable, Dict[str, Any]]],
        mode: str = "sequential"
    ) -> Dict[str, Any]:
        """
        Run a multi-step agent pipeline.
        
        Args:
            steps: List of (name, agent_function, params) tuples
            mode: 'sequential' or 'parallel'
            
        Returns:
            Dictionary with results from each step
        """
        results = {}
        
        if mode == "sequential":
            results = await self._run_sequential(steps)
        elif mode == "parallel":
            results = await self._run_parallel(steps)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        return results
        
    async def _run_sequential(
        self,
        steps: List[Tuple[str, Callable, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Execute steps sequentially"""
        results = {}
        previous_result = None
        
        for step_name, agent_func, params in steps:
            try:
                # Allow referencing previous result
                if "from_previous" in str(params):
                    params = self._resolve_params(params, previous_result)
                    
                logger.info(f"Executing step: {step_name}")
                
                # Execute step
                if asyncio.iscoroutinefunction(agent_func):
                    result = await agent_func(**params)
                else:
                    result = agent_func(**params)
                    
                results[step_name] = result
                previous_result = result
                
                # Track execution
                self.execution_history.append({
                    "step": step_name,
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error in step {step_name}: {e}")
                results[step_name] = {"error": str(e)}
                
                self.execution_history.append({
                    "step": step_name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Stop on error in sequential mode
                break
                
        return results
        
    async def _run_parallel(
        self,
        steps: List[Tuple[str, Callable, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Execute steps in parallel"""
        tasks = []
        step_names = []
        
        for step_name, agent_func, params in steps:
            step_names.append(step_name)
            
            if asyncio.iscoroutinefunction(agent_func):
                tasks.append(agent_func(**params))
            else:
                # Wrap sync function
                tasks.append(asyncio.to_thread(agent_func, **params))
                
        # Execute all in parallel
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Map results back to step names
        results = {}
        for step_name, result in zip(step_names, task_results):
            if isinstance(result, Exception):
                results[step_name] = {"error": str(result)}
            else:
                results[step_name] = result
                
        return results
        
    def _resolve_params(self, params: Dict[str, Any], previous_result: Any) -> Dict[str, Any]:
        """Resolve parameter references to previous results"""
        resolved = {}
        for key, value in params.items():
            if value == "from_previous":
                resolved[key] = previous_result
            else:
                resolved[key] = value
        return resolved
        
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.execution_history
        
    def clear_history(self):
        """Clear execution history"""
        self.execution_history = []


class StreamingResponseManager:
    """
    Manages streaming responses with backpressure and cancellation.
    """
    
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.cancelled = False
        
    async def stream_with_backpressure(
        self,
        generator: AsyncGenerator[str, None],
        process_chunk: Optional[Callable[[str], str]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream with backpressure handling.
        
        Args:
            generator: Async generator producing chunks
            process_chunk: Optional function to process each chunk
            
        Yields:
            Processed chunks
        """
        try:
            async for chunk in generator:
                if self.cancelled:
                    logger.info("Streaming cancelled")
                    break
                    
                # Process chunk if needed
                if process_chunk:
                    chunk = process_chunk(chunk)
                    
                # Add to buffer
                self.buffer.append(chunk)
                
                yield chunk
                
                # Simple backpressure: wait if buffer full
                if len(self.buffer) >= self.buffer_size:
                    await asyncio.sleep(0.01)
                    
        except asyncio.CancelledError:
            logger.info("Streaming cancelled by client")
            self.cancelled = True
            raise
            
    def cancel(self):
        """Cancel streaming"""
        self.cancelled = True


class PromptTemplate:
    """
    Advanced prompt template with few-shot examples and chain-of-thought.
    """
    
    def __init__(self, template: str, examples: Optional[List[Dict[str, str]]] = None):
        """
        Initialize prompt template.
        
        Args:
            template: Base template with {placeholders}
            examples: Few-shot examples
        """
        self.template = template
        self.examples = examples or []
        
    def format(self, **kwargs) -> str:
        """Format template with provided values"""
        prompt = self.template.format(**kwargs)
        
        # Add few-shot examples if available
        if self.examples:
            examples_text = "\n\n".join([
                f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
                for i, ex in enumerate(self.examples)
            ])
            prompt = f"{examples_text}\n\n{prompt}"
            
        return prompt
        
    def add_example(self, input_text: str, output_text: str):
        """Add a few-shot example"""
        self.examples.append({
            "input": input_text,
            "output": output_text
        })
        
    def add_chain_of_thought(self, base_prompt: str) -> str:
        """Add chain-of-thought reasoning instructions"""
        cot_instruction = (
            "\n\nPlease think through this step-by-step:\n"
            "1. First, identify the key information needed\n"
            "2. Then, reason through the problem\n"
            "3. Finally, provide your answer\n\n"
        )
        return base_prompt + cot_instruction


# Export public API
__all__ = [
    'Message',
    'ConversationMemory',
    'AgentOrchestrator',
    'StreamingResponseManager',
    'PromptTemplate'
]
