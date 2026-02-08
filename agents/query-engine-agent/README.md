# Query Engine Agent

Advanced querying agent for LlamaIndex with multiple retrieval strategies and response synthesis methods.

## Features

- 🔍 **Multiple query modes:** Similarity, hybrid, sub-question, fusion, tree
- 🎯 **Advanced retrieval:** Top-k, MMR, similarity threshold
- 📝 **Custom prompts:** Flexible prompt templates
- 🔄 **Query transformation:** Expansion, decomposition, refinement
- ⚡ **Streaming support:** Real-time response streaming
- 📊 **Response synthesis:** Multiple synthesis strategies

## Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `similarity` | Vector similarity search | Simple queries, single-hop |
| `hybrid` | Vector + keyword search | Balanced precision/recall |
| `sub-question` | Break into sub-questions | Complex multi-hop queries |
| `fusion` | Multiple retrieval strategies | Maximum coverage |
| `tree` | Tree-based decomposition | Hierarchical questions |
| `router` | Route to specialized engines | Multi-domain queries |

## Installation

```bash
cd query-engine-agent
pip install -r requirements.txt
cp .env.example .env
# Add your API keys
```

## Quick Start

```bash
# Simple similarity search
python main.py --mode similarity --query "What is RAG?" --index-path ./index

# Hybrid search
python main.py --mode hybrid --query "Compare vector and keyword search" --index-path ./index

# Sub-question decomposition
python main.py --mode sub-question --query "What are the benefits and drawbacks of RAG?" --index-path ./index

# With custom settings
python main.py --mode similarity --query "Explain embeddings" --top-k 5 --temperature 0.7
```

## Configuration Options

```
Query Options:
  --mode              Query mode (similarity, hybrid, sub-question, fusion, tree, router)
  --query             Query text
  --index-path        Path to index directory
  --top-k             Number of results to retrieve (default: 3)
  --similarity-threshold  Minimum similarity score (default: 0.7)
  
Response Options:
  --response-mode     Response synthesis mode (default: compact)
  --streaming         Enable streaming responses
  --temperature       LLM temperature (default: 0.1)
  
Prompt Options:
  --prompt-template   Custom prompt template file
  --system-prompt     System prompt for LLM
  
Advanced Options:
  --mmr              Enable MMR (Maximal Marginal Relevance)
  --query-transform   Apply query transformation
  --verbose          Enable verbose logging
```

## Python API

### Basic Usage

```python
from query_engine_agent import QueryEngineAgent, QueryConfig

# Load index and create agent
agent = QueryEngineAgent(index_path="./index")

# Simple query
response = agent.query("What is RAG?", mode="similarity")
print(response.response)

# With configuration
config = QueryConfig(
    mode="hybrid",
    top_k=5,
    similarity_threshold=0.75
)
response = agent.query("Explain embeddings", config=config)
```

### Advanced Usage

```python
from query_engine_agent import QueryEngineAgent, CustomPromptTemplate

agent = QueryEngineAgent(index_path="./index")

# Custom prompt template
template = CustomPromptTemplate(
    system="You are an expert in AI and machine learning.",
    user="Answer this question: {query}\nContext: {context}"
)

response = agent.query(
    "What is RAG?",
    mode="similarity",
    prompt_template=template,
    top_k=5,
    temperature=0.7
)

# Streaming response
for chunk in agent.query_stream("Explain transformers"):
    print(chunk, end="", flush=True)

# Sub-question decomposition
response = agent.query(
    "Compare the benefits and drawbacks of vector vs keyword search",
    mode="sub-question"
)

# Access sub-questions and answers
for sub_qa in response.source_nodes:
    print(f"Q: {sub_qa.sub_question}")
    print(f"A: {sub_qa.answer}")
```

### Multi-Engine Router

```python
# Create specialized engines for different domains
agent = QueryEngineAgent(index_path="./index")

# Configure router
agent.configure_router({
    "technical": {"index": "./tech_index", "description": "Technical documentation"},
    "business": {"index": "./business_index", "description": "Business content"},
    "general": {"index": "./general_index", "description": "General knowledge"}
})

# Query automatically routes to appropriate engine
response = agent.query("How does authentication work?", mode="router")
print(f"Routed to: {response.metadata['selected_engine']}")
```

## Use Cases

### Complex Question Answering
```python
# Multi-hop reasoning with sub-questions
response = agent.query(
    "What are the key differences between RAG and fine-tuning, "
    "and which is more cost-effective?",
    mode="sub-question"
)
```

### Comparative Analysis
```python
# Fusion retrieval for comprehensive comparison
response = agent.query(
    "Compare vector databases: Pinecone, Weaviate, and Chroma",
    mode="fusion"
)
```

### Interactive Chat
```python
# Maintain conversation context
agent = QueryEngineAgent(index_path="./index", chat_mode=True)

response1 = agent.query("What is RAG?")
response2 = agent.query("How does it differ from fine-tuning?")  # Uses context
response3 = agent.query("Give me an example")  # Continues conversation
```

## Prompt Templates

### Custom Templates

Create `prompts/custom_template.txt`:
```
You are an expert assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say so - don't make up information.

Context:
{context}

Question: {query}

Answer:
```

Use in code:
```python
response = agent.query(
    "What is RAG?",
    prompt_template="prompts/custom_template.txt"
)
```

### Available Template Variables
- `{query}` - User query
- `{context}` - Retrieved context
- `{history}` - Chat history (if chat mode enabled)
- `{custom_var}` - Any custom variables you pass

## Response Synthesis Modes

```python
# Compact (default) - Concatenate context chunks
response = agent.query("What is RAG?", response_mode="compact")

# Refine - Iteratively refine answer
response = agent.query("What is RAG?", response_mode="refine")

# Tree summarize - Build summary tree
response = agent.query("Summarize all content", response_mode="tree_summarize")

# Simple concatenate
response = agent.query("What is RAG?", response_mode="simple_summarize")
```

## Performance Optimization

```python
# Use caching for repeated queries
agent = QueryEngineAgent(index_path="./index", enable_cache=True)

# Adjust chunk size for retrieval
agent = QueryEngineAgent(
    index_path="./index",
    chunk_size=512,
    chunk_overlap=50
)

# Parallel sub-question processing
config = QueryConfig(
    mode="sub-question",
    parallel_processing=True,
    max_workers=4
)
```

## Examples

See `examples/` directory:
- `basic_queries.py` - Simple query examples
- `advanced_retrieval.py` - Advanced retrieval strategies
- `custom_prompts.py` - Custom prompt templates
- `streaming_responses.py` - Streaming implementation
- `multi_engine_router.py` - Router configuration

## Best Practices

1. **Choose the right mode:**
   - Simple queries → `similarity`
   - Keyword-heavy → `hybrid`
   - Complex questions → `sub-question`
   - Maximum coverage → `fusion`

2. **Tune retrieval parameters:**
   - Start with `top_k=3`, increase if needed
   - Use `similarity_threshold` to filter low-quality results
   - Enable MMR to reduce redundancy

3. **Optimize prompts:**
   - Be specific in system prompts
   - Include domain context
   - Provide examples when helpful

4. **Monitor performance:**
   - Track query latency
   - Measure retrieval quality
   - A/B test different configurations

## Troubleshooting

**Low-quality responses:**
- Increase `top_k`
- Lower `similarity_threshold`
- Try `hybrid` mode
- Improve index quality

**Slow queries:**
- Reduce `top_k`
- Disable MMR if not needed
- Use caching
- Optimize index

**Irrelevant results:**
- Increase `similarity_threshold`
- Enable MMR
- Use query transformation
- Refine prompts

## License

Part of LlamaIndex project. See main repository for license.

## Version

Current version: 1.0.0
