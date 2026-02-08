# LlamaIndex Usage Guide

**Last Updated:** February 2025  
**Target Audience:** Developers, Data Scientists, ML Engineers

This comprehensive guide covers everything you need to know to effectively use LlamaIndex in your applications, from core concepts to advanced features and production best practices.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Data Ingestion Patterns](#data-ingestion-patterns)
3. [Index Types](#index-types)
4. [Query Configurations](#query-configurations)
5. [Advanced Features](#advanced-features)
6. [Best Practices](#best-practices)

---

## Core Concepts

### Overview

LlamaIndex is a data framework that enables LLM applications to interact with external data sources. Understanding the core concepts is essential for building effective applications.

### Documents

**Documents** are the fundamental units of data in LlamaIndex. They represent a piece of text with associated metadata.

```python
from llama_index.core import Document

# Simple document
doc = Document(text="This is my document text.")

# Document with metadata
doc = Document(
    text="LlamaIndex is a data framework for LLM applications.",
    metadata={
        "source": "documentation",
        "author": "LlamaIndex Team",
        "date": "2024-01-15",
        "category": "tutorial"
    }
)

# Document with ID (useful for updates)
doc = Document(
    text="Content that may be updated",
    doc_id="doc_123",
    metadata={"version": "1.0"}
)
```

**Key Properties:**
- `text`: The actual content
- `metadata`: Dictionary of key-value pairs
- `doc_id`: Unique identifier (optional, auto-generated if not provided)
- `embedding`: Vector representation (computed automatically)

**When to use metadata:**
- Filtering documents during retrieval
- Providing context to LLMs
- Tracking document provenance
- Implementing access control

### Nodes

**Nodes** are atomic units of data that represent "chunks" of Documents. LlamaIndex automatically splits Documents into Nodes for efficient retrieval.

```python
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter

# Default node parser (simple splitting)
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents([doc])

# Advanced: Sentence-aware splitting
parser = SentenceSplitter(
    chunk_size=1024,        # Target chunk size in tokens
    chunk_overlap=200,      # Overlap between chunks
    separator=" ",          # Split on spaces
)
nodes = parser.get_nodes_from_documents(documents)

# Custom metadata extraction
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)

extractors = [
    TitleExtractor(nodes=5),  # Extract title from first 5 nodes
    QuestionsAnsweredExtractor(questions=3),  # Generate Q&A pairs
]

parser = SimpleNodeParser.from_defaults(
    metadata_extractors=extractors
)
nodes = parser.get_nodes_from_documents(documents)
```

**Node Relationships:**
- Nodes maintain parent-child relationships
- Preserve document context
- Enable hierarchical retrieval

**Why chunking matters:**
- LLMs have token limits
- Smaller chunks = more precise retrieval
- Larger chunks = more context
- Balance based on use case

### Indices

**Indices** are data structures that enable efficient querying over your documents. They organize Nodes for fast retrieval.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create index (in-memory)
index = VectorStoreIndex.from_documents(documents)

# Create index with custom embeddings
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# Persist index to disk
index.storage_context.persist(persist_dir="./storage")

# Load index from disk
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

**Index types covered in detail below.**

### Query Engines

**Query Engines** are interfaces for asking questions over your indexed data. They handle retrieval, ranking, and response generation.

```python
# Basic query engine
query_engine = index.as_query_engine()
response = query_engine.query("What is LlamaIndex?")
print(response)

# Configure retrieval
query_engine = index.as_query_engine(
    similarity_top_k=5,        # Retrieve top 5 most similar chunks
    response_mode="tree_summarize",  # Summarization strategy
)

# Custom response synthesis
query_engine = index.as_query_engine(
    response_mode="compact",    # Compact similar chunks
    verbose=True,               # Show detailed logs
)

# Access source nodes
response = query_engine.query("Explain RAG")
for node in response.source_nodes:
    print(f"Score: {node.score}")
    print(f"Text: {node.text[:200]}...")
    print(f"Metadata: {node.metadata}")
```

**Response Modes:**
- `refine`: Iteratively refine answer by going through retrieved nodes
- `compact`: Compact retrieved nodes into fewer prompts
- `tree_summarize`: Build tree of summaries
- `simple_summarize`: Concatenate all nodes and summarize
- `no_text`: Return retrieved nodes without synthesis

### Chat Engines

**Chat Engines** enable conversational interactions with your data, maintaining conversation history and context.

```python
# Basic chat engine
chat_engine = index.as_chat_engine()

response = chat_engine.chat("Hello, what can you help me with?")
print(response)

response = chat_engine.chat("Tell me about RAG")
print(response)

# Configure chat mode
chat_engine = index.as_chat_engine(
    chat_mode="condense_question",  # Condense chat history into question
    verbose=True,
)

# Context chat engine (maintains separate context)
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
chat_engine = ContextChatEngine.from_defaults(
    retriever=index.as_retriever(),
    memory=memory,
)
```

**Chat Modes:**
- `simple`: Use context + conversation history directly
- `condense_question`: Condense history into standalone question
- `context`: Retrieve relevant context for each turn
- `condense_plus_context`: Combine both strategies
- `react`: ReAct agent-based chat

---

## Data Ingestion Patterns

### Basic File Loading

```python
from llama_index.core import SimpleDirectoryReader

# Load all files from directory
documents = SimpleDirectoryReader("./data").load_data()

# Load specific file types
documents = SimpleDirectoryReader(
    "./data",
    file_extractor={".pdf": "PDFReader", ".docx": "DocxReader"}
).load_data()

# Recursive loading
documents = SimpleDirectoryReader(
    "./data",
    recursive=True,  # Search subdirectories
    required_exts=[".txt", ".md", ".pdf"]
).load_data()

# Load with file metadata
documents = SimpleDirectoryReader(
    "./data",
    file_metadata=lambda filename: {"source": filename}
).load_data()
```

### Advanced Data Connectors

LlamaIndex supports 300+ data connectors through LlamaHub:

```python
# Database connector
from llama_index.readers.database import DatabaseReader

reader = DatabaseReader(
    sql_database=sql_database,  # SQLAlchemy database
    query="SELECT * FROM documents WHERE category='AI'"
)
documents = reader.load_data()

# Web scraping
from llama_index.readers.web import SimpleWebPageReader

documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["https://example.com/page1", "https://example.com/page2"]
)

# API data
from llama_index.readers.notion import NotionPageReader

reader = NotionPageReader(integration_token=notion_token)
documents = reader.load_data(page_ids=["page_id_1", "page_id_2"])

# GitHub repository
from llama_index.readers.github import GithubRepositoryReader, GithubClient

github_client = GithubClient(github_token=token)
reader = GithubRepositoryReader(
    github_client=github_client,
    owner="run-llama",
    repo="llama_index",
    filter_file_extensions=[".py", ".md"],
)
documents = reader.load_data(branch="main")
```

### Streaming Data Ingestion

For large datasets, use streaming to avoid memory issues:

```python
from llama_index.core import SimpleDirectoryReader

# Lazy loading
reader = SimpleDirectoryReader("./large_data", lazy_load=True)

# Process in batches
batch_size = 100
for docs in reader.iter_data():
    # Process batch
    index.insert_nodes(docs)
```

### Custom Data Transformations

```python
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    KeywordExtractor,
)

# Define transformation pipeline
transformations = [
    # 1. Parse into nodes
    SimpleNodeParser.from_defaults(chunk_size=512),
    
    # 2. Extract metadata
    SummaryExtractor(summaries=["prev", "self"]),
    QuestionsAnsweredExtractor(questions=3),
    KeywordExtractor(keywords=10),
]

# Apply transformations
from llama_index.core.ingestion import IngestionPipeline

pipeline = IngestionPipeline(transformations=transformations)
nodes = pipeline.run(documents=documents)

# Create index from transformed nodes
index = VectorStoreIndex(nodes)
```

### Incremental Updates

```python
# Insert new documents
new_docs = [Document(text="New content", doc_id="doc_new")]
index.insert_nodes(new_docs)

# Update existing document
updated_doc = Document(text="Updated content", doc_id="doc_123")
index.update_ref_doc(updated_doc)

# Delete document
index.delete_ref_doc("doc_123")

# Refresh specific documents
index.refresh_ref_docs([doc1, doc2])
```

---

## Index Types

### VectorStoreIndex

**Use case:** General-purpose semantic search, most common use case

**How it works:** Embeds documents and enables similarity-based retrieval

```python
from llama_index.core import VectorStoreIndex

# In-memory vector store
index = VectorStoreIndex.from_documents(documents)

# With external vector database
from llama_index.vector_stores.pinecone import PineconeVectorStore
import pinecone

pinecone.init(api_key="your-api-key", environment="us-west1-gcp")
pinecone_index = pinecone.Index("llamaindex")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

from llama_index.core import StorageContext
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# Query
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("What is RAG?")
```

**Pros:**
- Fast retrieval (approximate nearest neighbor search)
- Scales to millions of documents
- Semantic similarity matching
- Works with any embedding model

**Cons:**
- Requires embedding computation (cost + time)
- Quality depends on embedding model
- May miss exact keyword matches

**Best for:** Most RAG applications, Q&A systems, semantic search

### TreeIndex

**Use case:** Hierarchical summarization, building summaries of summaries

**How it works:** Builds a tree structure where leaf nodes are document chunks and internal nodes are summaries

```python
from llama_index.core import TreeIndex

index = TreeIndex.from_documents(documents)

# Query with tree traversal
query_engine = index.as_query_engine()
response = query_engine.query("Summarize the main themes")
```

**Pros:**
- Efficient for summarization tasks
- Hierarchical structure preserves relationships
- Good for understanding document collections

**Cons:**
- Slower to build (requires multiple LLM calls)
- More expensive (summarization costs)
- Less suitable for specific fact retrieval

**Best for:** Document summarization, hierarchical analysis, topic extraction

### ListIndex

**Use case:** Simple sequential processing, exhaustive search

**How it works:** Sequential list of nodes, queries each node in order

```python
from llama_index.core import ListIndex

index = ListIndex.from_documents(documents)

# Sequential query (checks all nodes)
query_engine = index.as_query_engine(response_mode="tree_summarize")
response = query_engine.query("What are all the key points?")
```

**Pros:**
- Simple and straightforward
- No embedding required
- Exhaustive (doesn't miss anything)

**Cons:**
- Slow for large datasets
- Expensive (queries all nodes)
- Doesn't scale well

**Best for:** Small datasets, exhaustive analysis, when embeddings aren't available

### KeywordTableIndex

**Use case:** Keyword-based retrieval, exact matching

**How it works:** Extracts keywords from each chunk, retrieves based on keyword matching

```python
from llama_index.core import KeywordTableIndex

index = KeywordTableIndex.from_documents(documents)

# Keyword query
query_engine = index.as_query_engine()
response = query_engine.query("python programming")
```

**Pros:**
- Fast keyword lookup
- No embedding required
- Good for exact term matching
- Complements semantic search

**Cons:**
- Misses semantic similarity
- Sensitive to keyword selection
- Limited to explicit mentions

**Best for:** Technical documentation, code search, exact term lookup

### KnowledgeGraphIndex

**Use case:** Relationship-based queries, connected information

**How it works:** Extracts entities and relationships, builds a knowledge graph

```python
from llama_index.core import KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore

# With Neo4j backend
graph_store = Neo4jGraphStore(
    username="neo4j",
    password="password",
    url="bolt://localhost:7687",
    database="neo4j"
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)
index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=10
)

# Query with graph traversal
query_engine = index.as_query_engine()
response = query_engine.query("How is entity A related to entity B?")
```

**Pros:**
- Captures relationships explicitly
- Enables graph-based reasoning
- Good for entity-centric queries
- Visual graph representation

**Cons:**
- Expensive to build (entity/relation extraction)
- Requires graph database for scale
- More complex setup

**Best for:** Knowledge bases, entity relationships, research papers, business intelligence

### Composable Indices

Combine multiple indices for powerful hybrid approaches:

```python
from llama_index.core import VectorStoreIndex, KeywordTableIndex
from llama_index.core.indices.composability import ComposableGraph

# Create individual indices
vector_index = VectorStoreIndex.from_documents(doc_set_1)
keyword_index = KeywordTableIndex.from_documents(doc_set_2)

# Compose into graph
graph = ComposableGraph.from_indices(
    [vector_index, keyword_index],
    index_summaries=["Vector-based docs", "Keyword-based docs"]
)

# Query across indices
query_engine = graph.as_query_engine()
response = query_engine.query("Query across both index types")
```

---

## Query Configurations

### Retrieval Configuration

```python
# Basic retrieval parameters
query_engine = index.as_query_engine(
    similarity_top_k=5,           # Number of chunks to retrieve
    similarity_cutoff=0.7,        # Minimum similarity score
)

# Advanced retriever
from llama_index.core.retrievers import VectorIndexRetriever

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
    vector_store_query_mode="default"
)

# Custom query engine from retriever
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine(retriever=retriever)
```

### Response Synthesis

```python
# Configure synthesis
from llama_index.core.response_synthesizers import get_response_synthesizer

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
    use_async=True,          # Async LLM calls
    streaming=True,          # Stream response
)

query_engine = index.as_query_engine(
    response_synthesizer=response_synthesizer
)
```

### Filtering and Metadata

```python
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# Filter by metadata
filters = MetadataFilters(
    filters=[
        ExactMatchFilter(key="category", value="AI"),
        ExactMatchFilter(key="author", value="John Doe")
    ]
)

query_engine = index.as_query_engine(filters=filters)
response = query_engine.query("AI content from John Doe")
```

### Reranking

Improve retrieval quality with reranking:

```python
from llama_index.core.postprocessor import SimilarityPostprocessor, LLMRerank

# Similarity-based reranking
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)

query_engine = index.as_query_engine(
    node_postprocessors=[postprocessor],
    similarity_top_k=10  # Retrieve 10, rerank to fewer
)

# LLM-based reranking
reranker = LLMRerank(top_n=3)  # Rerank to top 3

query_engine = index.as_query_engine(
    node_postprocessors=[reranker],
    similarity_top_k=10
)
```

### Streaming Responses

```python
# Streaming query engine
query_engine = index.as_query_engine(streaming=True)

response = query_engine.query("Explain LlamaIndex")

# Stream tokens as they arrive
for token in response.response_gen:
    print(token, end="", flush=True)
```

### Custom Prompts

```python
from llama_index.core import PromptTemplate

# Custom QA prompt
qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context, answer the question: {query_str}\n"
    "If the context doesn't contain the answer, say 'I don't know'.\n"
    "Answer: "
)

query_engine = index.as_query_engine(
    text_qa_template=qa_prompt
)

# Custom refine prompt
refine_prompt = PromptTemplate(
    "The original question is: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)

query_engine = index.as_query_engine(
    refine_template=refine_prompt
)
```

---

## Advanced Features

### Agents

Agents can reason about which tools to use and execute multi-step operations:

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# Create query engine tools
query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_index.as_query_engine(),
        metadata=ToolMetadata(
            name="vector_search",
            description="Useful for semantic search over documents"
        )
    ),
    QueryEngineTool(
        query_engine=keyword_index.as_query_engine(),
        metadata=ToolMetadata(
            name="keyword_search",
            description="Useful for exact keyword matching"
        )
    )
]

# Create agent
agent = ReActAgent.from_tools(
    query_engine_tools,
    verbose=True
)

# Agent reasoning and tool usage
response = agent.chat("Find documents about AI and machine learning")
print(response)
```

### Custom Tools

```python
from llama_index.core.tools import FunctionTool

# Define custom function
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return f"Web results for: {query}"

# Create tools
multiply_tool = FunctionTool.from_defaults(fn=multiply)
search_tool = FunctionTool.from_defaults(fn=search_web)

# Add to agent
agent = ReActAgent.from_tools(
    [multiply_tool, search_tool, *query_engine_tools],
    verbose=True
)

response = agent.chat("What is 5 * 7 and search for AI agents")
```

### Workflows

Workflows enable complex multi-step processing:

```python
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step

class RAGWorkflow(Workflow):
    @step
    async def retrieve(self, ev: StartEvent) -> RetrieveEvent:
        """Retrieve relevant documents."""
        query = ev.query
        nodes = await self.retriever.aretrieve(query)
        return RetrieveEvent(nodes=nodes, query=query)
    
    @step
    async def rerank(self, ev: RetrieveEvent) -> RerankEvent:
        """Rerank retrieved documents."""
        nodes = ev.nodes
        reranked = await self.reranker.arerank(nodes, ev.query)
        return RerankEvent(nodes=reranked, query=ev.query)
    
    @step
    async def synthesize(self, ev: RerankEvent) -> StopEvent:
        """Generate response."""
        response = await self.synthesizer.asynthesize(
            ev.query,
            nodes=ev.nodes
        )
        return StopEvent(result=response)

# Run workflow
workflow = RAGWorkflow(timeout=60)
result = await workflow.run(query="What is RAG?")
```

### Sub-Question Query Engine

Break complex questions into sub-questions:

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

# Multiple query engines for different data sources
tool1 = QueryEngineTool(
    query_engine=index1.as_query_engine(),
    metadata=ToolMetadata(
        name="documentation",
        description="Documentation about feature X"
    )
)

tool2 = QueryEngineTool(
    query_engine=index2.as_query_engine(),
    metadata=ToolMetadata(
        name="api_reference",
        description="API reference for feature X"
    )
)

# Sub-question engine
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[tool1, tool2],
    verbose=True
)

# Automatically breaks down complex query
response = query_engine.query(
    "How do I use feature X? What are the available APIs?"
)
```

### Router Query Engine

Route queries to appropriate index based on content:

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# Define query engines with descriptions
query_engines = {
    "technical": QueryEngineTool(
        query_engine=technical_index.as_query_engine(),
        metadata=ToolMetadata(
            name="technical_docs",
            description="Technical documentation and API references"
        )
    ),
    "user_guide": QueryEngineTool(
        query_engine=guide_index.as_query_engine(),
        metadata=ToolMetadata(
            name="user_guides",
            description="User guides and tutorials"
        )
    )
}

# Router with LLM selector
router = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=list(query_engines.values())
)

# Automatically routes to appropriate engine
response = router.query("How do I install the library?")
```

### Multi-Modal Support

Work with images, audio, and other modalities:

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

# Load documents with images
documents = SimpleDirectoryReader(
    "./data",
    file_extractor={".jpg": "ImageReader", ".png": "ImageReader"}
).load_data()

# Multi-modal LLM
multi_modal_llm = OpenAIMultiModal(model="gpt-4-vision-preview")

# Query with images
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex

index = MultiModalVectorStoreIndex.from_documents(
    documents,
    image_embed_model=embed_model
)

query_engine = index.as_query_engine(
    multi_modal_llm=multi_modal_llm
)

response = query_engine.query("What's in this image?")
```

---

## Best Practices

### 1. Chunking Strategy

**Choose appropriate chunk size:**

```python
# Small chunks (256-512 tokens): Precise retrieval, less context
# Medium chunks (512-1024 tokens): Balanced (recommended for most use cases)
# Large chunks (1024-2048 tokens): More context, less precise

from llama_index.core.node_parser import SentenceSplitter

parser = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=200,  # 20% overlap recommended
)
```

**Use semantic-aware chunking:**

```python
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceWindowNodeParser
)

# Semantic splitting (breaks at semantic boundaries)
parser = SemanticSplitterNodeParser(
    embed_model=embed_model,
    breakpoint_percentile_threshold=95
)

# Sentence window (retrieve sentences with context window)
parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,  # 3 sentences before and after
    window_metadata_key="window",
    original_text_metadata_key="original_sentence"
)
```

### 2. Embedding Selection

**Choose embedding model based on needs:**

```python
# High quality, expensive
from llama_index.embeddings.openai import OpenAIEmbedding
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# Balanced quality and cost (recommended)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Open-source, free, local
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

### 3. Caching

Implement caching to reduce costs and improve performance:

```python
from llama_index.core import set_global_handler

# Enable caching
set_global_handler("simple")

# Custom cache
from llama_index.core.cache import SimpleCache

cache = SimpleCache()

from llama_index.core import Settings
Settings.cache = cache
```

### 4. Error Handling

```python
from llama_index.core.base.base_query_engine import BaseQueryEngine

class RobustQueryEngine:
    def __init__(self, query_engine: BaseQueryEngine):
        self.query_engine = query_engine
    
    def query(self, query_str: str, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                response = self.query_engine.query(query_str)
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error after {max_retries} attempts: {str(e)}"
                time.sleep(2 ** attempt)  # Exponential backoff

# Usage
robust_engine = RobustQueryEngine(index.as_query_engine())
response = robust_engine.query("What is RAG?")
```

### 5. Observability

Monitor your LlamaIndex applications:

```python
# Enable tracing
from llama_index.core import set_global_handler

set_global_handler("arize_phoenix")

# Or use OpenTelemetry
set_global_handler("opentelemetry")

# Custom callbacks
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

debug_handler = LlamaDebugHandler()
callback_manager = CallbackManager([debug_handler])

from llama_index.core import Settings
Settings.callback_manager = callback_manager

# Track metrics
query_engine = index.as_query_engine()
response = query_engine.query("What is LlamaIndex?")

# View event logs
event_pairs = debug_handler.get_llm_inputs_outputs()
for event in event_pairs:
    print(f"Prompt: {event[0]}")
    print(f"Response: {event[1]}")
```

### 6. Rate Limiting

```python
from llama_index.core.llms import OpenAI
import time

class RateLimitedLLM:
    def __init__(self, llm, calls_per_minute=60):
        self.llm = llm
        self.calls_per_minute = calls_per_minute
        self.last_call_time = 0
    
    def complete(self, prompt):
        # Enforce rate limit
        elapsed = time.time() - self.last_call_time
        if elapsed < 60 / self.calls_per_minute:
            time.sleep((60 / self.calls_per_minute) - elapsed)
        
        self.last_call_time = time.time()
        return self.llm.complete(prompt)
```

### 7. Testing and Evaluation

```python
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator
)

# Set up evaluators
faithfulness_evaluator = FaithfulnessEvaluator()
relevancy_evaluator = RelevancyEvaluator()

# Evaluate response
response = query_engine.query("What is RAG?")

faithfulness_result = faithfulness_evaluator.evaluate_response(
    response=response
)
print(f"Faithfulness: {faithfulness_result.passing}")

relevancy_result = relevancy_evaluator.evaluate_response(
    query="What is RAG?",
    response=response
)
print(f"Relevancy: {relevancy_result.passing}")
```

### 8. Security Considerations

```python
# Sanitize inputs
import re

def sanitize_query(query: str) -> str:
    # Remove potential injection attempts
    query = re.sub(r'[<>]', '', query)
    # Limit length
    return query[:500]

# Filter sensitive metadata
def filter_metadata(metadata: dict) -> dict:
    sensitive_keys = ['password', 'api_key', 'secret']
    return {k: v for k, v in metadata.items() if k not in sensitive_keys}

# Use metadata filtering to implement access control
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

def user_query_engine(index, user_id: str):
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="accessible_by", value=user_id)]
    )
    return index.as_query_engine(filters=filters)
```

### 9. Cost Optimization

```python
# Use smaller models for retrieval
from llama_index.llms.openai import OpenAI

cheap_llm = OpenAI(model="gpt-3.5-turbo")  # For summarization
expensive_llm = OpenAI(model="gpt-4")  # For final synthesis

# Reduce chunk count
query_engine = index.as_query_engine(
    similarity_top_k=3  # Start small, increase if needed
)

# Use caching aggressively
from llama_index.core.cache import SimpleCache
Settings.cache = SimpleCache()

# Batch operations
from llama_index.core import VectorStoreIndex

# Build index once, query many times
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist()  # Save for reuse
```

### 10. Production Checklist

Before deploying to production:

- [ ] **Error handling**: Implement retry logic and graceful degradation
- [ ] **Monitoring**: Set up observability and logging
- [ ] **Caching**: Enable caching for repeated queries
- [ ] **Rate limiting**: Implement rate limits for external APIs
- [ ] **Security**: Sanitize inputs, filter sensitive data
- [ ] **Testing**: Write unit and integration tests
- [ ] **Evaluation**: Set up automated evaluation pipelines
- [ ] **Documentation**: Document your prompts and configurations
- [ ] **Backup**: Persist indices and implement backup strategy
- [ ] **Scaling**: Plan for horizontal scaling if needed
- [ ] **Cost monitoring**: Track API usage and costs
- [ ] **Version control**: Version your indices and configurations

---

## Next Steps

1. **Explore Examples**: Check out [EXAMPLES.md](./EXAMPLES.md) for practical implementations
2. **Understand Architecture**: Read [ARCHITECTURE.md](./ARCHITECTURE.md) for system design
3. **Join Community**: Join [Discord](https://discord.gg/dGcwcsnxhU) for support
4. **Build**: Start building your first LlamaIndex application!

## Additional Resources

- [Official Documentation](https://docs.llamaindex.ai/)
- [API Reference](https://docs.llamaindex.ai/en/stable/api_reference/)
- [LlamaHub](https://llamahub.ai/) - Data loaders and tools
- [GitHub Repository](https://github.com/run-llama/llama_index)
- [Blog](https://www.llamaindex.ai/blog)

---

**Questions or feedback?** Open an issue on [GitHub](https://github.com/run-llama/llama_index/issues) or ask in [Discord](https://discord.gg/dGcwcsnxhU)!
