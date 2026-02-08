# LlamaIndex Examples

**Last Updated:** February 2025  
**Target Audience:** Developers, ML Engineers, Data Scientists

This guide provides practical, production-ready examples for building LlamaIndex applications. Each example includes complete code, explanations, and best practices.

## Table of Contents

1. [Basic RAG Pipeline](#basic-rag-pipeline)
2. [Multi-Document Querying](#multi-document-querying)
3. [Chat Engine Implementation](#chat-engine-implementation)
4. [Agent-Based Systems](#agent-based-systems)
5. [Custom Retrievers](#custom-retrievers)
6. [Evaluation Pipelines](#evaluation-pipelines)
7. [Production Deployment Patterns](#production-deployment-patterns)

---

## Basic RAG Pipeline

### Simple RAG Application

The most fundamental LlamaIndex use case: load documents, create an index, and query.

```python
"""
Basic RAG Pipeline
Time to implement: 5 minutes
Use case: Simple Q&A over documents
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os

# Configure API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Configure LLM and embedding model
Settings.llm = OpenAI(model="gpt-4", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Load documents
documents = SimpleDirectoryReader("./data").load_data()
print(f"Loaded {len(documents)} documents")

# Create index
index = VectorStoreIndex.from_documents(documents, show_progress=True)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=3)

# Query
response = query_engine.query("What are the main topics in these documents?")
print(f"\nResponse: {response}")

# View source nodes
print("\nSource Nodes:")
for i, node in enumerate(response.source_nodes, 1):
    print(f"\n{i}. Score: {node.score:.4f}")
    print(f"   Text: {node.text[:200]}...")
    print(f"   Metadata: {node.metadata}")
```

### RAG with Custom Chunking

```python
"""
RAG with Optimized Chunking
Time to implement: 10 minutes
Use case: Better retrieval with custom chunk sizes
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create ingestion pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        # Split into chunks
        SentenceSplitter(
            chunk_size=512,
            chunk_overlap=128,
        ),
        # Extract metadata
        TitleExtractor(llm=Settings.llm, nodes=5),
        QuestionsAnsweredExtractor(llm=Settings.llm, questions=3),
    ]
)

# Process documents
nodes = pipeline.run(documents=documents, show_progress=True)
print(f"Created {len(nodes)} nodes from {len(documents)} documents")

# Create index from nodes
index = VectorStoreIndex(nodes)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("Summarize the key findings")
print(response)
```

### Persisting and Loading Index

```python
"""
Persistent RAG Pipeline
Time to implement: 5 minutes
Use case: Avoid rebuilding index every time
"""

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
import os

PERSIST_DIR = "./storage"

def build_or_load_index():
    """Build index or load from disk if exists."""
    if os.path.exists(PERSIST_DIR):
        # Load existing index
        print("Loading index from disk...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print("Index loaded!")
    else:
        # Build new index
        print("Building new index...")
        documents = SimpleDirectoryReader("./data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        
        # Persist to disk
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print(f"Index built and saved to {PERSIST_DIR}")
    
    return index

# Usage
index = build_or_load_index()
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic?")
print(response)
```

### RAG with External Vector Database

```python
"""
RAG with Pinecone Vector Database
Time to implement: 15 minutes
Use case: Production-scale vector storage
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create index if it doesn't exist
index_name = "llamaindex-demo"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to index
pinecone_index = pc.Index(index_name)

# Create vector store
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load and index documents
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What information is available?")
print(response)

# Later: Load existing index
# vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
# index = VectorStoreIndex.from_vector_store(vector_store)
```

---

## Multi-Document Querying

### Query Multiple Document Sets

```python
"""
Multi-Document Query Engine
Time to implement: 15 minutes
Use case: Query across different document collections
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

# Load different document sets
docs_2023 = SimpleDirectoryReader("./data/2023").load_data()
docs_2024 = SimpleDirectoryReader("./data/2024").load_data()

# Create separate indices
index_2023 = VectorStoreIndex.from_documents(docs_2023)
index_2024 = VectorStoreIndex.from_documents(docs_2024)

# Create query engine tools
query_engine_tools = [
    QueryEngineTool(
        query_engine=index_2023.as_query_engine(),
        metadata=ToolMetadata(
            name="docs_2023",
            description="Documentation and reports from 2023"
        )
    ),
    QueryEngineTool(
        query_engine=index_2024.as_query_engine(),
        metadata=ToolMetadata(
            name="docs_2024",
            description="Documentation and reports from 2024"
        )
    )
]

# Create sub-question query engine
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    verbose=True
)

# Query across both document sets
response = query_engine.query(
    "Compare the key findings between 2023 and 2024"
)
print(response)
```

### Router Query Engine

```python
"""
Router Query Engine
Time to implement: 15 minutes
Use case: Automatically route queries to appropriate index
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# Load different types of documents
technical_docs = SimpleDirectoryReader("./technical").load_data()
user_guides = SimpleDirectoryReader("./guides").load_data()
faq_docs = SimpleDirectoryReader("./faq").load_data()

# Create indices
technical_index = VectorStoreIndex.from_documents(technical_docs)
guide_index = VectorStoreIndex.from_documents(user_guides)
faq_index = VectorStoreIndex.from_documents(faq_docs)

# Create query engine tools with descriptions
query_engine_tools = [
    QueryEngineTool(
        query_engine=technical_index.as_query_engine(),
        metadata=ToolMetadata(
            name="technical_docs",
            description=(
                "Useful for answering questions about technical details, "
                "API references, architecture, and implementation specifics."
            )
        )
    ),
    QueryEngineTool(
        query_engine=guide_index.as_query_engine(),
        metadata=ToolMetadata(
            name="user_guides",
            description=(
                "Useful for answering how-to questions, tutorials, "
                "getting started guides, and step-by-step instructions."
            )
        )
    ),
    QueryEngineTool(
        query_engine=faq_index.as_query_engine(),
        metadata=ToolMetadata(
            name="faq",
            description=(
                "Useful for common questions, troubleshooting, "
                "and frequently encountered issues."
            )
        )
    )
]

# Create router
router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=query_engine_tools,
    verbose=True
)

# The router automatically selects the right index
queries = [
    "How do I install the package?",  # Should route to user_guides
    "What's the API signature for the search function?",  # technical_docs
    "Why am I getting a timeout error?",  # faq
]

for query in queries:
    print(f"\nQuery: {query}")
    response = router_query_engine.query(query)
    print(f"Response: {response}\n")
```

### Hierarchical Document Querying

```python
"""
Hierarchical Document Index
Time to implement: 20 minutes
Use case: Query documents at different levels of abstraction
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create hierarchical node parser
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # Large -> Medium -> Small chunks
)

# Parse into hierarchical nodes
nodes = node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)

# Create index from leaf nodes
index = VectorStoreIndex(leaf_nodes)

# Create auto-merging retriever
retriever = AutoMergingRetriever(
    index.as_retriever(similarity_top_k=6),
    index.storage_context,
    verbose=True
)

# Create query engine
query_engine = RetrieverQueryEngine.from_args(retriever)

# Query - automatically merges retrieved chunks with parent nodes
response = query_engine.query(
    "What are the key implementation details?"
)
print(response)
```

---

## Chat Engine Implementation

### Basic Chat Engine

```python
"""
Basic Chat Engine
Time to implement: 10 minutes
Use case: Conversational Q&A over documents
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer

# Load and index documents
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create chat engine with memory
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    system_prompt=(
        "You are a helpful assistant that answers questions "
        "based on the provided documents. "
        "Always cite your sources and admit when you don't know something."
    ),
    verbose=True
)

# Conversation
print("Chat with your documents (type 'exit' to quit)")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'exit':
        break
    
    response = chat_engine.chat(user_input)
    print(f"\nAssistant: {response}")
```

### Streaming Chat Engine

```python
"""
Streaming Chat Engine
Time to implement: 10 minutes
Use case: Real-time response streaming for better UX
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer

# Setup
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create streaming chat engine
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    streaming=True,
    verbose=True
)

# Streaming conversation
def chat_with_streaming(query: str):
    """Stream chat response token by token."""
    print(f"\nYou: {query}")
    print("Assistant: ", end="", flush=True)
    
    response = chat_engine.stream_chat(query)
    for token in response.response_gen:
        print(token, end="", flush=True)
    print()  # New line after response
    
    return response

# Example conversation
response1 = chat_with_streaming("What are the main topics covered?")
response2 = chat_with_streaming("Can you elaborate on the first topic?")
response3 = chat_with_streaming("How does it relate to the second topic?")
```

### Context-Aware Chat Engine

```python
"""
Context-Aware Chat Engine
Time to implement: 15 minutes
Use case: Chat with document context retrieval per turn
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SimilarityPostprocessor

# Setup
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create retriever with filtering
retriever = index.as_retriever(
    similarity_top_k=5,
)

# Post-processor to filter low-relevance results
node_postprocessors = [
    SimilarityPostprocessor(similarity_cutoff=0.7)
]

# Memory buffer
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# Create context chat engine
chat_engine = ContextChatEngine.from_defaults(
    retriever=retriever,
    memory=memory,
    node_postprocessors=node_postprocessors,
    system_prompt=(
        "You are a knowledgeable assistant. "
        "Use the provided context to answer questions accurately. "
        "If the context doesn't contain relevant information, say so."
    ),
    verbose=True
)

# Chat
conversation_history = []

def chat(message: str):
    """Chat and track conversation."""
    response = chat_engine.chat(message)
    conversation_history.append({
        "message": message,
        "response": str(response),
        "sources": [node.text[:100] for node in response.source_nodes]
    })
    return response

# Example conversation
chat("What is the document about?")
chat("What are the key findings?")
chat("Can you provide more details on the second finding?")

# Review conversation
for i, turn in enumerate(conversation_history, 1):
    print(f"\nTurn {i}:")
    print(f"User: {turn['message']}")
    print(f"Assistant: {turn['response']}")
    print(f"Sources: {len(turn['sources'])} chunks used")
```

---

## Agent-Based Systems

### ReAct Agent with Query Tools

```python
"""
ReAct Agent with Multiple Tools
Time to implement: 20 minutes
Use case: Multi-step reasoning and tool selection
"""

from llama_index.core.agent import ReActAgent
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.llms.openai import OpenAI
import requests

# Create query engine tools
docs = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(docs)

query_tool = QueryEngineTool(
    query_engine=index.as_query_engine(),
    metadata=ToolMetadata(
        name="document_search",
        description="Search through internal documents for information"
    )
)

# Create custom function tools
def search_web(query: str) -> str:
    """Search the web for current information."""
    # Mock implementation - replace with actual search API
    return f"Web search results for: {query}"

def calculate(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

web_search_tool = FunctionTool.from_defaults(fn=search_web)
calculator_tool = FunctionTool.from_defaults(fn=calculate)

# Create ReAct agent
llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(
    [query_tool, web_search_tool, calculator_tool],
    llm=llm,
    verbose=True,
    max_iterations=10
)

# Agent can reason about which tools to use
queries = [
    "What information is in the documents about AI?",
    "Calculate 15% of 250",
    "Search the web for latest AI developments and compare with our docs",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    response = agent.chat(query)
    print(f"\nFinal Answer: {response}")
```

### Function Calling Agent

```python
"""
Function Calling Agent
Time to implement: 20 minutes
Use case: Structured data extraction and API interactions
"""

from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from typing import List, Dict
import json

# Define functions that agent can call
def create_user(name: str, email: str, role: str) -> Dict:
    """Create a new user in the system."""
    # Mock implementation
    user = {
        "id": "user_123",
        "name": name,
        "email": email,
        "role": role,
        "created_at": "2024-01-15"
    }
    return user

def get_users(role: str = None) -> List[Dict]:
    """Get list of users, optionally filtered by role."""
    # Mock implementation
    users = [
        {"id": "user_1", "name": "Alice", "role": "admin"},
        {"id": "user_2", "name": "Bob", "role": "user"},
    ]
    if role:
        users = [u for u in users if u["role"] == role]
    return users

def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email to a user."""
    # Mock implementation
    print(f"Sending email to {to}: {subject}")
    return True

# Create function tools
tools = [
    FunctionTool.from_defaults(fn=create_user),
    FunctionTool.from_defaults(fn=get_users),
    FunctionTool.from_defaults(fn=send_email),
]

# Create agent
llm = OpenAI(model="gpt-4")
agent = FunctionCallingAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True
)

# Agent handles complex multi-step tasks
response = agent.chat(
    "Create a user named John Doe with email john@example.com as an admin, "
    "then get all admin users and send them a welcome email"
)
print(response)
```

### Research Agent

```python
"""
Research Agent
Time to implement: 25 minutes
Use case: Multi-document research and synthesis
"""

from llama_index.core.agent import ReActAgent
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.llms.openai import OpenAI
from typing import List

# Load multiple document sources
research_papers = SimpleDirectoryReader("./papers").load_data()
news_articles = SimpleDirectoryReader("./news").load_data()
technical_docs = SimpleDirectoryReader("./technical").load_data()

# Create specialized indices
papers_index = VectorStoreIndex.from_documents(research_papers)
news_index = VectorStoreIndex.from_documents(news_articles)
tech_index = VectorStoreIndex.from_documents(technical_docs)

# Create query tools
query_tools = [
    QueryEngineTool(
        query_engine=papers_index.as_query_engine(),
        metadata=ToolMetadata(
            name="research_papers",
            description="Academic research papers and studies"
        )
    ),
    QueryEngineTool(
        query_engine=news_index.as_query_engine(),
        metadata=ToolMetadata(
            name="news_articles",
            description="Recent news articles and current events"
        )
    ),
    QueryEngineTool(
        query_engine=tech_index.as_query_engine(),
        metadata=ToolMetadata(
            name="technical_docs",
            description="Technical documentation and specifications"
        )
    ),
]

# Add synthesis tool
def synthesize_findings(findings: List[str]) -> str:
    """Synthesize multiple findings into a coherent summary."""
    synthesis = "\n\n".join([
        f"{i+1}. {finding}"
        for i, finding in enumerate(findings)
    ])
    return f"Synthesized findings:\n{synthesis}"

synthesis_tool = FunctionTool.from_defaults(fn=synthesize_findings)

# Create research agent
llm = OpenAI(model="gpt-4", temperature=0.1)
agent = ReActAgent.from_tools(
    [*query_tools, synthesis_tool],
    llm=llm,
    verbose=True,
    max_iterations=15
)

# Complex research query
research_query = (
    "Research the topic of 'transformer architectures in LLMs'. "
    "1. Find relevant academic papers "
    "2. Check recent news for practical applications "
    "3. Review technical specifications "
    "4. Synthesize findings into a comprehensive summary"
)

response = agent.chat(research_query)
print(f"\n\nResearch Summary:\n{response}")
```

---

## Custom Retrievers

### Hybrid Retriever (Vector + Keyword)

```python
"""
Hybrid Retriever
Time to implement: 20 minutes
Use case: Combine semantic and keyword search
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core.indices.keyword_table import SimpleKeywordTableIndex
from llama_index.core.schema import NodeWithScore
from typing import List

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create both indices
vector_index = VectorStoreIndex.from_documents(documents)
keyword_index = SimpleKeywordTableIndex.from_documents(documents)

# Create retrievers
vector_retriever = VectorIndexRetriever(
    index=vector_index,
    similarity_top_k=10,
)
keyword_retriever = KeywordTableSimpleRetriever(
    index=keyword_index
)

# Custom hybrid retriever
class HybridRetriever:
    def __init__(self, vector_retriever, keyword_retriever, mode="OR"):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.mode = mode  # "OR" or "AND"
    
    def retrieve(self, query: str) -> List[NodeWithScore]:
        # Get results from both retrievers
        vector_nodes = self.vector_retriever.retrieve(query)
        keyword_nodes = self.keyword_retriever.retrieve(query)
        
        # Combine results
        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}
        
        if self.mode == "AND":
            # Intersection: nodes that appear in both
            combined_ids = vector_ids & keyword_ids
        else:  # OR
            # Union: nodes from either retriever
            combined_ids = vector_ids | keyword_ids
        
        # Create node dict for deduplication
        id_to_node = {}
        for node in vector_nodes + keyword_nodes:
            if node.node.node_id in combined_ids:
                # Keep highest score
                if node.node.node_id not in id_to_node or \
                   node.score > id_to_node[node.node.node_id].score:
                    id_to_node[node.node.node_id] = node
        
        # Sort by score
        combined_nodes = sorted(
            id_to_node.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        return combined_nodes

# Use hybrid retriever
hybrid_retriever = HybridRetriever(
    vector_retriever,
    keyword_retriever,
    mode="OR"
)

# Query
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(hybrid_retriever)
response = query_engine.query("machine learning algorithms")
print(response)
```

### Reranking Retriever

```python
"""
Reranking Retriever
Time to implement: 15 minutes
Use case: Improve retrieval quality with reranking
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    LLMRerank,
    SentenceTransformerRerank
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Load and index
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=20,  # Retrieve more, then rerank
)

# Option 1: Similarity-based filtering
similarity_postprocessor = SimilarityPostprocessor(
    similarity_cutoff=0.7
)

# Option 2: LLM-based reranking
llm_reranker = LLMRerank(
    top_n=5,  # Keep top 5 after reranking
    choice_batch_size=10
)

# Option 3: Cross-encoder reranking (most accurate)
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2",
    top_n=5
)

# Create query engine with reranking
query_engine = RetrieverQueryEngine.from_args(
    retriever,
    node_postprocessors=[reranker],
)

# Query
response = query_engine.query("What are the best practices for deployment?")
print(response)

# Show reranking effect
print("\nReranked source nodes:")
for i, node in enumerate(response.source_nodes, 1):
    print(f"{i}. Score: {node.score:.4f}")
    print(f"   Text preview: {node.text[:150]}...\n")
```

### Metadata-Filtered Retriever

```python
"""
Metadata-Filtered Retriever
Time to implement: 15 minutes
Use case: Filter by document metadata (date, category, etc.)
"""

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator
)
from datetime import datetime

# Create documents with rich metadata
documents = [
    Document(
        text="Q1 2024 financial report shows 20% growth",
        metadata={
            "date": "2024-03-31",
            "category": "finance",
            "department": "accounting",
            "quarter": "Q1"
        }
    ),
    Document(
        text="Q2 2024 engineering update on new features",
        metadata={
            "date": "2024-06-30",
            "category": "engineering",
            "department": "product",
            "quarter": "Q2"
        }
    ),
    Document(
        text="Q3 2024 marketing campaign results",
        metadata={
            "date": "2024-09-30",
            "category": "marketing",
            "department": "marketing",
            "quarter": "Q3"
        }
    ),
]

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query with metadata filters
def query_with_filters(query: str, filters: MetadataFilters):
    """Query with metadata filtering."""
    query_engine = index.as_query_engine(filters=filters)
    return query_engine.query(query)

# Example 1: Single filter
finance_filters = MetadataFilters(
    filters=[
        MetadataFilter(key="category", value="finance", operator=FilterOperator.EQ)
    ]
)
response = query_with_filters("What were the results?", finance_filters)
print(f"Finance results: {response}\n")

# Example 2: Multiple filters (AND)
q2_engineering_filters = MetadataFilters(
    filters=[
        MetadataFilter(key="quarter", value="Q2"),
        MetadataFilter(key="category", value="engineering")
    ]
)
response = query_with_filters("What updates are available?", q2_engineering_filters)
print(f"Q2 Engineering: {response}\n")

# Example 3: Date range filter
recent_filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="date",
            value="2024-06-01",
            operator=FilterOperator.GTE  # Greater than or equal
        )
    ]
)
response = query_with_filters("Recent updates?", recent_filters)
print(f"Recent docs: {response}")
```

---

## Evaluation Pipelines

### Retrieval Evaluation

```python
"""
Retrieval Evaluation Pipeline
Time to implement: 25 minutes
Use case: Measure and improve retrieval quality
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.llms.openai import OpenAI
import pandas as pd

# Load and index documents
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create test queries with expected relevant docs
eval_queries = [
    {
        "query": "What is machine learning?",
        "expected_ids": ["doc_1", "doc_3"]  # IDs of relevant documents
    },
    {
        "query": "How does neural network training work?",
        "expected_ids": ["doc_2", "doc_5"]
    },
    # Add more test queries
]

# Create retriever
retriever = index.as_retriever(similarity_top_k=5)

# Create evaluator
evaluator = RetrieverEvaluator.from_metric_names(
    ["hit_rate", "mrr"],  # Mean Reciprocal Rank
    retriever=retriever
)

# Evaluate
results = []
for item in eval_queries:
    query = item["query"]
    expected_ids = item["expected_ids"]
    
    # Evaluate this query
    result = evaluator.evaluate(query, expected_ids=expected_ids)
    results.append({
        "query": query,
        "hit_rate": result.hit_rate,
        "mrr": result.mrr
    })

# Analyze results
df = pd.DataFrame(results)
print("Retrieval Evaluation Results:")
print(df)
print(f"\nAverage Hit Rate: {df['hit_rate'].mean():.3f}")
print(f"Average MRR: {df['mrr'].mean():.3f}")
```

### Response Evaluation

```python
"""
Response Quality Evaluation
Time to implement: 20 minutes
Use case: Evaluate generated responses
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    BatchEvalRunner
)
from llama_index.llms.openai import OpenAI

# Setup
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Create evaluators
llm = OpenAI(model="gpt-4")
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

# Test queries
test_queries = [
    "What is the main topic of the documents?",
    "Explain the key findings in detail",
    "What are the recommendations?",
]

# Evaluate responses
results = []
for query in test_queries:
    # Get response
    response = query_engine.query(query)
    
    # Evaluate faithfulness (response based on context)
    faithfulness_result = faithfulness_evaluator.evaluate_response(
        response=response
    )
    
    # Evaluate relevancy (response relevant to query)
    relevancy_result = relevancy_evaluator.evaluate_response(
        query=query,
        response=response
    )
    
    results.append({
        "query": query,
        "response": str(response)[:100] + "...",
        "faithful": faithfulness_result.passing,
        "relevant": relevancy_result.passing,
        "faithfulness_score": faithfulness_result.score,
        "relevancy_score": relevancy_result.score
    })

# Display results
for i, result in enumerate(results, 1):
    print(f"\nQuery {i}: {result['query']}")
    print(f"Faithful: {'✓' if result['faithful'] else '✗'} "
          f"({result['faithfulness_score']:.2f})")
    print(f"Relevant: {'✓' if result['relevant'] else '✗'} "
          f"({result['relevancy_score']:.2f})")
```

### End-to-End Evaluation

```python
"""
Comprehensive RAG Evaluation
Time to implement: 30 minutes
Use case: Full pipeline evaluation with metrics
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    BatchEvalRunner
)
from llama_index.llms.openai import OpenAI
import pandas as pd
from typing import List, Dict

# Load documents
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create eval dataset
eval_dataset = [
    {
        "query": "What is LlamaIndex?",
        "reference_answer": "LlamaIndex is a data framework for LLM applications."
    },
    {
        "query": "How do you create an index?",
        "reference_answer": "Create an index using VectorStoreIndex.from_documents()"
    },
    # Add more eval examples
]

# Setup evaluators
llm = OpenAI(model="gpt-4")
evaluators = {
    "faithfulness": FaithfulnessEvaluator(llm=llm),
    "relevancy": RelevancyEvaluator(llm=llm),
}

# Run batch evaluation
def evaluate_rag_pipeline(
    index: VectorStoreIndex,
    eval_dataset: List[Dict],
    evaluators: Dict
) -> pd.DataFrame:
    """Evaluate RAG pipeline on dataset."""
    results = []
    
    query_engine = index.as_query_engine()
    
    for item in eval_dataset:
        query = item["query"]
        reference = item.get("reference_answer")
        
        # Get response
        response = query_engine.query(query)
        
        # Evaluate
        eval_results = {
            "query": query,
            "response": str(response),
            "reference": reference,
        }
        
        for eval_name, evaluator in evaluators.items():
            if eval_name == "faithfulness":
                result = evaluator.evaluate_response(response=response)
            elif eval_name == "relevancy":
                result = evaluator.evaluate_response(
                    query=query,
                    response=response
                )
            
            eval_results[f"{eval_name}_score"] = result.score
            eval_results[f"{eval_name}_pass"] = result.passing
        
        results.append(eval_results)
    
    return pd.DataFrame(results)

# Run evaluation
results_df = evaluate_rag_pipeline(index, eval_dataset, evaluators)

# Analyze results
print("Evaluation Results:")
print(results_df[["query", "faithfulness_pass", "relevancy_pass"]])

print("\nSummary Statistics:")
for metric in ["faithfulness_score", "relevancy_score"]:
    print(f"{metric}: {results_df[metric].mean():.3f} ± {results_df[metric].std():.3f}")

# Save results
results_df.to_csv("evaluation_results.csv", index=False)
```

---

## Production Deployment Patterns

### API Server with FastAPI

```python
"""
Production RAG API Server
Time to implement: 30 minutes
Use case: Deploy RAG as REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.response.schema import Response
from typing import List, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="LlamaIndex RAG API", version="1.0")

# Global index
index = None

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    filters: Optional[dict] = None

class QueryResponse(BaseModel):
    response: str
    sources: List[dict]

@app.on_event("startup")
async def load_index():
    """Load index on startup."""
    global index
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        print("Index loaded successfully")
    except Exception as e:
        print(f"Error loading index: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "index_loaded": index is not None
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document index."""
    if index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    try:
        # Create query engine
        query_engine = index.as_query_engine(
            similarity_top_k=request.top_k,
            filters=request.filters
        )
        
        # Query
        response = query_engine.query(request.query)
        
        # Format response
        sources = [
            {
                "text": node.text[:200],
                "score": node.score,
                "metadata": node.metadata
            }
            for node in response.source_nodes
        ]
        
        return QueryResponse(
            response=str(response),
            sources=sources
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """Chat endpoint with conversation history."""
    # Implement chat engine with session management
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Async RAG Pipeline

```python
"""
Async RAG Pipeline
Time to implement: 25 minutes
Use case: High-throughput production system
"""

import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from typing import List, Dict
import time

# Configure for async
Settings.llm = OpenAI(model="gpt-4")

# Load index
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

async def aquery(query_engine, query: str) -> Dict:
    """Async query with timing."""
    start_time = time.time()
    response = await query_engine.aquery(query)
    duration = time.time() - start_time
    
    return {
        "query": query,
        "response": str(response),
        "duration": duration
    }

async def batch_query(queries: List[str]) -> List[Dict]:
    """Process multiple queries concurrently."""
    query_engine = index.as_query_engine()
    
    # Create tasks for all queries
    tasks = [aquery(query_engine, q) for q in queries]
    
    # Run concurrently
    results = await asyncio.gather(*tasks)
    
    return results

# Usage
queries = [
    "What is machine learning?",
    "Explain neural networks",
    "What are transformers?",
    "How does attention work?",
    "What is fine-tuning?"
]

# Run async batch processing
start = time.time()
results = asyncio.run(batch_query(queries))
total_duration = time.time() - start

# Analyze performance
print(f"Processed {len(queries)} queries in {total_duration:.2f}s")
print(f"Average time per query: {total_duration/len(queries):.2f}s")
print(f"Speedup vs sequential: "
      f"{sum(r['duration'] for r in results)/total_duration:.2f}x")

for result in results:
    print(f"\nQuery: {result['query']}")
    print(f"Time: {result['duration']:.2f}s")
    print(f"Response: {result['response'][:100]}...")
```

### Error Handling and Retry Logic

```python
"""
Production Error Handling
Time to implement: 20 minutes
Use case: Robust production deployment
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load index
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Retry decorator for transient failures
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    reraise=True
)
def query_with_retry(query_engine, query: str) -> str:
    """Query with automatic retry on failure."""
    try:
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise

# Graceful degradation
class RobustRAGSystem:
    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self.fallback_responses = {
            "error": "I apologize, but I'm having trouble processing your request.",
            "timeout": "The query is taking longer than expected. Please try again.",
            "no_results": "I couldn't find relevant information for your query."
        }
    
    def query(self, query_str: str, timeout: int = 30) -> Dict:
        """Query with comprehensive error handling."""
        query_engine = self.index.as_query_engine()
        
        try:
            # Try normal query with retry
            response = query_with_retry(query_engine, query_str)
            
            # Check if response is meaningful
            if len(response.strip()) < 10:
                logger.warning("Response too short, may be low quality")
                return {
                    "status": "warning",
                    "response": response,
                    "message": "Response may be incomplete"
                }
            
            return {
                "status": "success",
                "response": response
            }
        
        except TimeoutError:
            logger.error(f"Query timeout: {query_str}")
            return {
                "status": "error",
                "response": self.fallback_responses["timeout"],
                "error_type": "timeout"
            }
        
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {
                "status": "error",
                "response": self.fallback_responses["error"],
                "error_type": type(e).__name__,
                "error_message": str(e)
            }

# Usage
rag_system = RobustRAGSystem(index)

# Test with various queries
test_queries = [
    "What is the main topic?",
    "Very specific technical detail that may not exist",
    "Normal query about content",
]

for query in test_queries:
    result = rag_system.query(query)
    print(f"\nQuery: {query}")
    print(f"Status: {result['status']}")
    print(f"Response: {result['response'][:200]}...")
```

### Monitoring and Observability

```python
"""
RAG System Monitoring
Time to implement: 25 minutes
Use case: Production monitoring and debugging
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events import BaseEvent
from dataclasses import dataclass
from typing import List, Dict
import time
import json

@dataclass
class QueryMetrics:
    query: str
    response_time: float
    num_chunks_retrieved: int
    total_tokens: int
    success: bool
    error: Optional[str] = None

class RAGMonitor:
    def __init__(self):
        self.metrics: List[QueryMetrics] = []
        self.debug_handler = LlamaDebugHandler()
        
        # Setup callback manager
        callback_manager = CallbackManager([self.debug_handler])
        Settings.callback_manager = callback_manager
    
    def log_query(self, metrics: QueryMetrics):
        """Log query metrics."""
        self.metrics.append(metrics)
    
    def get_stats(self) -> Dict:
        """Get aggregate statistics."""
        if not self.metrics:
            return {}
        
        successful = [m for m in self.metrics if m.success]
        
        return {
            "total_queries": len(self.metrics),
            "successful_queries": len(successful),
            "failed_queries": len(self.metrics) - len(successful),
            "avg_response_time": sum(m.response_time for m in successful) / len(successful) if successful else 0,
            "avg_chunks_retrieved": sum(m.num_chunks_retrieved for m in successful) / len(successful) if successful else 0,
            "total_tokens": sum(m.total_tokens for m in self.metrics),
        }
    
    def export_metrics(self, filename: str = "metrics.json"):
        """Export metrics to file."""
        with open(filename, 'w') as f:
            json.dump([vars(m) for m in self.metrics], f, indent=2)

# Setup
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
monitor = RAGMonitor()

def monitored_query(query: str) -> str:
    """Execute query with monitoring."""
    query_engine = index.as_query_engine()
    
    start_time = time.time()
    try:
        response = query_engine.query(query)
        duration = time.time() - start_time
        
        # Get LLM events
        llm_events = monitor.debug_handler.get_llm_inputs_outputs()
        
        # Calculate tokens (approximate)
        total_tokens = sum(
            len(event[0].split()) + len(event[1].split())
            for event in llm_events
        )
        
        # Log metrics
        metrics = QueryMetrics(
            query=query,
            response_time=duration,
            num_chunks_retrieved=len(response.source_nodes),
            total_tokens=total_tokens,
            success=True
        )
        monitor.log_query(metrics)
        
        return str(response)
    
    except Exception as e:
        duration = time.time() - start_time
        metrics = QueryMetrics(
            query=query,
            response_time=duration,
            num_chunks_retrieved=0,
            total_tokens=0,
            success=False,
            error=str(e)
        )
        monitor.log_query(metrics)
        raise

# Run queries
queries = [
    "What is the main topic?",
    "Explain the key findings",
    "What are the recommendations?",
]

for query in queries:
    try:
        response = monitored_query(query)
        print(f"Query: {query}\nResponse: {response[:100]}...\n")
    except Exception as e:
        print(f"Query failed: {e}\n")

# Print statistics
print("\nSystem Statistics:")
stats = monitor.get_stats()
for key, value in stats.items():
    print(f"{key}: {value}")

# Export metrics
monitor.export_metrics()
print("\nMetrics exported to metrics.json")
```

---

## Additional Resources

- **[USAGE_GUIDE.md](./USAGE_GUIDE.md)** - Comprehensive usage documentation
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - System architecture guide
- **[Official Docs](https://docs.llamaindex.ai/)** - Complete documentation
- **[LlamaHub](https://llamahub.ai/)** - Data loaders and tools
- **[Discord Community](https://discord.gg/dGcwcsnxhU)** - Get help and share

## Next Steps

1. **Start Small**: Begin with the basic RAG pipeline
2. **Iterate**: Add features based on your needs
3. **Evaluate**: Implement evaluation pipelines early
4. **Scale**: Use production patterns when ready
5. **Monitor**: Always monitor in production

---

**Questions?** Open an issue on [GitHub](https://github.com/run-llama/llama_index/issues) or ask in [Discord](https://discord.gg/dGcwcsnxhU)!
