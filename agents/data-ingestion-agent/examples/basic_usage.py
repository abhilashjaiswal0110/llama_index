"""
Basic usage examples for Data Ingestion Agent
"""

from pathlib import Path
from data_ingestion_agent.agent import DataIngestionAgent, IngestionConfig

# Example 1: Simple PDF ingestion
def example_pdf_ingestion():
    """Ingest a single PDF file"""
    agent = DataIngestionAgent()
    documents = agent.ingest_pdf("sample.pdf")
    print(f"Processed {len(documents)} documents")
    
    # Create and save index
    index = agent.create_index(documents)
    agent.save_index(index, "./index")


# Example 2: Directory ingestion with filters
def example_directory_ingestion():
    """Ingest entire directory with filters"""
    config = IngestionConfig(
        chunk_size=512,
        chunk_overlap=100,
        output_dir="./my_index"
    )
    
    agent = DataIngestionAgent(config)
    documents = agent.ingest_directory(
        "./docs",
        recursive=True,
        extensions=["pdf", "md", "txt"]
    )
    
    print(f"Processed {len(documents)} documents")
    metrics = agent.get_metrics()
    print(f"Processing time: {metrics['processing_time']:.2f}s")


# Example 3: Custom preprocessing
def example_custom_preprocessing():
    """Add custom preprocessing steps"""
    agent = DataIngestionAgent()
    
    # Add custom preprocessor
    def add_timestamp(doc):
        from datetime import datetime
        doc.metadata['processed_at'] = datetime.now().isoformat()
        return doc
    
    agent.add_preprocessor(add_timestamp)
    documents = agent.ingest_pdf("document.pdf")


# Example 4: Web scraping
def example_web_scraping():
    """Scrape web content"""
    agent = DataIngestionAgent()
    documents = agent.ingest_web(
        url="https://docs.llamaindex.ai",
        depth=2,
        max_pages=20
    )
    
    index = agent.create_index(documents)
    agent.save_index(index)


# Example 5: Batch processing with progress
def example_batch_processing():
    """Process multiple files with progress tracking"""
    config = IngestionConfig(
        batch_size=32,
        num_workers=4,
        show_progress=True
    )
    
    agent = DataIngestionAgent(config)
    documents = agent.ingest_directory("./large_corpus", recursive=True)
    
    # Get detailed metrics
    metrics = agent.get_metrics()
    print(f"Files: {metrics['files_processed']}")
    print(f"Chunks: {metrics['total_chunks']}")
    print(f"Time: {metrics['processing_time']:.2f}s")


if __name__ == "__main__":
    print("Running Data Ingestion Agent examples...")
    
    # Run examples
    # example_pdf_ingestion()
    # example_directory_ingestion()
    # example_custom_preprocessing()
    # example_web_scraping()
    # example_batch_processing()
