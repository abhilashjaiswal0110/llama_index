"""
Data Ingestion Agent - Main CLI Interface

Enterprise-grade data ingestion tool for LlamaIndex.
Supports multiple data sources, intelligent chunking, and metadata extraction.
"""

import click
import logging
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from agent import DataIngestionAgent, IngestionConfig
from utils import setup_logging, load_config, validate_path

console = Console()


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-file', type=str, default='ingestion.log', help='Log file path')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, verbose: bool, log_file: str, config: Optional[str]):
    """Data Ingestion Agent for LlamaIndex
    
    Load and process data from multiple sources into LlamaIndex format.
    """
    ctx.ensure_object(dict)
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(log_level, log_file)
    
    # Load configuration
    if config:
        ctx.obj['config'] = load_config(config)
    else:
        ctx.obj['config'] = IngestionConfig()
    
    ctx.obj['verbose'] = verbose


@cli.command()
@click.option('--path', '-p', required=True, type=click.Path(exists=True), help='PDF file path')
@click.option('--output-dir', '-o', type=str, default='./index', help='Output directory')
@click.option('--chunk-size', type=int, default=1024, help='Chunk size in tokens')
@click.option('--chunk-overlap', type=int, default=200, help='Chunk overlap')
@click.option('--extract-images', is_flag=True, help='Extract and describe images')
@click.option('--ocr', is_flag=True, help='Use OCR for scanned PDFs')
@click.pass_context
def pdf(ctx, path: str, output_dir: str, chunk_size: int, chunk_overlap: int,
        extract_images: bool, ocr: bool):
    """Process PDF documents"""
    config = ctx.obj['config']
    config.chunk_size = chunk_size
    config.chunk_overlap = chunk_overlap
    config.output_dir = output_dir
    
    agent = DataIngestionAgent(config)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing PDF...", total=None)
            
            documents = agent.ingest_pdf(
                path,
                extract_images=extract_images,
                ocr_enabled=ocr
            )
            
            progress.update(task, completed=True)
        
        console.print(f"✅ Successfully processed {len(documents)} documents", style="bold green")
        console.print(f"📊 Total chunks: {sum(len(doc.text.split()) for doc in documents)}")
        console.print(f"💾 Output: {output_dir}")
        
    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="bold red")
        logging.error(f"PDF processing failed: {e}", exc_info=True)
        raise click.Abort()


@cli.command()
@click.option('--path', '-p', required=True, type=click.Path(exists=True), help='Directory path')
@click.option('--output-dir', '-o', type=str, default='./index', help='Output directory')
@click.option('--recursive', '-r', is_flag=True, help='Recurse into subdirectories')
@click.option('--extensions', type=str, help='File extensions (comma-separated)')
@click.option('--exclude', type=str, help='Exclude patterns (comma-separated)')
@click.option('--chunk-size', type=int, default=1024, help='Chunk size in tokens')
@click.option('--chunk-overlap', type=int, default=200, help='Chunk overlap')
@click.pass_context
def directory(ctx, path: str, output_dir: str, recursive: bool, extensions: Optional[str],
              exclude: Optional[str], chunk_size: int, chunk_overlap: int):
    """Process entire directory of documents"""
    config = ctx.obj['config']
    config.chunk_size = chunk_size
    config.chunk_overlap = chunk_overlap
    config.output_dir = output_dir
    
    agent = DataIngestionAgent(config)
    
    # Parse extensions and exclude patterns
    ext_list = extensions.split(',') if extensions else None
    exclude_list = exclude.split(',') if exclude else None
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing directory...", total=None)
            
            documents = agent.ingest_directory(
                path,
                recursive=recursive,
                extensions=ext_list,
                exclude_patterns=exclude_list
            )
            
            progress.update(task, completed=True)
        
        console.print(f"✅ Successfully processed {len(documents)} documents", style="bold green")
        
        # Show statistics
        metrics = agent.get_metrics()
        console.print("\n📊 Statistics:")
        console.print(f"  • Files processed: {metrics.get('files_processed', 0)}")
        console.print(f"  • Total chunks: {metrics.get('total_chunks', 0)}")
        console.print(f"  • Processing time: {metrics.get('processing_time', 0):.2f}s")
        console.print(f"  • Output directory: {output_dir}")
        
    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="bold red")
        logging.error(f"Directory processing failed: {e}", exc_info=True)
        raise click.Abort()


@cli.command()
@click.option('--url', '-u', required=True, type=str, help='Starting URL')
@click.option('--output-dir', '-o', type=str, default='./index', help='Output directory')
@click.option('--depth', type=int, default=1, help='Crawl depth')
@click.option('--max-pages', type=int, default=50, help='Maximum pages to crawl')
@click.option('--include-pattern', type=str, help='URL pattern to include')
@click.option('--exclude-pattern', type=str, help='URL pattern to exclude')
@click.option('--chunk-size', type=int, default=1024, help='Chunk size in tokens')
@click.pass_context
def web(ctx, url: str, output_dir: str, depth: int, max_pages: int,
        include_pattern: Optional[str], exclude_pattern: Optional[str], chunk_size: int):
    """Scrape and process web content"""
    config = ctx.obj['config']
    config.chunk_size = chunk_size
    config.output_dir = output_dir
    
    agent = DataIngestionAgent(config)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Scraping {url}...", total=None)
            
            documents = agent.ingest_web(
                url=url,
                depth=depth,
                max_pages=max_pages,
                include_pattern=include_pattern,
                exclude_pattern=exclude_pattern
            )
            
            progress.update(task, completed=True)
        
        console.print(f"✅ Successfully scraped {len(documents)} pages", style="bold green")
        console.print(f"💾 Output: {output_dir}")
        
    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="bold red")
        logging.error(f"Web scraping failed: {e}", exc_info=True)
        raise click.Abort()


@cli.command()
@click.option('--endpoint', '-e', required=True, type=str, help='API endpoint URL')
@click.option('--output-dir', '-o', type=str, default='./index', help='Output directory')
@click.option('--headers', type=str, help='HTTP headers (JSON format)')
@click.option('--params', type=str, help='Query parameters (JSON format)')
@click.option('--text-field', type=str, required=True, help='Field containing text')
@click.option('--metadata-fields', type=str, help='Metadata fields (comma-separated)')
@click.option('--pagination', is_flag=True, help='Enable pagination')
@click.pass_context
def api(ctx, endpoint: str, output_dir: str, headers: Optional[str], params: Optional[str],
        text_field: str, metadata_fields: Optional[str], pagination: bool):
    """Ingest data from REST API"""
    import json
    
    config = ctx.obj['config']
    config.output_dir = output_dir
    
    agent = DataIngestionAgent(config)
    
    # Parse headers and params
    headers_dict = json.loads(headers) if headers else {}
    params_dict = json.loads(params) if params else {}
    metadata_list = metadata_fields.split(',') if metadata_fields else []
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching from API...", total=None)
            
            documents = agent.ingest_api(
                endpoint=endpoint,
                headers=headers_dict,
                params=params_dict,
                text_field=text_field,
                metadata_fields=metadata_list,
                pagination=pagination
            )
            
            progress.update(task, completed=True)
        
        console.print(f"✅ Successfully ingested {len(documents)} records", style="bold green")
        console.print(f"💾 Output: {output_dir}")
        
    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="bold red")
        logging.error(f"API ingestion failed: {e}", exc_info=True)
        raise click.Abort()


@cli.command()
@click.option('--connection', '-c', required=True, type=str, help='Database connection string')
@click.option('--output-dir', '-o', type=str, default='./index', help='Output directory')
@click.option('--query', '-q', required=True, type=str, help='SQL query to execute')
@click.option('--text-columns', required=True, type=str, help='Text columns (comma-separated)')
@click.option('--metadata-columns', type=str, help='Metadata columns (comma-separated)')
@click.option('--batch-size', type=int, default=1000, help='Batch size for fetching')
@click.pass_context
def database(ctx, connection: str, output_dir: str, query: str, text_columns: str,
             metadata_columns: Optional[str], batch_size: int):
    """Ingest data from database"""
    config = ctx.obj['config']
    config.output_dir = output_dir
    
    agent = DataIngestionAgent(config)
    
    text_cols = text_columns.split(',')
    metadata_cols = metadata_columns.split(',') if metadata_columns else []
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Querying database...", total=None)
            
            documents = agent.ingest_database(
                connection=connection,
                query=query,
                text_columns=text_cols,
                metadata_columns=metadata_cols,
                batch_size=batch_size
            )
            
            progress.update(task, completed=True)
        
        console.print(f"✅ Successfully ingested {len(documents)} records", style="bold green")
        console.print(f"💾 Output: {output_dir}")
        
    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="bold red")
        logging.error(f"Database ingestion failed: {e}", exc_info=True)
        raise click.Abort()


@cli.command()
@click.pass_context
def status(ctx):
    """Show ingestion status and metrics"""
    config = ctx.obj['config']
    agent = DataIngestionAgent(config)
    
    metrics = agent.get_metrics()
    
    console.print("\n📊 Data Ingestion Agent Status\n", style="bold")
    console.print(f"Configuration:")
    console.print(f"  • Output directory: {config.output_dir}")
    console.print(f"  • Chunk size: {config.chunk_size}")
    console.print(f"  • Chunk overlap: {config.chunk_overlap}")
    console.print(f"  • Chunking strategy: {config.chunking_strategy}")
    
    if metrics:
        console.print(f"\nMetrics:")
        console.print(f"  • Files processed: {metrics.get('files_processed', 0)}")
        console.print(f"  • Total chunks: {metrics.get('total_chunks', 0)}")
        console.print(f"  • Processing time: {metrics.get('processing_time', 0):.2f}s")


if __name__ == '__main__':
    cli(obj={})
