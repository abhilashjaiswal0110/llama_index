"""Indexing Agent - Main CLI"""
import click
from rich.console import Console
from agent import IndexingAgent

console = Console()

@click.group()
def cli():
    """Indexing Agent for LlamaIndex"""
    pass

@cli.command()
@click.option('--data-dir', required=True, help='Data directory')
@click.option('--index-type', default='vector', help='Index type')
@click.option('--output', default='./index', help='Output directory')
@click.option('--optimize', is_flag=True, help='Optimize index')
def create(data_dir, index_type, output, optimize):
    """Create index"""
    agent = IndexingAgent()
    index = agent.create_index(data_dir, index_type)
    
    if optimize:
        index = agent.optimize_index(index)
    
    agent.save_index(index, output)
    console.print(f"✅ Index created: {output}", style="bold green")

@cli.command()
@click.option('--data-dir', required=True, help='Data directory')
@click.option('--types', default='vector,tree,keyword', help='Index types to compare')
def compare(data_dir, types):
    """Compare index types"""
    agent = IndexingAgent()
    results = agent.compare_indexes(data_dir, types.split(','))
    
    console.print("\n📊 Index Comparison:", style="bold")
    for idx_type, metrics in results.items():
        console.print(f"\n{idx_type}:")
        for metric, value in metrics.items():
            console.print(f"  {metric}: {value}")

if __name__ == '__main__':
    cli()
