"""Query Engine Agent - Main CLI"""
import click
from rich.console import Console
from agent import QueryEngineAgent, QueryConfig

console = Console()

@click.group()
def cli():
    """Query Engine Agent for advanced LlamaIndex querying"""
    pass

@cli.command()
@click.option('--mode', type=click.Choice(['similarity', 'hybrid', 'sub-question', 'fusion', 'tree', 'router']), default='similarity')
@click.option('--query', '-q', required=True, help='Query text')
@click.option('--index-path', required=True, help='Path to index')
@click.option('--top-k', type=int, default=3, help='Number of results')
@click.option('--streaming', is_flag=True, help='Enable streaming')
def query(mode, query, index_path, top_k, streaming):
    """Execute a query"""
    config = QueryConfig(mode=mode, top_k=top_k)
    agent = QueryEngineAgent(index_path=index_path)
    
    if streaming:
        for chunk in agent.query_stream(query, config):
            console.print(chunk, end="")
    else:
        response = agent.query(query, config)
        console.print(f"\n[bold green]Answer:[/bold green]\n{response.response}")
        console.print(f"\n[dim]Sources: {len(response.source_nodes)} chunks retrieved[/dim]")

if __name__ == '__main__':
    cli()
