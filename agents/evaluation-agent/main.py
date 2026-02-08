"""Evaluation Agent - Main CLI"""
import click
import json
from rich.console import Console
from rich.table import Table
from agent import EvaluationAgent

console = Console()

@click.group()
def cli():
    """Evaluation Agent for RAG quality metrics"""
    pass

@cli.command()
@click.option('--pipeline-path', required=True, help='Pipeline directory')
@click.option('--test-set', required=True, help='Test queries JSON file')
@click.option('--metrics', default='faithfulness,relevance', help='Metrics to compute')
@click.option('--output', default='results.json', help='Output file')
def evaluate(pipeline_path, test_set, metrics, output):
    """Evaluate pipeline on test set"""
    agent = EvaluationAgent()
    
    # Load pipeline
    from llama_index.core import load_index_from_storage, StorageContext
    storage = StorageContext.from_defaults(persist_dir=pipeline_path)
    index = load_index_from_storage(storage)
    query_engine = index.as_query_engine()
    
    # Load test queries
    with open(test_set, 'r') as f:
        test_queries = json.load(f)
    
    # Run evaluation
    results = agent.evaluate(query_engine, test_queries, metrics.split(','))
    
    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")
    
    for metric, score in results.items():
        table.add_row(metric, f"{score:.3f}")
    
    console.print(table)
    
    # Save results
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n✅ Results saved to {output}")

@cli.command()
@click.option('--pipeline-a', required=True, help='First pipeline directory')
@click.option('--pipeline-b', required=True, help='Second pipeline directory')
@click.option('--test-set', required=True, help='Test queries JSON')
def compare(pipeline_a, pipeline_b, test_set):
    """Compare two pipelines"""
    agent = EvaluationAgent()
    
    # Load pipelines
    from llama_index.core import load_index_from_storage, StorageContext
    
    storage_a = StorageContext.from_defaults(persist_dir=pipeline_a)
    index_a = load_index_from_storage(storage_a)
    
    storage_b = StorageContext.from_defaults(persist_dir=pipeline_b)
    index_b = load_index_from_storage(storage_b)
    
    # Load test queries
    with open(test_set, 'r') as f:
        test_queries = json.load(f)
    
    # Compare
    comparison = agent.compare_pipelines(
        index_a.as_query_engine(),
        index_b.as_query_engine(),
        test_queries
    )
    
    # Display comparison
    table = Table(title="Pipeline Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Pipeline A", style="yellow")
    table.add_column("Pipeline B", style="yellow")
    table.add_column("Improvement", style="green")
    
    for metric, values in comparison.items():
        a = values['a']
        b = values['b']
        if a == 0:
            improvement_str = "N/A"
        else:
            improvement = ((b - a) / a) * 100
            improvement_str = f"{improvement:+.1f}%"
        table.add_row(
            metric,
            f"{a:.3f}",
            f"{b:.3f}",
            improvement_str
        )
    
    console.print(table)

@cli.command()
@click.option('--data-dir', required=True, help='Data directory')
@click.option('--output', default='test_queries.json', help='Output file')
@click.option('--num-queries', type=int, default=50, help='Number of queries to generate')
def generate_tests(data_dir, output, num_queries):
    """Generate test queries from documents"""
    agent = EvaluationAgent()
    
    console.print(f"Generating {num_queries} test queries from {data_dir}...")
    
    test_queries = agent.generate_test_queries(data_dir, num_queries)
    
    with open(output, 'w') as f:
        json.dump(test_queries, f, indent=2)
    
    console.print(f"✅ Generated {len(test_queries)} queries and saved to {output}")

if __name__ == '__main__':
    cli()
