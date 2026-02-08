"""RAG Pipeline Agent - Main CLI"""
import click
from rich.console import Console
from agent import RAGPipelineAgent

console = Console()

@click.group()
def cli():
    """RAG Pipeline Agent for end-to-end RAG system building"""
    pass

@cli.command()
@click.option('--data-dir', required=True, help='Data directory')
@click.option('--config', type=click.Path(exists=True), help='Pipeline config')
@click.option('--output', default='./pipeline', help='Output directory')
def build(data_dir, config, output):
    """Build a complete RAG pipeline"""
    agent = RAGPipelineAgent()
    pipeline = agent.build_pipeline(data_dir=data_dir, config_file=config)
    agent.save_pipeline(pipeline, output)
    console.print(f"✅ Pipeline built and saved to {output}", style="bold green")

@cli.command()
@click.option('--pipeline-path', required=True, help='Pipeline directory')
@click.option('--test-queries', required=True, help='Test queries JSON')
def optimize(pipeline_path, test_queries):
    """Optimize pipeline parameters"""
    agent = RAGPipelineAgent()
    pipeline = agent.load_pipeline(pipeline_path)
    optimized = agent.optimize(pipeline, test_queries)
    agent.save_pipeline(optimized, pipeline_path)
    console.print("✅ Pipeline optimized", style="bold green")

@cli.command()
@click.option('--pipeline-path', required=True, help='Pipeline directory')
@click.option('--test-set', required=True, help='Test set JSON')
def evaluate(pipeline_path, test_set):
    """Evaluate pipeline performance"""
    agent = RAGPipelineAgent()
    pipeline = agent.load_pipeline(pipeline_path)
    results = agent.evaluate(pipeline, test_set)
    console.print(f"\n📊 Evaluation Results:", style="bold")
    for metric, score in results.items():
        console.print(f"  {metric}: {score:.3f}")

@cli.command()
@click.option('--pipeline-path', required=True, help='Pipeline directory')
@click.option('--target', type=click.Choice(['docker', 'kubernetes', 'fastapi']), required=True)
@click.option('--output', default='./deployment', help='Output directory')
def deploy(pipeline_path, target, output):
    """Generate deployment artifacts"""
    agent = RAGPipelineAgent()
    pipeline = agent.load_pipeline(pipeline_path)
    agent.deploy(pipeline, target, output)
    console.print(f"✅ Deployment files generated in {output}", style="bold green")

if __name__ == '__main__':
    cli()
