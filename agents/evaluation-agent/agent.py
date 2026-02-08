"""Evaluation Agent Core Implementation"""
from typing import List, Dict, Any
import json
from datetime import datetime
from llama_index.core import SimpleDirectoryReader
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

class EvaluationAgent:
    def __init__(self):
        self.faithfulness_evaluator = None
        self.relevancy_evaluator = None
    
    def evaluate(self, query_engine, test_queries: List[Dict], metrics: List[str]) -> Dict[str, float]:
        """Evaluate query engine on test queries"""
        results = {}
        
        # Initialize evaluators
        if 'faithfulness' in metrics:
            self.faithfulness_evaluator = FaithfulnessEvaluator()
        if 'relevance' in metrics or 'relevancy' in metrics:
            self.relevancy_evaluator = RelevancyEvaluator()
        
        # Run queries and evaluate
        faithfulness_scores = []
        relevance_scores = []
        
        for test_query in test_queries:
            query = test_query.get('query', test_query) if isinstance(test_query, dict) else test_query
            
            response = query_engine.query(query)
            
            if 'faithfulness' in metrics and self.faithfulness_evaluator:
                eval_result = self.faithfulness_evaluator.evaluate_response(response=response)
                faithfulness_scores.append(eval_result.score if eval_result.score is not None else 0.0)
            
            if ('relevance' in metrics or 'relevancy' in metrics) and self.relevancy_evaluator:
                eval_result = self.relevancy_evaluator.evaluate_response(
                    query=query,
                    response=response
                )
                relevance_scores.append(eval_result.score if eval_result.score is not None else 0.0)
        
        # Aggregate scores
        if faithfulness_scores:
            results['faithfulness'] = sum(faithfulness_scores) / len(faithfulness_scores)
        if relevance_scores:
            results['relevance'] = sum(relevance_scores) / len(relevance_scores)
        
        # Add overall score
        if results:
            results['overall'] = sum(results.values()) / len(results)
        
        return results
    
    def compare_pipelines(self, pipeline_a, pipeline_b, test_queries: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Compare two pipelines"""
        metrics = ['faithfulness', 'relevance']
        
        results_a = self.evaluate(pipeline_a, test_queries, metrics)
        results_b = self.evaluate(pipeline_b, test_queries, metrics)
        
        comparison = {}
        for metric in metrics:
            if metric in results_a and metric in results_b:
                comparison[metric] = {
                    'a': results_a[metric],
                    'b': results_b[metric]
                }
        
        return comparison
    
    def generate_test_queries(self, data_dir: str, num_queries: int = 50) -> List[Dict[str, str]]:
        """Generate test queries from documents"""
        documents = SimpleDirectoryReader(data_dir).load_data()
        
        # Simplified test query generation
        test_queries = []
        
        for i, doc in enumerate(documents[:num_queries]):
            # Extract first sentence as a simple question generator
            text = doc.text[:200] if len(doc.text) > 200 else doc.text
            test_queries.append({
                'query': f"What is discussed in document {i+1}?",
                'context': text,
                'source': doc.metadata.get('filename', 'unknown')
            })
        
        return test_queries
    
    def generate_report(self, results: Dict[str, float], output_file: str):
        """Generate evaluation report"""
        report = {
            'summary': results,
            'timestamp': str(datetime.now()),
            'details': 'Evaluation complete'
        }
        
        if output_file.endswith('.html'):
            self._generate_html_report(report, output_file)
        else:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
    
    def _generate_html_report(self, report: Dict, output_file: str):
        """Generate HTML report"""
        html = f"""
        <html>
        <head><title>Evaluation Report</title></head>
        <body>
            <h1>RAG Evaluation Report</h1>
            <h2>Summary</h2>
            <ul>
            {''.join(f'<li>{k}: {v:.3f}</li>' for k, v in report['summary'].items())}
            </ul>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html)
