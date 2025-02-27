import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from src.product_ner import ProductNER

class ModelEvaluator:
    """Class for evaluating and visualizing NER model performance."""
    
    def __init__(self, model_path=None, test_data_path=None):
        """
        Initialize the ModelEvaluator.
        
        Args:
            model_path: Path to the trained spaCy model
            test_data_path: Path to the test data in spaCy binary format
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.ner = ProductNER()
        self.evaluation_results = None
        
        # Load model if provided
        if model_path:
            self.ner.load_model(model_path)
    
    def evaluate(self, model_path=None, test_data_path=None):
        """
        Evaluate the NER model and store results.
        
        Args:
            model_path: Path to the trained model (optional if already set)
            test_data_path: Path to the test data (optional if already set)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Update paths if provided
        if model_path:
            self.model_path = model_path
        if test_data_path:
            self.test_data_path = test_data_path
            
        # Validate paths
        if not self.model_path or not self.test_data_path:
            raise ValueError("Model path and test data path must be provided")
            
        # Evaluate the model
        self.evaluation_results = self.ner.evaluate_model(
            test_path=self.test_data_path,
            model_path=self.model_path
        )
        
        return self.evaluation_results
    
    def save_results(self, output_path="./evaluation"):
        """
        Save evaluation results to JSON file.
        
        Args:
            output_path: Directory to save evaluation results
        
        Returns:
            Path to the saved file
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate() first.")
            
        # Create output directory if it doesn't exist
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
        file_path = os.path.join(output_path, filename)
        
        # Save results to JSON
        with open(file_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=4)
            
        return file_path
    
    def plot_overall_metrics(self, output_path=None):
        """
        Plot overall precision, recall, and F-score.
        
        Args:
            output_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate() first.")
            
        # Extract overall metrics
        metrics = self.evaluation_results["overall"]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F-score'],
            'Value': [metrics['precision'], metrics['recall'], metrics['f_score']]
        })
        
        sns.barplot(x='Metric', y='Value', data=metrics_df, ax=ax)
        ax.set_ylim(0, 1.0)
        ax.set_title('Overall NER Model Performance')
        ax.set_ylabel('Score')
        
        # Add value labels on top of bars
        for i, v in enumerate(metrics_df['Value']):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
            
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_entity_metrics(self, output_path=None):
        """
        Plot precision, recall, and F-score for each entity type.
        
        Args:
            output_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate() first.")
            
        # Extract entity metrics
        entity_metrics = self.evaluation_results["per_entity"]
        
        # Create dataframe for plotting
        data = []
        for entity, metrics in entity_metrics.items():
            data.append({
                'Entity': entity,
                'Metric': 'Precision',
                'Value': metrics['p']
            })
            data.append({
                'Entity': entity,
                'Metric': 'Recall',
                'Value': metrics['r']
            })
            data.append({
                'Entity': entity,
                'Metric': 'F-score',
                'Value': metrics['f']
            })
            
        metrics_df = pd.DataFrame(data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot metrics
        sns.barplot(x='Entity', y='Value', hue='Metric', data=metrics_df, ax=ax)
        ax.set_ylim(0, 1.0)
        ax.set_title('NER Performance by Entity Type')
        ax.set_ylabel('Score')
        ax.legend(title='Metric')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def confusion_matrix(self, output_path=None):
        """
        Generate a confusion matrix for entity predictions.
        This requires running additional evaluation on the test data.
        
        Args:
            output_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # This would require additional processing of test examples
        # to track predicted vs. actual entity labels
        # For a complete implementation, you would need to:
        # 1. Load test examples
        # 2. Run predictions
        # 3. Compare predicted entities with gold standard
        # 4. Create and plot confusion matrix
        
        # Placeholder for future implementation
        raise NotImplementedError("Confusion matrix visualization not yet implemented")
    
    def sample_predictions(self, num_samples=5):
        """
        Show sample predictions on test data to qualitatively evaluate the model.
        
        Args:
            num_samples: Number of samples to show
            
        Returns:
            List of dictionaries with text and predictions
        """
        # This would require loading and processing test examples
        # For a complete implementation, you would need to:
        # 1. Load test examples
        # 2. Run predictions
        # 3. Format and return results
        
        # Placeholder for future implementation
        raise NotImplementedError("Sample predictions visualization not yet implemented")
    
    def generate_report(self, output_dir="./evaluation"):
        """
        Generate a comprehensive evaluation report with metrics and visualizations.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Path to the report directory
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save evaluation results
        results_path = self.save_results(output_dir)
        
        # Generate and save plots
        overall_plot_path = os.path.join(output_dir, "overall_metrics.png")
        self.plot_overall_metrics(overall_plot_path)
        
        entity_plot_path = os.path.join(output_dir, "entity_metrics.png")
        self.plot_entity_metrics(entity_plot_path)
        
        # Create HTML report
        html_path = os.path.join(output_dir, "evaluation_report.html")
        
        with open(html_path, 'w') as f:
            f.write(f"""
            <html>
            <head>
                <title>NER Model Evaluation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    .metrics {{ margin: 20px 0; }}
                    .plot {{ margin: 30px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <h1>NER Model Evaluation Report</h1>
                
                <div class="metrics">
                    <h2>Overall Metrics</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Precision</td>
                            <td>{self.evaluation_results["overall"]["precision"]:.4f}</td>
                        </tr>
                        <tr>
                            <td>Recall</td>
                            <td>{self.evaluation_results["overall"]["recall"]:.4f}</td>
                        </tr>
                        <tr>
                            <td>F-score</td>
                            <td>{self.evaluation_results["overall"]["f_score"]:.4f}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="metrics">
                    <h2>Entity-Level Metrics</h2>
                    <table>
                        <tr>
                            <th>Entity</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F-score</th>
                        </tr>
            """)
            
            # Add entity metrics rows
            for entity, metrics in self.evaluation_results["per_entity"].items():
                f.write(f"""
                        <tr>
                            <td>{entity}</td>
                            <td>{metrics["p"]:.4f}</td>
                            <td>{metrics["r"]:.4f}</td>
                            <td>{metrics["f"]:.4f}</td>
                        </tr>
                """)
            
            f.write(f"""
                    </table>
                </div>
                
                <div class="plot">
                    <h2>Overall Performance</h2>
                    <img src="overall_metrics.png" alt="Overall Metrics" style="max-width: 100%;">
                </div>
                
                <div class="plot">
                    <h2>Entity-Level Performance</h2>
                    <img src="entity_metrics.png" alt="Entity Metrics" style="max-width: 100%;">
                </div>
            </body>
            </html>
            """)
        
        return output_dir

# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path="./model",
        test_data_path="./data/test.spacy"
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Generate report
    report_path = evaluator.generate_report()
    
    print(f"Evaluation report generated at: {report_path}") 