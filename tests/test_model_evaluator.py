import unittest
from unittest.mock import patch, MagicMock
import os
import json
import tempfile
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

from src.model_evaluator import ModelEvaluator

class TestModelEvaluator(unittest.TestCase):
    """Test cases for the ModelEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample evaluation results for testing
        self.sample_results = {
            "overall": {
                "precision": 0.85,
                "recall": 0.78,
                "f_score": 0.81
            },
            "per_entity": {
                "PRODUCT": {
                    "p": 0.88,
                    "r": 0.82,
                    "f": 0.85
                },
                "BRAND": {
                    "p": 0.82,
                    "r": 0.75,
                    "f": 0.78
                }
            }
        }
        
        # Create a mock ProductNER instance
        self.mock_ner_patcher = patch('src.model_evaluator.ProductNER')
        self.mock_ner = self.mock_ner_patcher.start()
        
        # Configure the mock to return sample results
        self.mock_ner_instance = MagicMock()
        self.mock_ner_instance.evaluate_model.return_value = self.sample_results
        self.mock_ner.return_value = self.mock_ner_instance
        
        # Create evaluator instance
        self.evaluator = ModelEvaluator(
            model_path="dummy_model_path",
            test_data_path="dummy_test_path"
        )
        
        # Set evaluation results directly for testing visualization methods
        self.evaluator.evaluation_results = self.sample_results
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Stop patchers
        self.mock_ner_patcher.stop()
        
        # Close any open plots
        plt.close('all')
    
    def test_initialization(self):
        """Test that the evaluator initializes correctly."""
        evaluator = ModelEvaluator("model_path", "test_path")
        self.assertEqual(evaluator.model_path, "model_path")
        self.assertEqual(evaluator.test_data_path, "test_path")
        self.assertIsNone(evaluator.evaluation_results)
    
    def test_evaluate(self):
        """Test the evaluate method."""
        # Call evaluate
        results = self.evaluator.evaluate()
        
        # Check that the NER evaluate_model method was called with correct args
        self.mock_ner_instance.evaluate_model.assert_called_once_with(
            test_path="dummy_test_path",
            model_path="dummy_model_path"
        )
        
        # Check that results were stored and returned
        self.assertEqual(results, self.sample_results)
        self.assertEqual(self.evaluator.evaluation_results, self.sample_results)
    
    def test_evaluate_with_new_paths(self):
        """Test evaluate with new paths provided."""
        results = self.evaluator.evaluate("new_model", "new_test")
        
        # Check that paths were updated
        self.assertEqual(self.evaluator.model_path, "new_model")
        self.assertEqual(self.evaluator.test_data_path, "new_test")
        
        # Check that evaluate was called with new paths
        self.mock_ner_instance.evaluate_model.assert_called_once_with(
            test_path="new_test",
            model_path="new_model"
        )
    
    def test_evaluate_missing_paths(self):
        """Test that evaluate raises error when paths are missing."""
        evaluator = ModelEvaluator()
        with self.assertRaises(ValueError):
            evaluator.evaluate()
    
    def test_save_results(self):
        """Test saving evaluation results to file."""
        output_path = os.path.join(self.temp_dir, "results")
        file_path = self.evaluator.save_results(output_path)
        
        # Check that the directory was created
        self.assertTrue(os.path.exists(output_path))
        
        # Check that the file exists
        self.assertTrue(os.path.exists(file_path))
        
        # Check file contents
        with open(file_path, 'r') as f:
            saved_results = json.load(f)
            self.assertEqual(saved_results, self.sample_results)
    
    def test_save_results_no_evaluation(self):
        """Test that save_results raises error when no evaluation results exist."""
        evaluator = ModelEvaluator()
        with self.assertRaises(ValueError):
            evaluator.save_results()
    
    def test_plot_overall_metrics(self):
        """Test plotting overall metrics."""
        # Call the method
        fig = self.evaluator.plot_overall_metrics()
        
        # Check that a figure was returned
        self.assertIsNotNone(fig)
        
        # Test saving the plot
        output_path = os.path.join(self.temp_dir, "overall.png")
        fig = self.evaluator.plot_overall_metrics(output_path)
        self.assertTrue(os.path.exists(output_path))
    
    def test_plot_entity_metrics(self):
        """Test plotting entity metrics."""
        # Call the method
        fig = self.evaluator.plot_entity_metrics()
        
        # Check that a figure was returned
        self.assertIsNotNone(fig)
        
        # Test saving the plot
        output_path = os.path.join(self.temp_dir, "entity.png")
        fig = self.evaluator.plot_entity_metrics(output_path)
        self.assertTrue(os.path.exists(output_path))
    
    def test_generate_report(self):
        """Test generating a complete evaluation report."""
        report_dir = os.path.join(self.temp_dir, "report")
        output_dir = self.evaluator.generate_report(report_dir)
        
        # Check that the directory was created
        self.assertTrue(os.path.exists(report_dir))
        self.assertEqual(output_dir, report_dir)
        
        # Check that report files were created
        self.assertTrue(os.path.exists(os.path.join(report_dir, "evaluation_report.html")))
        self.assertTrue(os.path.exists(os.path.join(report_dir, "overall_metrics.png")))
        self.assertTrue(os.path.exists(os.path.join(report_dir, "entity_metrics.png")))
        
        # Check that at least one results JSON file exists
        json_files = list(Path(report_dir).glob("evaluation_results_*.json"))
        self.assertGreater(len(json_files), 0)

if __name__ == '__main__':
    unittest.main() 