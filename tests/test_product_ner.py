import unittest
import os
import tempfile
import pandas as pd
import spacy
from spacy.tokens import DocBin
from pathlib import Path
import shutil
import sys

# Add the parent directory to the path so we can import the product_ner module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from product_ner import ProductNER

class TestProductNER(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        self.ner = ProductNER()
        
        # Create temporary directories for test data and models
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.model_dir = os.path.join(self.temp_dir, "model")
        
        # Create test data
        self.create_test_data()
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directories
        shutil.rmtree(self.temp_dir)
    
    def create_test_data(self):
        """Create a small test dataset for testing."""
        # Create a sample DataFrame
        data = {
            'productId': [1, 2, 3],
            'headline': ['Apple iPhone 12 Pro 256GB Silber', 'Samsung Galaxy S21 128GB Schwarz', 'Xiaomi Mi 11 256GB Blau'],
            'description': ['Das iPhone 12 Pro mit 256GB Speicher in Silber', 'Das Samsung Galaxy S21 mit 128GB Speicher in Schwarz', 'Das Xiaomi Mi 11 mit 256GB Speicher in Blau'],
            'highlights': ['256GB Speicher, Silber', '128GB Speicher, Schwarz', '256GB Speicher, Blau'],
            'Brand': ['Apple', 'Samsung', 'Xiaomi'],
            'Speicherkapazität': ['256GB', '128GB', '256GB'],
            'Farbe': ['Silber', 'Schwarz', 'Blau']
        }
        
        df = pd.DataFrame(data)
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Save DataFrame to CSV
        self.test_csv_path = os.path.join(self.data_dir, "test_products.csv")
        df.to_csv(self.test_csv_path, index=False)
    
    def test_init(self):
        """Test initialization of ProductNER class."""
        self.assertEqual(self.ner.entity_types, ["BRAND", "STORAGE", "COLOR"])
        self.assertIsNone(self.ner.model)
        self.assertIsNone(self.ner.nlp)
    
    def test_clean_html(self):
        """Test HTML cleaning functionality."""
        html_text = "<p>This is a <b>test</b> with <i>HTML</i> tags</p>"
        expected = "This is a test with HTML tags"
        self.assertEqual(self.ner._clean_html(html_text), expected)
        
        # Test with non-string input
        self.assertEqual(self.ner._clean_html(None), "")
        self.assertEqual(self.ner._clean_html(123), "")
    
    def test_add_entity(self):
        """Test adding entities to the entities list."""
        text = "Apple iPhone with 256GB in Silber"
        entities = []
        
        # Test brand entity
        self.ner._add_entity(text, "Apple", "BRAND", entities)
        self.assertEqual(entities, [(0, 5, "BRAND")])
        
        # Test storage entity
        self.ner._add_entity(text, "256GB", "STORAGE", entities)
        self.assertEqual(entities, [(0, 5, "BRAND"), (17, 22, "STORAGE")])
        
        # Test color entity
        self.ner._add_entity(text, "Silber", "COLOR", entities)
        self.assertEqual(entities, [(0, 5, "BRAND"), (17, 22, "STORAGE"), (26, 32, "COLOR")])
    
    def test_preprocess_data(self):
        """Test data preprocessing functionality."""
        # Process the test data
        processed_data = self.ner.preprocess_data(self.test_csv_path, output_dir=self.data_dir)
        
        # Check if train and test files were created
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "train.spacy")))
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "test.spacy")))
        
        # Check if processed data contains train and test sets
        self.assertIn("train", processed_data)
        self.assertIn("test", processed_data)
        
        # Verify that the data contains examples
        total_examples = len(processed_data["train"]) + len(processed_data["test"])
        self.assertGreater(total_examples, 0)
    
    def test_save_and_load_data(self):
        """Test saving and loading spaCy binary data."""
        # Create some example data
        examples = [
            ("Apple iPhone 12", {"entities": [(0, 5, "BRAND")]}),
            ("Samsung Galaxy with 128GB", {"entities": [(0, 7, "BRAND"), (19, 24, "STORAGE")]})
        ]
        
        # Save the data
        output_path = os.path.join(self.data_dir, "test_save.spacy")
        self.ner._save_data(examples, output_path)
        
        # Check if the file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Load the data to verify it was saved correctly
        nlp = spacy.blank("de")
        doc_bin = DocBin().from_disk(output_path)
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        # Check if we have the correct number of documents
        self.assertEqual(len(docs), 2)
        
        # Check if entities were preserved
        self.assertEqual(len(docs[0].ents), 1)
        self.assertEqual(docs[0].ents[0].label_, "BRAND")
        self.assertEqual(docs[0].ents[0].text, "Apple")
        
        self.assertEqual(len(docs[1].ents), 2)
        self.assertEqual(docs[1].ents[0].label_, "BRAND")
        self.assertEqual(docs[1].ents[0].text, "Samsung")
        self.assertEqual(docs[1].ents[1].label_, "STORAGE")
        self.assertEqual(docs[1].ents[1].text, "128GB")
    
    def test_create_training_examples(self):
        """Test creation of training examples from DataFrame."""
        # Create a simple DataFrame
        data = {
            'text_headline': ['Apple iPhone 12 Pro 256GB Silber'],
            'Brand': ['Apple'],
            'Speicherkapazität': ['256GB'],
            'Farbe': ['Silber']
        }
        df = pd.DataFrame(data)
        
        # Create training examples
        examples = self.ner._create_training_examples(df, 'text_headline')
        
        # Check if we have examples
        self.assertEqual(len(examples), 1)
        
        # Check the content of the example
        text, annotations = examples[0]
        self.assertEqual(text, 'Apple iPhone 12 Pro 256GB Silber')
        
        # Check if all entities were found
        entities = annotations['entities']
        self.assertEqual(len(entities), 3)
        
        # Check entity types
        entity_types = [entity[2] for entity in entities]
        self.assertIn('BRAND', entity_types)
        self.assertIn('STORAGE', entity_types)
        self.assertIn('COLOR', entity_types)
    
    @unittest.skip("Skipping training test as it's resource-intensive")
    def test_train_model(self):
        """Test model training functionality."""
        # First preprocess the data
        self.ner.preprocess_data(self.test_csv_path, output_dir=self.data_dir)
        
        # Train the model with fewer iterations for testing
        train_path = os.path.join(self.data_dir, "train.spacy")
        model_path = self.ner.train_model(train_path, output_dir=self.model_dir, n_iter=2)
        
        # Check if model was saved
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(os.path.join(model_path, "config.cfg")))
        self.assertTrue(os.path.exists(os.path.join(model_path, "model")))
        
        # Check if model was loaded into the class
        self.assertIsNotNone(self.ner.model)
        self.assertIsNotNone(self.ner.nlp)
    
    def test_load_model(self):
        """Test loading a pre-trained model."""
        # Create a minimal spaCy model for testing
        nlp = spacy.blank("de")
        
        if "ner" not in nlp.pipe_names:
            nlp.add_pipe("ner")
        
        # Save the model
        os.makedirs(self.model_dir, exist_ok=True)
        nlp.to_disk(self.model_dir)
        
        # Load the model
        self.ner.load_model(self.model_dir)
        
        # Check if model was loaded
        self.assertIsNotNone(self.ner.model)
        self.assertIsNotNone(self.ner.nlp)
    
    def test_predict_without_model(self):
        """Test prediction without a loaded model."""
        with self.assertRaises(ValueError):
            self.ner.predict("Apple iPhone 12")
    
    def test_tag_text_without_model(self):
        """Test text tagging without a loaded model."""
        with self.assertRaises(ValueError):
            self.ner.tag_text("Apple iPhone 12")
    
    def test_evaluate_without_model(self):
        """Test evaluation without a loaded model."""
        with self.assertRaises(ValueError):
            self.ner.evaluate_model("dummy_path")
    
    def test_predict_and_tag_with_model(self):
        """Test prediction and tagging with a loaded model."""
        # Create and load a minimal model
        nlp = spacy.blank("de")
        ner = nlp.add_pipe("ner")
        ner.add_label("BRAND")
        
        # Add a simple rule-based entity recognizer for testing
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        patterns = [{"label": "BRAND", "pattern": "Apple"}]
        ruler.add_patterns(patterns)
        
        # Save the model
        os.makedirs(self.model_dir, exist_ok=True)
        nlp.to_disk(self.model_dir)
        
        # Load the model
        self.ner.load_model(self.model_dir)
        
        # Test prediction
        entities = self.ner.predict("Apple iPhone 12")
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["text"], "Apple")
        self.assertEqual(entities[0]["label"], "BRAND")
        
        # Test tagging
        tagged_text = self.ner.tag_text("Apple iPhone 12")
        self.assertEqual(tagged_text, "[Apple](BRAND) iPhone 12")

if __name__ == '__main__':
    unittest.main() 