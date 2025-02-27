import pandas as pd
import numpy as np
import re
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from pathlib import Path
import random
from tqdm import tqdm
import json
import os
from sklearn.model_selection import train_test_split
from src.utils import compounding, minibatch
from src.data_processor import ENTITY_TYPES, RANDOM_SEED

# Define constants
TRAIN_TEST_SPLIT = 0.2

class ProductNER:
    def __init__(self):
        """Initialize the ProductNER class with entity types."""
        self.entity_types = ENTITY_TYPES
        self.model = None
        self.nlp = None
    
    def preprocess_data(self, data_path, output_dir="./data"):
        """
        Preprocess the CSV data into a format suitable for NER training.
        
        Args:
            data_path: Path to the CSV file with product data
            output_dir: Directory to save processed data
        
        Returns:
            Dictionary with training and testing data
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Read the CSV file
        df = pd.read_csv(data_path)
        
        # Drop rows with missing products
        df = df.dropna(subset=['productId'])
        
        # Combine the text fields and clean HTML tags
        df['text_headline'] = df['headline'].apply(lambda x: self._clean_html(x) if isinstance(x, str) else "")
        df['text_description'] = df['description'].apply(lambda x: self._clean_html(x) if isinstance(x, str) else "")
        df['text_highlights'] = df['highlights'].apply(lambda x: self._clean_html(x) if isinstance(x, str) else "")
        
        # Create training data for each text field
        training_data = []
        
        # Process headline, description, and highlights separately
        for field in ['text_headline', 'text_description', 'text_highlights']:
            field_data = self._create_training_examples(df, field)
            training_data.extend(field_data)
        
        # Shuffle the data
        random.seed(RANDOM_SEED)
        random.shuffle(training_data)
        
        # Split into train and test sets
        train_data, test_data = train_test_split(
            training_data, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
        )
        
        # Save the data
        self._save_data(train_data, os.path.join(output_dir, "train.spacy"))
        self._save_data(test_data, os.path.join(output_dir, "test.spacy"))
        
        return {"train": train_data, "test": test_data}
    
    def _clean_html(self, text):
        """Remove HTML tags from text."""
        if not isinstance(text, str):
            return ""
        clean = re.compile('<.*?>')
        return re.sub(clean, ' ', text).strip()
    
    def _create_training_examples(self, df, text_field):
        """
        Create NER training examples from a dataframe column.
        
        Args:
            df: DataFrame with product data
            text_field: Field name containing the text to process
        
        Returns:
            List of (text, entities) tuples for training
        """
        examples = []
        nlp = spacy.blank("de")  # Use German language model
        
        for _, row in df.iterrows():
            text = row[text_field]
            if not text or not isinstance(text, str) or len(text.strip()) == 0:
                continue
                
            # Create a doc object
            doc = nlp.make_doc(text)
            entities = []
            
            # Add Brand entity if available
            if row['Brand'] and isinstance(row['Brand'], str):
                self._add_entity(text, row['Brand'], "BRAND", entities)
            
            # Add Storage entity if available
            if row['Speicherkapazität'] and not pd.isna(row['Speicherkapazität']):
                storage_text = str(row['Speicherkapazität'])
                self._add_entity(text, storage_text, "STORAGE", entities)
            
            # Add Color entity if available
            if row['Farbe'] and isinstance(row['Farbe'], str):
                self._add_entity(text, row['Farbe'], "COLOR", entities)
            
            # Add example if we found any entities
            if entities:
                examples.append((text, {"entities": entities}))
        
        return examples
    
    def _add_entity(self, text, entity_value, entity_label, entities_list):
        """
        Find all occurrences of an entity in the text and add them to the entities list.
        
        Args:
            text: The text to search in
            entity_value: The entity value to find
            entity_label: The entity label (BRAND, STORAGE, COLOR)
            entities_list: List to append found entities to
        """
        # Try to find exact matches first
        for match in re.finditer(re.escape(entity_value), text, re.IGNORECASE):
            entities_list.append((match.start(), match.end(), entity_label))
            
        # For storage, also try to find patterns like "500 GB" or "2 TB"
        if entity_label == "STORAGE":
            # Extract the numeric value and unit if available
            storage_match = re.match(r'(\d+(?:\.\d+)?)\s*(GB|TB|MB)', entity_value, re.IGNORECASE)
            if storage_match:
                size, unit = storage_match.groups()
                for match in re.finditer(r'\b' + re.escape(size) + r'\s*' + re.escape(unit) + r'\b', text, re.IGNORECASE):
                    entities_list.append((match.start(), match.end(), entity_label))
    
    def _save_data(self, examples, output_path):
        """
        Convert training examples to spaCy's binary format and save to disk.
        
        Args:
            examples: List of (text, entities) tuples
            output_path: Path to save the binary file
        """
        nlp = spacy.blank("de")  # Use German language model
        db = DocBin()
        
        for text, annot in examples:
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in annot["entities"]:
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    ents.append(span)
            doc.ents = ents
            db.add(doc)
        
        db.to_disk(output_path)
    
    def train_model(self, train_path, output_dir="./model", n_iter=30):
        """
        Train a spaCy NER model using the prepared data.
        
        Args:
            train_path: Path to the spaCy binary training data
            output_dir: Directory to save the trained model
            n_iter: Number of training iterations
            
        Returns:
            Trained model path
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create the NLP pipeline with NER component
        nlp = spacy.blank("de")
        
        # Create a new pipeline with the NER component
        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")
        
        # Add entity labels
        for entity_type in self.entity_types:
            ner.add_label(entity_type)
        
        # Start training with custom config
        from spacy.training.loop import train
        from spacy.training.initialize import init_nlp
        
        # Create a config with basic settings
        config = {
            "nlp": {"lang": "de", "pipeline": ["ner"]},
            "components": {
                "ner": {
                    "factory": "ner",
                    "moves": None,
                    "update_with_oracle_cut_size": 100
                }
            },
            "training": {
                "dev_corpus": "corpora.dev",
                "train_corpus": "corpora.train",
                "seed": RANDOM_SEED,
                "gpu_allocator": None,
                "dropout": 0.1,
                "patience": 1600,
                "max_steps": 20000,
                "eval_frequency": 200,
                "frozen_components": [],
                "before_to_disk": None,
                "batcher": {
                    "@batchers": "spacy.batch_by_words.v1",
                    "discard_oversize": False,
                    "tolerance": 0.2,
                    "get_length": None,
                    "size": {
                        "@schedules": "compounding.v1",
                        "start": 100,
                        "stop": 1000,
                        "compound": 1.001,
                        "t": 0.0
                    }
                },
                "logger": {"@loggers": "spacy.ConsoleLogger.v1", "progress_bar": True},
                "optimizer": {
                    "@optimizers": "Adam.v1",
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "L2_is_weight_decay": True,
                    "L2": 0.01,
                    "grad_clip": 1.0,
                    "use_averages": False,
                    "eps": 1e-8
                },
                "score_weights": {"ents_f": 1.0, "ents_p": 0.0, "ents_r": 0.0}
            }
        }
        
        # Initialize the pipeline
        nlp_initialized = init_nlp(config)
        
        # Load the training data
        train_examples = []
        doc_bin = DocBin().from_disk(train_path)
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        for doc in docs:
            train_examples.append(Example.from_dict(doc, {"entities": [(e.start_char, e.end_char, e.label_) for e in doc.ents]}))
        
        # Configure training parameters
        optimizer = nlp_initialized.resume_training()
        optimizer.learn_rate = 0.001
        
        # Start training
        batch_sizes = compounding(4.0, 32.0, 1.001)
        
        with tqdm(total=n_iter, desc="Training NER model") as pbar:
            for i in range(n_iter):
                losses = {}
                random.shuffle(train_examples)
                batches = minibatch(train_examples, size=batch_sizes)
                
                for batch in batches:
                    nlp_initialized.update(batch, drop=0.5, losses=losses)
                
                pbar.update(1)
                pbar.set_postfix(losses=losses)
        
        # Save the model
        nlp_initialized.to_disk(output_dir)
        
        # Update the class model
        self.model = nlp_initialized
        self.nlp = nlp_initialized
        
        return output_dir
    
    def evaluate_model(self, test_path, model_path=None):
        """
        Evaluate the NER model using test data.
        
        Args:
            test_path: Path to the spaCy binary test data
            model_path: Path to the trained model (optional if model already loaded)
            
        Returns:
            Dictionary with precision, recall, and f-score
        """
        # Load the model if path is provided
        if model_path:
            self.nlp = spacy.load(model_path)
            self.model = self.nlp
        
        if not self.nlp:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        # Load the test data
        doc_bin = DocBin().from_disk(test_path)
        test_docs = list(doc_bin.get_docs(self.nlp.vocab))
        
        # Create examples
        examples = []
        for doc in test_docs:
            examples.append(Example.from_dict(self.nlp.make_doc(doc.text), 
                                             {"entities": [(e.start_char, e.end_char, e.label_) for e in doc.ents]}))
        
        # Evaluate
        scorer = self.nlp.evaluate(examples)
        
        # Extract NER scores
        ner_scores = scorer.scores["ents_per_type"]
        
        # Calculate overall scores
        overall = {
            "precision": scorer.scores["ents_p"],
            "recall": scorer.scores["ents_r"],
            "f_score": scorer.scores["ents_f"]
        }
        
        # Return detailed scores
        return {
            "overall": overall,
            "per_entity": ner_scores
        }
    
    def predict(self, text):
        """
        Make NER predictions on a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of predicted entities
        """
        if not self.nlp:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        # Clean the text
        clean_text = self._clean_html(text)
        
        # Process the text
        doc = self.nlp(clean_text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        return entities
    
    def tag_text(self, text):
        """
        Tag entities in a text with their labels.
        
        Args:
            text: Text to tag
            
        Returns:
            Text with entities tagged
        """
        if not self.nlp:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        # Clean the text
        clean_text = self._clean_html(text)
        
        # Process the text
        doc = self.nlp(clean_text)
        
        # Create a copy of the text
        tagged_text = clean_text
        
        # Replace entities with tagged versions (process in reverse to avoid offsets changing)
        entities = [(ent.start_char, ent.end_char, ent.label_, ent.text) for ent in doc.ents]
        entities.sort(reverse=True)
        
        for start, end, label, text in entities:
            tagged_text = tagged_text[:start] + f"[{text}]({label})" + tagged_text[end:]
        
        return tagged_text
    
    def load_model(self, model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the trained model
        """
        self.nlp = spacy.load(model_path)
        self.model = self.nlp