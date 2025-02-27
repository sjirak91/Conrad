import pandas as pd
import numpy as np
import re
import spacy
from spacy.tokens import DocBin
from pathlib import Path
import random
import os
from sklearn.model_selection import train_test_split

# Define constants
ENTITY_TYPES = ["BRAND", "STORAGE", "COLOR"]
TRAIN_TEST_SPLIT = 0.2
RANDOM_SEED = 42

class DataProcessor:
    def __init__(self):
        """Initialize the DataProcessor class with entity types."""
        self.entity_types = ENTITY_TYPES
    
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