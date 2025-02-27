import spacy
from typing import List, Dict, Optional
from src.product_ner import ProductNER

class NERModel:
    """Wrapper class for ProductNER to provide API-friendly interface"""
    
    def __init__(self):
        """Initialize the NER model"""
        self.ner = ProductNER()
        
    def load_model(self, model_path: str):
        """Load a trained model from disk"""
        self.ner.load_model(model_path)
        
    def predict(self, text: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Predict entities in text with confidence scores
        
        Args:
            text: Text to analyze
            confidence_threshold: Minimum confidence score for entities
            
        Returns:
            List of entity dictionaries with text, label, start, end, and confidence
        """
        entities = self.ner.predict(text)
        
        # Add confidence scores (ProductNER doesn't provide these, so we'll use a placeholder)
        for entity in entities:
            entity["confidence"] = 1.0
            
        # Filter by confidence threshold
        return [e for e in entities if e["confidence"] >= confidence_threshold]
    
    def tag_text(self, text: str, entities: Optional[List[Dict]] = None) -> str:
        """
        Tag text with entity labels
        
        Args:
            text: Text to tag
            entities: Optional pre-computed entities
            
        Returns:
            Text with entities tagged
        """
        if entities is None:
            return self.ner.tag_text(text)
        
        # Create a copy of the text
        tagged_text = text
        
        # Replace entities with tagged versions (process in reverse to avoid offsets changing)
        sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)
        
        for entity in sorted_entities:
            start = entity["start"]
            end = entity["end"]
            label = entity["label"]
            entity_text = entity["text"]
            
            tagged_text = tagged_text[:start] + f"[{entity_text}]({label})" + tagged_text[end:]
        
        return tagged_text
    
    def get_entity_types(self) -> List[str]:
        """Get list of supported entity types"""
        return self.ner.entity_types 