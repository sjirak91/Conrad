from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os
from ner_model import NERModel

# Initialize FastAPI
app = FastAPI(
    title="Named Entity Recognition API",
    description="API for extracting named entities from text",
    version="1.0.0"
)

# Load the NER model
model_path = os.environ.get("MODEL_PATH", "./model")

# Initialize the NER model
try:
    ner_model = NERModel()
    ner_model.load_model(model_path)
    print(f"NER model loaded from {model_path}")
except Exception as e:
    print(f"Warning: Failed to load model. Error: {str(e)}")
    print("Will attempt to load model in the predict endpoint.")
    ner_model = None

# Define Pydantic models for request and response
class PredictionRequest(BaseModel):
    text: str
    confidence_threshold: Optional[float] = 0.5

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float

class PredictionResponse(BaseModel):
    text: str
    entities: List[Entity]
    tagged_text: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint with health check and API information."""
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "model_loaded": ner_model is not None,
        "endpoints": {
            "POST /predict": "Predict entities in text",
            "GET /health": "API health status",
            "GET /entities": "Get list of supported entity types"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "model_loaded": ner_model is not None,
        "model_path": model_path
    }

@app.get("/entities")
async def get_entities():
    """Get list of supported entity types."""
    global ner_model
    
    if ner_model is None:
        try:
            ner_model = NERModel()
            ner_model.load_model(model_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    try:
        entity_types = ner_model.get_entity_types()
        return {"entity_types": entity_types}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving entity types: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict entities in the provided text.
    
    This endpoint extracts named entities from the input text based on the trained model.
    You can optionally specify a confidence threshold to filter results.
    """
    global ner_model
    
    # Check if model is loaded
    if ner_model is None:
        try:
            ner_model = NERModel()
            ner_model.load_model(model_path)
            print(f"NER model loaded from {model_path}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    # Get predictions
    try:
        entities = ner_model.predict(
            request.text, 
            confidence_threshold=request.confidence_threshold
        )
        tagged_text = ner_model.tag_text(request.text, entities)
        
        return {
            "text": request.text,
            "entities": entities,
            "tagged_text": tagged_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """
    Batch predict entities for multiple text inputs.
    """
    global ner_model
    
    # Check if model is loaded
    if ner_model is None:
        try:
            ner_model = NERModel()
            ner_model.load_model(model_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    results = []
    for request in requests:
        try:
            entities = ner_model.predict(
                request.text, 
                confidence_threshold=request.confidence_threshold
            )
            tagged_text = ner_model.tag_text(request.text, entities)
            
            results.append({
                "text": request.text,
                "entities": entities,
                "tagged_text": tagged_text
            })
        except Exception as e:
            results.append({
                "text": request.text,
                "error": str(e)
            })
    
    return {"results": results}

# Main entry point for running the API server directly
if __name__ == "__main__":
    # Get the port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False) 