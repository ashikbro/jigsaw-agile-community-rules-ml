from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
from text_analyzer import text_analyzer
from model_manager import model_manager

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, description="The text to analyze")
    model_type: str = Field("simple_model", description="The model to use for prediction")
    use_cache: bool = Field(True, description="Whether to use cached predictions")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of texts to analyze")
    model_type: str = Field("simple_model", description="The model to use for predictions")
    use_cache: bool = Field(True, description="Whether to use cached predictions")

class AnalysisResult(BaseModel):
    text: str
    language: str
    sentiment: Dict[str, float]
    statistics: Dict[str, Any]
    prediction: Optional[Dict[str, Any]] = None

app = FastAPI(
    title="Text Analysis API",
    description="API for text analysis and classification with model management and caching",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"""
    Request: {request.method} {request.url}
    Status: {response.status_code}
    Process Time: {process_time:.2f}ms
    """)
    
    return response

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Text Analysis API is running",
        "version": "1.1.0",
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    }

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    models = model_manager.list_models()
    return {
        "status": "ok",
        "models_loaded": [m['name'] for m in models if m['loaded']],
        "total_models": len(models),
        "timestamp": datetime.utcnow().isoformat()
    }

# Model management endpoints
@app.get("/models", tags=["Models"], response_model=List[Dict[str, Any]])
async def list_models():
    """List all available models with their metadata."""
    return model_manager.list_models()

@app.get("/models/{model_name}", tags=["Models"])
async def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    return model_manager.get_model_info(model_name)

@app.post("/models/clear-cache", tags=["Models"])
async def clear_cache():
    """Clear the prediction cache."""
    return model_manager.clear_cache()

# Text analysis endpoints
@app.post("/analyze", response_model=Dict[str, Any], tags=["Analysis"])
async def analyze_text(input_data: TextInput):
    """
    Analyze text with sentiment analysis, language detection, and statistics.
    Optionally include prediction if models are available.
    """
    if not input_data.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty"
        )
    
    # Get text analysis
    analysis = text_analyzer.analyze_text(input_data.text)
    
    # Add prediction if models are available
    if model_manager.models:
        prediction = model_manager.predict(
            text=input_data.text,
            model_name=input_data.model_type,
            use_cache=input_data.use_cache
        )
        if prediction.get('status') == 'success':
            analysis["prediction"] = {
                k: v for k, v in prediction.items()
                if k not in ('status', 'error')
            }
    
    return analysis

@app.post("/predict", tags=["Prediction"])
async def predict(input_data: TextInput):
    """
    Make a prediction using the specified model.
    Returns only the prediction without additional analysis.
    """
    result = model_manager.predict(
        text=input_data.text,
        model_name=input_data.model_type,
        use_cache=input_data.use_cache
    )
    
    if result.get('status') == 'error':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get('error', 'Prediction failed')
        )
    
    return {
        "text": input_data.text,
        **{k: v for k, v in result.items() if k != 'status'}
    }

@app.post("/batch-analyze", tags=["Batch Processing"])
async def batch_analyze(input_data: BatchTextInput):
    """
    Analyze multiple texts in a single request.
    Returns analysis for each text including predictions if models are available.
    """
    if not input_data.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No texts provided"
        )
    
    results = []
    for text in input_data.texts:
        if not text.strip():
            results.append({
                "error": "Empty text",
                "status": "error"
            })
            continue
            
        analysis = text_analyzer.analyze_text(text)
        
        # Add prediction if models are available
        if model_manager.models:
            prediction = model_manager.predict(
                text=text,
                model_name=input_data.model_type,
                use_cache=input_data.use_cache
            )
            if prediction.get('status') == 'success':
                analysis["prediction"] = {
                    k: v for k, v in prediction.items()
                    if k not in ('status', 'error')
                }
        
        results.append(analysis)
    
    return {
        "count": len(results),
        "results": results,
        "cached_items": sum(1 for r in results if r.get('prediction', {}).get('cached', False)),
        "status": "success"
    }

# Cache management endpoints
@app.get("/cache/stats", tags=["Cache"])
async def get_cache_stats():
    """Get statistics about the prediction cache."""
    return model_manager.get_cache_stats()

@app.post("/cache/clear", tags=["Cache"])
async def clear_prediction_cache():
    """Clear the prediction cache."""
    return model_manager.clear_cache()

if __name__ == "__main__":
    uvicorn.run(
        "simple_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        workers=1
    )
