#!/usr/bin/env python3
"""
Content Moderation REST API Server

A comprehensive FastAPI-based REST API for content moderation with:
- Multiple model support (Human-aware, Context-aware, Original)
- Explainable AI integration
- Batch processing capabilities
- Health monitoring
- Rate limiting
- Authentication support

Author: AI Assistant
Date: 2025
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import joblib
import os
import glob
import time
import logging
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager
import uvicorn

# Import our modules
try:
    from explainable_ai_module import ContentModerationExplainer, load_explainer_from_model_dir
except ImportError:
    print("Warning: Explainable AI module not available")
    ContentModerationExplainer = None
    load_explainer_from_model_dir = None

try:
    from multilingual_support import MultilingualContentModerator
except ImportError:
    print("Warning: Multilingual support module not available")
    MultilingualContentModerator = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
models = {}
vectorizers = {}
explainers = {}
model_info = {}
multilingual_moderator = None

# Rate limiting storage (in production, use Redis)
rate_limit_storage = {}

# Pydantic models for API
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")
    model_type: Optional[str] = Field("auto", description="Model type: auto, human_aware, context_aware, or original")
    include_explanation: Optional[bool] = Field(False, description="Include explainable AI analysis")
    include_probabilities: Optional[bool] = Field(True, description="Include prediction probabilities")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to classify")
    model_type: Optional[str] = Field("auto", description="Model type: auto, human_aware, context_aware, or original")
    include_explanation: Optional[bool] = Field(False, description="Include explainable AI analysis")
    include_probabilities: Optional[bool] = Field(True, description="Include prediction probabilities")

class PredictionResponse(BaseModel):
    text: str
    prediction: str  # "safe" or "violation"
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    explanation: Optional[Dict[str, Any]] = None
    model_used: str
    timestamp: str
    processing_time_ms: float

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_processed: int
    total_time_ms: float
    model_used: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    explainers_available: Dict[str, bool]
    uptime_seconds: float
    version: str = "1.0.0"

class ModelInfoResponse(BaseModel):
    available_models: Dict[str, Dict[str, Any]]
    default_model: str
    explainable_ai_available: bool

class MultilingualTextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify in any language")
    target_language: Optional[str] = Field("en", description="Target language for translation (ISO 639-1 code)")
    auto_translate: Optional[bool] = Field(True, description="Automatically translate to target language")
    model_type: Optional[str] = Field("auto", description="Model type: auto, human_aware, context_aware, or original")
    include_explanation: Optional[bool] = Field(False, description="Include explainable AI analysis")
    include_probabilities: Optional[bool] = Field(True, description="Include prediction probabilities")

class MultilingualPredictionResponse(BaseModel):
    original_text: str
    detected_language: str
    language_confidence: float
    translated_text: Optional[str] = None
    prediction: str  # "safe" or "violation"
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    explanation: Optional[Dict[str, Any]] = None
    model_used: str
    timestamp: str
    processing_time_ms: float

class LanguageStatsResponse(BaseModel):
    total_texts_processed: int
    language_distribution: Dict[str, Dict[str, Union[int, float]]]
    supported_languages: List[str]
    translation_enabled: bool

# Security
security = HTTPBearer(auto_error=False)

def load_all_models():
    """Load all available models on startup."""
    global models, vectorizers, explainers, model_info
    
    logger.info("Loading models...")
    
    # Load human-aware models
    human_aware_dirs = glob.glob("models/human_aware_*")
    for model_dir in human_aware_dirs:
        try:
            model_files = glob.glob(os.path.join(model_dir, "*_model.joblib"))
            vectorizer_files = glob.glob(os.path.join(model_dir, "*_vectorizer.joblib"))
            
            if model_files and vectorizer_files:
                model_name = os.path.basename(model_files[0]).replace('_model.joblib', '')
                key = f"human_aware_{model_name}"
                
                models[key] = joblib.load(model_files[0])
                vectorizers[key] = joblib.load(vectorizer_files[0])
                
                if load_explainer_from_model_dir:
                    explainers[key] = load_explainer_from_model_dir(model_dir)
                
                model_info[key] = {
                    "type": "human_aware",
                    "path": model_dir,
                    "algorithm": model_name,
                    "description": "Enhanced model with human vs non-human target distinction"
                }
                logger.info(f"Loaded human-aware model: {key}")
        except Exception as e:
            logger.error(f"Error loading human-aware model from {model_dir}: {e}")
    
    # Load context-aware models
    context_aware_dirs = glob.glob("models/context_aware_*")
    for model_dir in context_aware_dirs:
        try:
            model_files = glob.glob(os.path.join(model_dir, "*_model.joblib"))
            vectorizer_files = glob.glob(os.path.join(model_dir, "*_vectorizer.joblib"))
            
            if model_files and vectorizer_files:
                model_name = os.path.basename(model_files[0]).replace('_model.joblib', '')
                key = f"context_aware_{model_name}"
                
                models[key] = joblib.load(model_files[0])
                vectorizers[key] = joblib.load(vectorizer_files[0])
                
                if load_explainer_from_model_dir:
                    explainers[key] = load_explainer_from_model_dir(model_dir)
                
                model_info[key] = {
                    "type": "context_aware",
                    "path": model_dir,
                    "algorithm": model_name,
                    "description": "Enhanced model with better context understanding"
                }
                logger.info(f"Loaded context-aware model: {key}")
        except Exception as e:
            logger.error(f"Error loading context-aware model from {model_dir}: {e}")
    
    # Load original models
    original_model_dirs = ["models/large_dataset_*", "models/medium_dataset_*", "models"]
    for pattern in original_model_dirs:
        for model_dir in glob.glob(pattern):
            try:
                if os.path.isfile(model_dir):  # Skip files
                    continue
                    
                model_files = glob.glob(os.path.join(model_dir, "*_model.joblib")) or \
                             glob.glob(os.path.join(model_dir, "*.joblib"))
                vectorizer_files = glob.glob(os.path.join(model_dir, "*_vectorizer.joblib")) or \
                                  glob.glob(os.path.join(model_dir, "vectorizer.joblib"))
                
                if model_files and vectorizer_files:
                    model_name = os.path.basename(model_files[0]).replace('_model.joblib', '').replace('.joblib', '')
                    key = f"original_{model_name}"
                    
                    if key not in models:  # Avoid duplicates
                        models[key] = joblib.load(model_files[0])
                        vectorizers[key] = joblib.load(vectorizer_files[0])
                        
                        model_info[key] = {
                            "type": "original",
                            "path": model_dir,
                            "algorithm": model_name,
                            "description": "Original baseline model"
                        }
                        logger.info(f"Loaded original model: {key}")
            except Exception as e:
                logger.error(f"Error loading original model from {model_dir}: {e}")
    
    # Load multilingual models
    multilingual_model_files = glob.glob("models/*_multilingual.joblib")
    for model_file in multilingual_model_files:
        try:
            model_name = os.path.basename(model_file).replace('_multilingual.joblib', '')
            vectorizer_file = f"models/{model_name}_vectorizer.joblib"
            
            if os.path.exists(vectorizer_file):
                key = f"multilingual_{model_name}"
                models[key] = joblib.load(model_file)
                vectorizers[key] = joblib.load(vectorizer_file)
                
                model_info[key] = {
                    "type": "multilingual",
                    "path": "models",
                    "algorithm": model_name,
                    "description": "Multilingual model with Bengali support"
                }
                logger.info(f"Loaded multilingual model: {key}")
        except Exception as e:
            logger.error(f"Error loading multilingual model from {model_file}: {e}")
    
    logger.info(f"Total models loaded: {len(models)}")
    logger.info(f"Available models: {list(models.keys())}")
    
    # Initialize multilingual moderator
    global multilingual_moderator
    if MultilingualContentModerator:
        try:
            multilingual_moderator = MultilingualContentModerator()
            logger.info("Multilingual content moderator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing multilingual moderator: {e}")
            multilingual_moderator = None
    else:
        logger.warning("Multilingual support not available")

def get_best_model(model_type: str = "auto"):
    """Get the best available model based on type preference."""
    if model_type != "auto" and model_type in ["multilingual", "human_aware", "context_aware", "original"]:
        # Find best model of specific type
        matching_models = [k for k in models.keys() if k.startswith(model_type)]
        if matching_models:
            # Prefer large dataset models
            large_models = [k for k in matching_models if "large" in k]
            if large_models:
                return large_models[0]
            # For multilingual models, prefer random_forest if available
            if model_type == "multilingual":
                rf_models = [k for k in matching_models if "random_forest" in k]
                if rf_models:
                    return rf_models[0]
            return matching_models[0]
    
    # Auto selection: prefer multilingual > human_aware > context_aware > original
    for prefix in ["multilingual", "human_aware", "context_aware", "original"]:
        matching_models = [k for k in models.keys() if k.startswith(prefix)]
        if matching_models:
            # Prefer large dataset models
            large_models = [k for k in matching_models if "large" in k]
            if large_models:
                return large_models[0]
            # For multilingual models, prefer random_forest if available
            if prefix == "multilingual":
                rf_models = [k for k in matching_models if "random_forest" in k]
                if rf_models:
                    return rf_models[0]
            return matching_models[0]
    
    return None

def check_rate_limit(client_ip: str, limit: int = 100, window: int = 3600) -> bool:
    """Simple rate limiting (in production, use Redis)."""
    now = time.time()
    
    if client_ip not in rate_limit_storage:
        rate_limit_storage[client_ip] = []
    
    # Clean old requests
    rate_limit_storage[client_ip] = [
        req_time for req_time in rate_limit_storage[client_ip] 
        if now - req_time < window
    ]
    
    # Check limit
    if len(rate_limit_storage[client_ip]) >= limit:
        return False
    
    # Add current request
    rate_limit_storage[client_ip].append(now)
    return True

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token verification (in production, use proper JWT)."""
    # For demo purposes, accept any token or no token
    # In production, implement proper authentication
    return True

# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Content Moderation API...")
    load_all_models()
    yield
    # Shutdown
    logger.info("Shutting down Content Moderation API...")

app = FastAPI(
    title="Content Moderation API",
    description="Advanced content moderation API with explainable AI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store startup time for uptime calculation
startup_time = time.time()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Content Moderation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/models"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    models_status = {model_key: True for model_key in models.keys()}
    explainers_status = {model_key: explainers.get(model_key) is not None for model_key in models.keys()}
    
    return HealthResponse(
        status="healthy" if models else "no_models",
        models_loaded=models_status,
        explainers_available=explainers_status,
        uptime_seconds=time.time() - startup_time
    )

@app.get("/models", response_model=ModelInfoResponse)
async def get_models_info():
    """Get information about available models."""
    best_model = get_best_model("auto")
    
    return ModelInfoResponse(
        available_models=model_info,
        default_model=best_model or "none",
        explainable_ai_available=len(explainers) > 0
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_text(
    input_data: TextInput,
    request: Request,
    _: bool = Depends(verify_token)
):
    """Predict if text violates content rules."""
    start_time = time.time()
    
    # Rate limiting
    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Get model
    model_key = get_best_model(input_data.model_type)
    if not model_key:
        raise HTTPException(status_code=503, detail="No models available")
    
    try:
        # Make prediction
        model = models[model_key]
        vectorizer = vectorizers[model_key]
        
        X = vectorizer.transform([input_data.text])
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[0]
            prediction = "violation" if probabilities[1] > 0.5 else "safe"
            confidence = max(probabilities)
        else:
            prediction_binary = model.predict(X)[0]
            prediction = "violation" if prediction_binary == 1 else "safe"
            confidence = 0.8  # Default confidence for models without predict_proba
            probabilities = [0.2, 0.8] if prediction_binary == 1 else [0.8, 0.2]
        
        # Prepare response
        response_data = {
            "text": input_data.text,
            "prediction": prediction,
            "confidence": float(confidence),
            "model_used": model_key,
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": (time.time() - start_time) * 1000
        }
        
        # Add probabilities if requested
        if input_data.include_probabilities:
            response_data["probabilities"] = {
                "safe": float(probabilities[0]),
                "violation": float(probabilities[1])
            }
        
        # Add explanation if requested and available
        if input_data.include_explanation and model_key in explainers and explainers[model_key]:
            try:
                explanation = explainers[model_key].explain_comprehensive(input_data.text)
                response_data["explanation"] = explanation
            except Exception as e:
                logger.error(f"Error generating explanation: {e}")
                response_data["explanation"] = {"error": "Explanation generation failed"}
        
        return PredictionResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    input_data: BatchTextInput,
    request: Request,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_token)
):
    """Predict multiple texts in batch."""
    start_time = time.time()
    
    # Rate limiting (stricter for batch)
    client_ip = request.client.host
    if not check_rate_limit(client_ip, limit=10, window=3600):
        raise HTTPException(status_code=429, detail="Batch rate limit exceeded")
    
    # Get model
    model_key = get_best_model(input_data.model_type)
    if not model_key:
        raise HTTPException(status_code=503, detail="No models available")
    
    try:
        model = models[model_key]
        vectorizer = vectorizers[model_key]
        explainer = explainers.get(model_key) if input_data.include_explanation else None
        
        results = []
        
        # Process each text
        for text in input_data.texts:
            text_start_time = time.time()
            
            X = vectorizer.transform([text])
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[0]
                prediction = "violation" if probabilities[1] > 0.5 else "safe"
                confidence = max(probabilities)
            else:
                prediction_binary = model.predict(X)[0]
                prediction = "violation" if prediction_binary == 1 else "safe"
                confidence = 0.8
                probabilities = [0.2, 0.8] if prediction_binary == 1 else [0.8, 0.2]
            
            result_data = {
                "text": text,
                "prediction": prediction,
                "confidence": float(confidence),
                "model_used": model_key,
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": (time.time() - text_start_time) * 1000
            }
            
            if input_data.include_probabilities:
                result_data["probabilities"] = {
                    "safe": float(probabilities[0]),
                    "violation": float(probabilities[1])
                }
            
            if input_data.include_explanation and explainer:
                try:
                    explanation = explainer.explain_comprehensive(text)
                    result_data["explanation"] = explanation
                except Exception as e:
                    result_data["explanation"] = {"error": "Explanation generation failed"}
            
            results.append(PredictionResponse(**result_data))
        
        return BatchPredictionResponse(
            results=results,
            total_processed=len(results),
            total_time_ms=(time.time() - start_time) * 1000,
            model_used=model_key,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict/multilingual", response_model=MultilingualPredictionResponse)
async def predict_multilingual(input_data: MultilingualTextInput, request: Request):
    """Predict content moderation for multilingual text with language detection and translation."""
    start_time = time.time()
    
    try:
        logger.info(f"Starting multilingual prediction for input: {type(input_data.text)}")
        logger.info(f"Input text preview: {str(input_data.text)[:50]}")
        logger.info(f"Input data attributes: {dir(input_data.text)}")
        
        # Check if input_data.text is somehow a sparse matrix
        if hasattr(input_data.text, 'toarray'):
            logger.error(f"CRITICAL ERROR: input_data.text is a sparse matrix: {type(input_data.text)}")
            logger.error(f"Sparse matrix shape: {input_data.text.shape}")
            logger.error(f"This should never happen - FastAPI should parse JSON text as string")
            raise HTTPException(status_code=400, detail="Invalid input: received sparse matrix instead of text string")
        
        # Simple language detection
        detected_lang = 'unknown'
        lang_confidence = 0.0
        translated_text = None
        
        if multilingual_moderator and multilingual_moderator.language_detector:
            try:
                logger.info(f"About to detect language for text type: {type(input_data.text)}")
                logger.info(f"Text content preview: {str(input_data.text)[:100]}")
                
                # Ensure we're passing a string, not a sparse matrix
                text_for_detection = str(input_data.text)
                if hasattr(input_data.text, 'toarray'):  # Check if it's a sparse matrix
                    logger.error(f"ERROR: Sparse matrix passed to language detection: {type(input_data.text)}")
                    raise ValueError("Sparse matrix passed to language detection")
                
                detected_lang, lang_confidence = multilingual_moderator.language_detector.detect_language(
                    text_for_detection, return_confidence=True
                )
                logger.info(f"Language detection successful: {detected_lang}, confidence: {lang_confidence}")
            except Exception as e:
                logger.error(f"Language detection failed: {e}")
                detected_lang = 'unknown'
                lang_confidence = 0.0
        
        # Translation if needed
        text_to_process = input_data.text
        if (input_data.auto_translate and multilingual_moderator and 
            multilingual_moderator.translator and detected_lang != input_data.target_language):
            try:
                # Ensure we're passing a string to translation
                text_for_translation = str(input_data.text)
                if hasattr(input_data.text, 'toarray'):  # Check if it's a sparse matrix
                    logger.error(f"ERROR: Sparse matrix passed to translation: {type(input_data.text)}")
                    raise ValueError("Sparse matrix passed to translation")
                
                translation_result = multilingual_moderator.translator.translate_text(
                    text_for_translation, detected_lang
                )
                translated_text = translation_result.get('translated_text')
                if translated_text:
                    text_to_process = translated_text
            except Exception as e:
                logger.error(f"Translation failed: {e}")
                pass  # Use original text if translation fails
        
        # Get model and vectorizer - prefer multilingual models
        model_key = get_best_model("multilingual" if input_data.model_type == "auto" else input_data.model_type)
        if not model_key:
            model_key = get_best_model("auto")  # Fallback to any available model
        if not model_key:
            raise HTTPException(status_code=503, detail="No models available")
        
        model = models[model_key]
        vectorizer = vectorizers[model_key]
        explainer = explainers.get(model_key)
        
        # Make prediction
        logger.info(f"About to make prediction with text: {type(text_to_process)}")
        logger.info(f"Text preview: {str(text_to_process)[:50]}")
        logger.info(f"Using model: {model_key}")
        logger.info(f"Model type: {type(model)}")
        
        # Check if model is a pipeline (contains TF-IDF step)
        is_pipeline = hasattr(model, 'steps')
        logger.info(f"Model is pipeline: {is_pipeline}")
        
        if is_pipeline:
            logger.info(f"Pipeline steps: {[step[0] for step in model.steps]}")
            # For pipeline models, pass raw text directly
            model_input = [text_to_process]
        else:
            # For non-pipeline models, use vectorizer
            logger.info(f"Vectorizer type: {type(vectorizer)}")
            model_input = vectorizer.transform([text_to_process])
            logger.info(f"Vectorized input type: {type(model_input)}, shape: {model_input.shape}")
        
        # Add detailed debugging for model prediction
        logger.info(f"About to call model prediction with input type: {type(model_input)}")
        try:
            if hasattr(model, 'predict_proba'):
                logger.info("Calling model.predict_proba...")
                probabilities = model.predict_proba(model_input)[0]
                prediction = "violation" if probabilities[1] > 0.5 else "safe"
                confidence = max(probabilities)
                logger.info(f"predict_proba successful, probabilities: {probabilities}")
            else:
                logger.info("Calling model.predict...")
                prediction_binary = model.predict(model_input)[0]
                prediction = "violation" if prediction_binary == 1 else "safe"
                confidence = 0.8
                probabilities = [0.2, 0.8] if prediction_binary == 1 else [0.8, 0.2]
                logger.info(f"predict successful, prediction: {prediction_binary}")
        except Exception as pred_error:
            logger.error(f"Model prediction failed: {pred_error}")
            logger.error(f"Model attributes: {dir(model)}")
            if hasattr(model, 'steps'):
                logger.error(f"Model is a pipeline with steps: {[step[0] for step in model.steps]}")
            raise pred_error
        
        result_data = {
            "original_text": input_data.text,
            "detected_language": detected_lang,
            "language_confidence": float(lang_confidence),
            "translated_text": translated_text,
            "prediction": prediction,
            "confidence": float(confidence),
            "model_used": model_key,
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": (time.time() - start_time) * 1000
        }
        
        if input_data.include_probabilities:
            result_data["probabilities"] = {
                "safe": float(probabilities[0]),
                "violation": float(probabilities[1])
            }
        
        if input_data.include_explanation and explainer:
            try:
                # Temporarily disable explanations to isolate the issue
                # explanation = explainer.explain_comprehensive(text_to_process)
                # result_data["explanation"] = explanation
                result_data["explanation"] = {"info": "Explanations temporarily disabled for debugging"}
            except Exception as e:
                result_data["explanation"] = {"error": "Explanation generation failed"}
        
        return MultilingualPredictionResponse(**result_data)
        
    except Exception as e:
        logger.error(f"Multilingual prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Multilingual prediction failed: {str(e)}")

@app.get("/multilingual/stats", response_model=LanguageStatsResponse)
async def get_multilingual_stats():
    """Get multilingual processing statistics."""
    if not multilingual_moderator:
        raise HTTPException(status_code=503, detail="Multilingual support not available")
    
    stats = multilingual_moderator.get_language_statistics()
    total_processed = sum(count['count'] if isinstance(count, dict) else count for count in stats.values())
    
    return LanguageStatsResponse(
        total_texts_processed=total_processed,
        language_distribution=stats,
        supported_languages=multilingual_moderator.get_supported_languages(),
        translation_enabled=True
    )

@app.get("/stats")
async def get_stats():
    """Get API usage statistics."""
    return {
        "total_requests": sum(len(requests) for requests in rate_limit_storage.values()),
        "active_clients": len(rate_limit_storage),
        "models_loaded": len(models),
        "explainers_available": len(explainers),
        "uptime_seconds": time.time() - startup_time
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )