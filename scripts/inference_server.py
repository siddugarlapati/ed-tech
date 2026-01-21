#!/usr/bin/env python3
"""
Production inference server for Advanced QLoRA models.
Optimized for deployment with FastAPI and comprehensive model support.
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import time
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, GenerationConfig
)
from peft import PeftModel
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import psutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_utils import ModelCompatibilityManager, MemoryProfiler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response Models
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    max_new_tokens: int = Field(default=256, ge=1, le=2048, description="Maximum new tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(default=50, ge=1, le=100, description="Top-k sampling parameter")
    do_sample: bool = Field(default=True, description="Whether to use sampling")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Repetition penalty")
    stream: bool = Field(default=False, description="Whether to stream the response")

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    generation_time: float
    tokens_generated: int
    tokens_per_second: float

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    quantization: str
    memory_usage_gb: float
    device: str
    lora_adapters: List[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    memory_usage: Dict[str, float]
    uptime_seconds: float

# Global model state
class ModelState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.adapter_paths = []
        self.generation_config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.start_time = time.time()
        self.request_count = 0

model_state = ModelState()

# FastAPI app
app = FastAPI(
    title="Advanced QLoRA Inference Server",
    description="Production-ready inference server for QLoRA fine-tuned models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelManager:
    """Manages model loading and inference operations."""
    
    @staticmethod
    def load_model(
        model_name_or_path: str,
        adapter_paths: Optional[List[str]] = None,
        quantization_config: Optional[Dict[str, Any]] = None,
        device_map: str = "auto"
    ):
        """Load model with optional LoRA adapters."""
        logger.info(f"Loading model: {model_name_or_path}")
        
        # Configure quantization
        if quantization_config:
            bnb_config = BitsAndBytesConfig(**quantization_config)
        else:
            # Default 4-bit quantization for inference
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left"  # For batch inference
        )
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Apply model fixes
        model = ModelCompatibilityManager.apply_model_fixes(model, model_name_or_path)
        
        # Load LoRA adapters if provided
        if adapter_paths:
            for adapter_path in adapter_paths:
                logger.info(f"Loading LoRA adapter: {adapter_path}")
                model = PeftModel.from_pretrained(model, adapter_path)
        
        # Set to evaluation mode
        model.eval()
        
        # Create generation config
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
        
        logger.info("Model loaded successfully")
        return model, tokenizer, generation_config
    
    @staticmethod
    def generate_text(
        model,
        tokenizer,
        prompt: str,
        generation_config: GenerationConfig,
        max_new_tokens: int = 256,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text with comprehensive error handling."""
        start_time = time.time()
        
        try:
            # Update generation config with request parameters
            gen_config = GenerationConfig(
                **generation_config.to_dict(),
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(model.device)
            
            input_length = inputs.input_ids.shape[1]
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=gen_config,
                    use_cache=True
                )
            
            # Decode output
            generated_tokens = outputs[0][input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            tokens_generated = len(generated_tokens)
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            return {
                "generated_text": generated_text,
                "prompt": prompt,
                "generation_time": generation_time,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_per_second
            }
            
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory during generation")
            torch.cuda.empty_cache()
            raise HTTPException(status_code=500, detail="GPU out of memory")
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    logger.info("Starting Advanced QLoRA Inference Server")
    
    # Load model configuration from environment or config file
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B")
    adapter_paths = os.getenv("ADAPTER_PATHS", "").split(",") if os.getenv("ADAPTER_PATHS") else None
    
    try:
        model, tokenizer, generation_config = ModelManager.load_model(
            model_name,
            adapter_paths=adapter_paths
        )
        
        model_state.model = model
        model_state.tokenizer = tokenizer
        model_state.model_name = model_name
        model_state.adapter_paths = adapter_paths or []
        model_state.generation_config = generation_config
        
        logger.info("Model loaded successfully on startup")
        
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # Continue startup but mark model as not loaded

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    memory_info = MemoryProfiler.get_gpu_memory_info() if torch.cuda.is_available() else {}
    
    return HealthResponse(
        status="healthy" if model_state.model is not None else "model_not_loaded",
        model_loaded=model_state.model is not None,
        memory_usage=memory_info,
        uptime_seconds=time.time() - model_state.start_time
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    if model_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    memory_info = MemoryProfiler.get_gpu_memory_info()
    
    return ModelInfo(
        model_name=model_state.model_name,
        model_type=type(model_state.model).__name__,
        quantization="4-bit NF4",
        memory_usage_gb=memory_info.get("allocated_gb", 0),
        device=str(model_state.device),
        lora_adapters=model_state.adapter_paths
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate text from the loaded model."""
    if model_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Increment request counter
    model_state.request_count += 1
    
    # Generate text
    result = ModelManager.generate_text(
        model=model_state.model,
        tokenizer=model_state.tokenizer,
        prompt=request.prompt,
        generation_config=model_state.generation_config,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        do_sample=request.do_sample,
        repetition_penalty=request.repetition_penalty
    )
    
    # Schedule memory cleanup
    background_tasks.add_task(cleanup_memory)
    
    return GenerationResponse(**result)

@app.post("/model/load")
async def load_model(
    model_name: str,
    adapter_paths: Optional[List[str]] = None,
    quantization_config: Optional[Dict[str, Any]] = None
):
    """Load a new model (admin endpoint)."""
    try:
        model, tokenizer, generation_config = ModelManager.load_model(
            model_name,
            adapter_paths=adapter_paths,
            quantization_config=quantization_config
        )
        
        # Update global state
        model_state.model = model
        model_state.tokenizer = tokenizer
        model_state.model_name = model_name
        model_state.adapter_paths = adapter_paths or []
        model_state.generation_config = generation_config
        
        return {"status": "success", "message": f"Model {model_name} loaded successfully"}
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/model/unload")
async def unload_model():
    """Unload the current model to free memory."""
    if model_state.model is not None:
        del model_state.model
        del model_state.tokenizer
        model_state.model = None
        model_state.tokenizer = None
        model_state.generation_config = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {"status": "success", "message": "Model unloaded successfully"}
    else:
        return {"status": "info", "message": "No model was loaded"}

@app.get("/stats")
async def get_stats():
    """Get server statistics."""
    memory_info = MemoryProfiler.get_gpu_memory_info() if torch.cuda.is_available() else {}
    
    return {
        "uptime_seconds": time.time() - model_state.start_time,
        "request_count": model_state.request_count,
        "model_loaded": model_state.model is not None,
        "memory_usage": memory_info,
        "cpu_usage": psutil.cpu_percent(),
        "ram_usage": psutil.virtual_memory().percent
    }

async def cleanup_memory():
    """Background task to clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# CLI for running the server
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced QLoRA Inference Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", type=str, help="Model name or path to load on startup")
    parser.add_argument("--adapters", type=str, nargs="+", help="LoRA adapter paths")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", type=str, default="info", 
                       choices=["debug", "info", "warning", "error"])
    
    args = parser.parse_args()
    
    # Set environment variables for startup
    if args.model:
        os.environ["MODEL_NAME"] = args.model
    if args.adapters:
        os.environ["ADAPTER_PATHS"] = ",".join(args.adapters)
    
    # Run server
    uvicorn.run(
        "inference_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        reload=False
    )

if __name__ == "__main__":
    main()