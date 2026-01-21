"""
Advanced model utilities for QLoRA fine-tuning.
Addresses compatibility issues with latest model architectures.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
import logging
from transformers import AutoConfig, AutoModelForCausalLM
import warnings

logger = logging.getLogger(__name__)

class ModelCompatibilityManager:
    """Handles compatibility issues across different model architectures."""
    
    # Model-specific configurations to fix known issues
    MODEL_FIXES = {
        "qwen": {
            "config_updates": {
                "use_cache": False,
                "pretraining_tp": 1,
            },
            "attention_fix": True,
            "embedding_fix": True
        },
        "llama": {
            "config_updates": {
                "use_cache": False,
            },
            "rope_scaling_fix": True,
            "attention_fix": True
        },
        "mistral": {
            "config_updates": {
                "use_cache": False,
            },
            "sliding_window_fix": True
        },
        "gemma": {
            "config_updates": {
                "use_cache": False,
                "attn_logit_softcapping": None,  # Fix for Gemma 2
            },
            "normalization_fix": True
        },
        "phi": {
            "config_updates": {
                "use_cache": False,
            },
            "attention_bias_fix": True
        }
    }
    
    @classmethod
    def detect_model_family(cls, model_name: str) -> str:
        """Detect model family from name."""
        model_name_lower = model_name.lower()
        
        if any(x in model_name_lower for x in ["qwen", "qwen2", "qwen2.5"]):
            return "qwen"
        elif any(x in model_name_lower for x in ["llama", "alpaca", "vicuna"]):
            return "llama"
        elif any(x in model_name_lower for x in ["mistral", "mixtral"]):
            return "mistral"
        elif "gemma" in model_name_lower:
            return "gemma"
        elif "phi" in model_name_lower:
            return "phi"
        else:
            logger.warning(f"Unknown model family for {model_name}, using llama defaults")
            return "llama"
    
    @classmethod
    def apply_model_fixes(cls, model: nn.Module, model_name: str) -> nn.Module:
        """Apply model-specific fixes to address known issues."""
        model_family = cls.detect_model_family(model_name)
        fixes = cls.MODEL_FIXES.get(model_family, {})
        
        # Apply config updates
        config_updates = fixes.get("config_updates", {})
        for key, value in config_updates.items():
            if hasattr(model.config, key):
                setattr(model.config, key, value)
                logger.info(f"Applied config fix: {key} = {value}")
        
        # Apply specific fixes
        if fixes.get("attention_fix"):
            cls._fix_attention_implementation(model, model_family)
        
        if fixes.get("embedding_fix"):
            cls._fix_embedding_issues(model)
        
        if fixes.get("rope_scaling_fix"):
            cls._fix_rope_scaling(model)
        
        if fixes.get("sliding_window_fix"):
            cls._fix_sliding_window(model)
        
        if fixes.get("normalization_fix"):
            cls._fix_normalization(model)
        
        if fixes.get("attention_bias_fix"):
            cls._fix_attention_bias(model)
        
        return model
    
    @staticmethod
    def _fix_attention_implementation(model: nn.Module, model_family: str):
        """Fix attention implementation issues."""
        try:
            # Ensure proper attention implementation
            if hasattr(model.config, '_attn_implementation'):
                if model.config._attn_implementation == "flash_attention_2":
                    # Verify flash attention is properly loaded
                    try:
                        import flash_attn
                        logger.info("Flash Attention 2 verified and active")
                    except ImportError:
                        logger.warning("Flash Attention 2 requested but not available, falling back to eager")
                        model.config._attn_implementation = "eager"
        except Exception as e:
            logger.warning(f"Could not fix attention implementation: {e}")
    
    @staticmethod
    def _fix_embedding_issues(model: nn.Module):
        """Fix embedding-related issues."""
        try:
            # Ensure embeddings are properly initialized
            if hasattr(model, 'get_input_embeddings'):
                embeddings = model.get_input_embeddings()
                if embeddings is not None and hasattr(embeddings, 'weight'):
                    # Check for NaN or inf values
                    if torch.isnan(embeddings.weight).any() or torch.isinf(embeddings.weight).any():
                        logger.warning("Found NaN/inf in embeddings, reinitializing...")
                        nn.init.normal_(embeddings.weight, mean=0.0, std=0.02)
        except Exception as e:
            logger.warning(f"Could not fix embedding issues: {e}")
    
    @staticmethod
    def _fix_rope_scaling(model: nn.Module):
        """Fix RoPE scaling issues for LLaMA models."""
        try:
            if hasattr(model.config, 'rope_scaling') and model.config.rope_scaling is not None:
                # Ensure rope_scaling is properly configured
                rope_scaling = model.config.rope_scaling
                if isinstance(rope_scaling, dict):
                    if 'type' not in rope_scaling:
                        rope_scaling['type'] = 'linear'
                    if 'factor' not in rope_scaling:
                        rope_scaling['factor'] = 1.0
                    logger.info(f"Fixed RoPE scaling: {rope_scaling}")
        except Exception as e:
            logger.warning(f"Could not fix RoPE scaling: {e}")
    
    @staticmethod
    def _fix_sliding_window(model: nn.Module):
        """Fix sliding window attention for Mistral models."""
        try:
            if hasattr(model.config, 'sliding_window'):
                # Ensure sliding window is properly set
                if model.config.sliding_window is None:
                    model.config.sliding_window = 4096
                    logger.info("Set default sliding window to 4096")
        except Exception as e:
            logger.warning(f"Could not fix sliding window: {e}")
    
    @staticmethod
    def _fix_normalization(model: nn.Module):
        """Fix normalization issues for Gemma models."""
        try:
            # Gemma uses RMSNorm, ensure it's properly configured
            if hasattr(model.config, 'rms_norm_eps'):
                if model.config.rms_norm_eps is None:
                    model.config.rms_norm_eps = 1e-6
                    logger.info("Set default RMS norm epsilon")
        except Exception as e:
            logger.warning(f"Could not fix normalization: {e}")
    
    @staticmethod
    def _fix_attention_bias(model: nn.Module):
        """Fix attention bias issues for Phi models."""
        try:
            # Phi models may have attention bias configuration issues
            if hasattr(model.config, 'attention_bias'):
                # Ensure attention bias is properly configured
                pass  # Model-specific fixes can be added here
        except Exception as e:
            logger.warning(f"Could not fix attention bias: {e}")

class QuantizationOptimizer:
    """Advanced quantization optimizations."""
    
    @staticmethod
    def optimize_bnb_config(model_name: str, available_memory_gb: float) -> Dict[str, Any]:
        """Generate optimized BitsAndBytes configuration based on model and hardware."""
        
        # Estimate model size (rough heuristic)
        if "405b" in model_name.lower():
            estimated_size_gb = 800
        elif any(x in model_name.lower() for x in ["70b", "72b"]):
            estimated_size_gb = 140
        elif any(x in model_name.lower() for x in ["30b", "33b", "34b"]):
            estimated_size_gb = 60
        elif any(x in model_name.lower() for x in ["13b", "14b", "15b"]):
            estimated_size_gb = 26
        elif any(x in model_name.lower() for x in ["7b", "8b", "9b"]):
            estimated_size_gb = 14
        else:
            estimated_size_gb = 14  # Default assumption
        
        # Choose quantization strategy
        if estimated_size_gb > available_memory_gb * 0.8:
            # Aggressive quantization needed
            return {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_storage": torch.uint8,
                "llm_int8_enable_fp32_cpu_offload": True,
            }
        elif estimated_size_gb > available_memory_gb * 0.6:
            # Standard 4-bit quantization
            return {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            }
        else:
            # Light quantization or no quantization
            return {
                "load_in_8bit": True,
                "llm_int8_threshold": 6.0,
            }

class MemoryProfiler:
    """Memory profiling and optimization utilities."""
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """Get detailed GPU memory information."""
        if not torch.cuda.is_available():
            return {"total_gb": 0, "available_gb": 0, "used_gb": 0}
        
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        
        total_gb = total_memory / (1024**3)
        allocated_gb = allocated_memory / (1024**3)
        cached_gb = cached_memory / (1024**3)
        available_gb = total_gb - cached_gb
        
        return {
            "total_gb": total_gb,
            "allocated_gb": allocated_gb,
            "cached_gb": cached_gb,
            "available_gb": available_gb,
            "utilization": (cached_gb / total_gb) * 100
        }
    
    @staticmethod
    def optimize_batch_size(model: nn.Module, tokenizer, max_seq_length: int = 2048) -> int:
        """Dynamically determine optimal batch size."""
        memory_info = MemoryProfiler.get_gpu_memory_info()
        available_gb = memory_info["available_gb"]
        
        if available_gb < 8:
            return 1
        elif available_gb < 16:
            return 2
        elif available_gb < 32:
            return 4
        else:
            return 8

def find_all_linear_names(model: nn.Module, exclude_lm_head: bool = True) -> List[str]:
    """
    Find all linear layer names in the model for LoRA targeting.
    Enhanced version that handles different model architectures.
    """
    import bitsandbytes as bnb
    
    # Determine the linear layer class based on quantization
    if hasattr(model, 'hf_device_map'):
        # Model is quantized
        linear_cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    else:
        linear_cls = (nn.Linear,)
    
    lora_module_names = set()
    
    for name, module in model.named_modules():
        if isinstance(module, linear_cls):
            # Extract the last part of the name
            names = name.split('.')
            lora_module_names.add(names[-1])
    
    # Common exclusions
    exclusions = {'lm_head'} if exclude_lm_head else set()
    
    # Model-specific exclusions
    model_name = getattr(model.config, '_name_or_path', '').lower()
    if 'qwen' in model_name:
        exclusions.update({'wte', 'ln_f'})
    elif 'gemma' in model_name:
        exclusions.update({'embed_tokens', 'norm'})
    
    # Remove exclusions
    lora_module_names = lora_module_names - exclusions
    
    logger.info(f"Found LoRA target modules: {sorted(lora_module_names)}")
    return list(lora_module_names)

def setup_model_for_training(
    model: nn.Module,
    model_name: str,
    use_gradient_checkpointing: bool = True,
    use_flash_attention: bool = True
) -> nn.Module:
    """
    Setup model for efficient training with all optimizations.
    """
    
    # Apply model-specific fixes
    model = ModelCompatibilityManager.apply_model_fixes(model, model_name)
    
    # Enable gradient checkpointing for memory efficiency
    if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Setup flash attention if available and requested
    if use_flash_attention:
        try:
            import flash_attn
            if hasattr(model.config, '_attn_implementation'):
                model.config._attn_implementation = "flash_attention_2"
                logger.info("Flash Attention 2 enabled")
        except ImportError:
            logger.warning("Flash Attention not available")
    
    # Optimize for training
    model.train()
    
    # Enable TF32 for better performance on Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    return model

class ModelValidator:
    """Validate model configuration and catch common issues."""
    
    @staticmethod
    def validate_model_config(model: nn.Module, tokenizer) -> List[str]:
        """Validate model configuration and return list of issues found."""
        issues = []
        
        # Check tokenizer compatibility
        if tokenizer.pad_token is None:
            issues.append("Tokenizer missing pad_token")
        
        if tokenizer.eos_token is None:
            issues.append("Tokenizer missing eos_token")
        
        # Check model configuration
        if hasattr(model.config, 'vocab_size'):
            if len(tokenizer) != model.config.vocab_size:
                issues.append(f"Tokenizer vocab size ({len(tokenizer)}) != model vocab size ({model.config.vocab_size})")
        
        # Check for common configuration issues
        if hasattr(model.config, 'use_cache') and model.config.use_cache:
            issues.append("use_cache should be False for training")
        
        # Check quantization compatibility
        if hasattr(model, 'hf_device_map'):
            # Model is quantized, check for common issues
            for name, param in model.named_parameters():
                if param.requires_grad and param.dtype not in [torch.float16, torch.bfloat16, torch.float32]:
                    issues.append(f"Trainable parameter {name} has unexpected dtype: {param.dtype}")
        
        return issues
    
    @staticmethod
    def fix_common_issues(model: nn.Module, tokenizer, issues: List[str]) -> tuple:
        """Attempt to fix common issues automatically."""
        fixed_issues = []
        
        for issue in issues:
            if "missing pad_token" in issue:
                tokenizer.pad_token = tokenizer.eos_token
                fixed_issues.append("Set pad_token to eos_token")
            
            elif "use_cache should be False" in issue:
                model.config.use_cache = False
                fixed_issues.append("Set use_cache to False")
            
            elif "vocab size" in issue:
                model.resize_token_embeddings(len(tokenizer))
                fixed_issues.append("Resized token embeddings")
        
        return model, tokenizer, fixed_issues