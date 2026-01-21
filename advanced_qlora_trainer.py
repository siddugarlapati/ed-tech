#!/usr/bin/env python3
"""
Advanced QLoRA Fine-Tuning System
Production-ready implementation with enterprise features and comprehensive model support.

Author: Advanced ML Engineer
License: MIT
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig, set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging as hf_logging

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    PeftModel, TaskType
)
import bitsandbytes as bnb
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

# Optional imports for enhanced features
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available. Install for experiment tracking.")

try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    warnings.warn("flash-attn not available. Install for 2-4x speedup.")

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup enhanced logging with rich formatting if available."""
    handlers = []
    
    if RICH_AVAILABLE:
        handlers.append(RichHandler(rich_tracebacks=True))
    else:
        handlers.append(logging.StreamHandler())
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    
    # Suppress some verbose logs
    hf_logging.set_verbosity_error()
    transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Enhanced model configuration with support for latest architectures."""
    model_name_or_path: str = "meta-llama/Llama-3.1-8B"
    trust_remote_code: bool = False
    use_auth_token: Optional[str] = None
    torch_dtype: str = "auto"  # auto, float16, bfloat16, float32
    attn_implementation: str = "flash_attention_2"  # eager, sdpa, flash_attention_2
    
    # Quantization settings
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"  # fp4, nf4
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_storage: str = "uint8"
    
    # Memory optimization
    max_memory_mb: Optional[int] = None
    device_map: str = "auto"
    low_cpu_mem_usage: bool = True

@dataclass 
class LoRAConfig:
    """Enhanced LoRA configuration."""
    r: int = 64
    alpha: int = 16
    dropout: float = 0.1
    bias: str = "none"  # none, all, lora_only
    task_type: str = "CAUSAL_LM"
    target_modules: Optional[List[str]] = None
    modules_to_save: Optional[List[str]] = None
    
    # Advanced LoRA features
    use_rslora: bool = False  # Rank-Stabilized LoRA
    use_dora: bool = False    # Weight-Decomposed Low-Rank Adaptation
    init_lora_weights: Union[bool, str] = True

@dataclass
class DataConfig:
    """Data processing configuration."""
    dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str = "test"
    
    # Text processing
    max_seq_length: int = 2048
    truncation: bool = True
    padding: str = "max_length"
    
    # Data filtering and sampling
    min_length: int = 10
    max_length: int = 4096
    filter_duplicates: bool = True
    sample_ratio: Optional[float] = None

@dataclass
class TrainingConfig:
    """Enhanced training configuration."""
    output_dir: str = "./output"
    run_name: Optional[str] = None
    
    # Training hyperparameters
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    
    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate scheduling
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    
    # Evaluation and saving
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Memory and performance
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True
    
    # Logging and monitoring
    logging_steps: int = 10
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # Advanced features
    group_by_length: bool = True
    length_column_name: str = "length"
    remove_unused_columns: bool = False
    
    # Optimization flags
    optim: str = "paged_adamw_8bit"
    neftune_noise_alpha: Optional[float] = None

class ModelRegistry:
    """Registry for supported model architectures with optimized configurations."""
    
    SUPPORTED_MODELS = {
        # LLaMA Family
        "llama": {
            "patterns": ["llama", "alpaca", "vicuna", "guanaco"],
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "modules_to_save": ["embed_tokens", "lm_head"],
            "special_tokens": {"pad_token": "<pad>"}
        },
        
        # Qwen Family  
        "qwen": {
            "patterns": ["qwen", "qwen2", "qwen2.5"],
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "modules_to_save": ["embed_tokens", "lm_head"],
            "special_tokens": {"pad_token": "<|endoftext|>"}
        },
        
        # Mistral Family
        "mistral": {
            "patterns": ["mistral", "mixtral"],
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "modules_to_save": ["embed_tokens", "lm_head"],
            "special_tokens": {"pad_token": "</s>"}
        },
        
        # Gemma Family
        "gemma": {
            "patterns": ["gemma"],
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "modules_to_save": ["embed_tokens", "lm_head"],
            "special_tokens": {"pad_token": "<pad>"}
        },
        
        # Phi Family
        "phi": {
            "patterns": ["phi"],
            "target_modules": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
            "modules_to_save": ["embed_tokens", "lm_head"],
            "special_tokens": {"pad_token": "<|endoftext|>"}
        },
        
        # T5 Family
        "t5": {
            "patterns": ["t5", "flan-t5"],
            "target_modules": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
            "modules_to_save": ["shared", "lm_head"],
            "special_tokens": {"pad_token": "<pad>"}
        }
    }
    
    @classmethod
    def detect_model_type(cls, model_name: str) -> str:
        """Detect model architecture from model name."""
        model_name_lower = model_name.lower()
        
        for model_type, config in cls.SUPPORTED_MODELS.items():
            for pattern in config["patterns"]:
                if pattern in model_name_lower:
                    return model_type
        
        logger.warning(f"Unknown model type for {model_name}, using default llama config")
        return "llama"
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """Get optimized configuration for model type."""
        model_type = cls.detect_model_type(model_name)
        return cls.SUPPORTED_MODELS[model_type]

class AdvancedDataCollator:
    """Enhanced data collator with dynamic padding and length grouping."""
    
    def __init__(self, tokenizer, max_length: int = 2048, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Dynamic padding to the longest sequence in batch
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length
        )
        
        # Pad to multiple for tensor core optimization
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) 
                      // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        batch = {}
        for key in features[0].keys():
            if key == "input_ids":
                batch[key] = torch.stack([
                    torch.cat([
                        torch.tensor(f[key][:max_len]),
                        torch.full((max_len - len(f[key][:max_len]),), 
                                 self.tokenizer.pad_token_id)
                    ]) for f in features
                ])
            elif key == "attention_mask":
                batch[key] = torch.stack([
                    torch.cat([
                        torch.ones(len(f["input_ids"][:max_len])),
                        torch.zeros(max_len - len(f["input_ids"][:max_len]))
                    ]) for f in features
                ])
            elif key == "labels":
                batch[key] = torch.stack([
                    torch.cat([
                        torch.tensor(f[key][:max_len]),
                        torch.full((max_len - len(f[key][:max_len]),), -100)
                    ]) for f in features
                ])
        
        return batch

class MemoryOptimizer:
    """Advanced memory optimization utilities."""
    
    @staticmethod
    def get_optimal_batch_size(model, tokenizer, max_length: int = 2048) -> int:
        """Dynamically determine optimal batch size based on available memory."""
        if not torch.cuda.is_available():
            return 1
            
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        
        # Estimate memory usage per sample (rough heuristic)
        bytes_per_param = 2 if model.dtype == torch.float16 else 4
        model_memory = sum(p.numel() for p in model.parameters()) * bytes_per_param
        
        # Reserve memory for gradients, optimizer states, and activations
        available_memory = total_memory * 0.7 - model_memory * 3
        
        # Estimate memory per sample
        memory_per_sample = max_length * model.config.hidden_size * bytes_per_param * 4
        
        optimal_batch_size = max(1, int(available_memory // memory_per_sample))
        logger.info(f"Estimated optimal batch size: {optimal_batch_size}")
        
        return min(optimal_batch_size, 32)  # Cap at reasonable maximum
    
    @staticmethod
    def setup_memory_efficient_attention(model):
        """Setup memory efficient attention mechanisms."""
        if hasattr(model.config, 'use_flash_attention_2'):
            model.config.use_flash_attention_2 = FLASH_ATTN_AVAILABLE
            
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

class AdvancedQLoRATrainer:
    """Production-ready QLoRA trainer with enterprise features."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        data_config: DataConfig,
        training_config: TrainingConfig,
        experiment_name: Optional[str] = None
    ):
        self.model_config = model_config
        self.lora_config = lora_config
        self.data_config = data_config
        self.training_config = training_config
        self.experiment_name = experiment_name or f"qlora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize components
        self.accelerator = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Setup experiment tracking
        self._setup_experiment_tracking()
        
    def _setup_experiment_tracking(self):
        """Initialize experiment tracking and logging."""
        if WANDB_AVAILABLE and "wandb" in self.training_config.report_to:
            wandb.init(
                project="advanced-qlora",
                name=self.experiment_name,
                config={
                    "model": self.model_config.__dict__,
                    "lora": self.lora_config.__dict__,
                    "data": self.data_config.__dict__,
                    "training": self.training_config.__dict__
                }
            )
    
    def load_model_and_tokenizer(self) -> Tuple[nn.Module, transformers.PreTrainedTokenizer]:
        """Load and configure model with advanced optimizations."""
        logger.info(f"Loading model: {self.model_config.model_name_or_path}")
        
        # Detect model architecture and get optimized config
        model_arch_config = ModelRegistry.get_model_config(self.model_config.model_name_or_path)
        
        # Configure quantization
        if self.model_config.load_in_4bit or self.model_config.load_in_8bit:
            compute_dtype = getattr(torch, self.model_config.bnb_4bit_compute_dtype)
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.model_config.load_in_4bit,
                load_in_8bit=self.model_config.load_in_8bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.model_config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.model_config.bnb_4bit_use_double_quant,
                bnb_4bit_quant_storage=getattr(torch, self.model_config.bnb_4bit_quant_storage)
            )
        else:
            quantization_config = None
        
        # Configure torch dtype
        if self.model_config.torch_dtype == "auto":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = getattr(torch, self.model_config.torch_dtype)
        
        # Load model with optimizations
        model_kwargs = {
            "pretrained_model_name_or_path": self.model_config.model_name_or_path,
            "quantization_config": quantization_config,
            "device_map": self.model_config.device_map,
            "torch_dtype": torch_dtype,
            "trust_remote_code": self.model_config.trust_remote_code,
            "low_cpu_mem_usage": self.model_config.low_cpu_mem_usage,
        }
        
        # Add attention implementation if supported
        try:
            if self.model_config.attn_implementation == "flash_attention_2" and FLASH_ATTN_AVAILABLE:
                model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception as e:
            logger.warning(f"Could not set flash attention: {e}")
        
        # Handle memory constraints
        if self.model_config.max_memory_mb:
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            max_memory = {i: f"{self.model_config.max_memory_mb}MB" for i in range(n_gpus)}
            model_kwargs["max_memory"] = max_memory
        
        try:
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback without advanced features
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name_or_path,
            trust_remote_code=self.model_config.trust_remote_code,
            padding_side="right",
            use_fast=True
        )
        
        # Configure special tokens
        special_tokens = model_arch_config.get("special_tokens", {})
        if tokenizer.pad_token is None:
            if "pad_token" in special_tokens:
                tokenizer.pad_token = special_tokens["pad_token"]
            else:
                tokenizer.pad_token = tokenizer.eos_token
        
        # Resize embeddings if needed
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        
        # Setup memory efficient attention
        MemoryOptimizer.setup_memory_efficient_attention(model)
        
        # Prepare model for k-bit training
        if quantization_config:
            model = prepare_model_for_kbit_training(
                model, 
                use_gradient_checkpointing=self.training_config.gradient_checkpointing
            )
        
        self.model = model
        self.tokenizer = tokenizer
        
        logger.info(f"Model loaded successfully. Vocab size: {len(tokenizer)}")
        return model, tokenizer
    
    def setup_lora(self, model: nn.Module) -> nn.Module:
        """Configure and apply LoRA adapters."""
        logger.info("Setting up LoRA configuration")
        
        # Get model-specific target modules
        model_arch_config = ModelRegistry.get_model_config(self.model_config.model_name_or_path)
        target_modules = self.lora_config.target_modules or model_arch_config["target_modules"]
        modules_to_save = self.lora_config.modules_to_save or model_arch_config.get("modules_to_save")
        
        # Create LoRA config
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            bias=self.lora_config.bias,
            task_type=getattr(TaskType, self.lora_config.task_type),
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            init_lora_weights=self.lora_config.init_lora_weights
        )
        
        # Apply advanced LoRA features if available
        if hasattr(peft_config, 'use_rslora') and self.lora_config.use_rslora:
            peft_config.use_rslora = True
            
        if hasattr(peft_config, 'use_dora') and self.lora_config.use_dora:
            peft_config.use_dora = True
        
        # Apply LoRA to model
        model = get_peft_model(model, peft_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        
        return model
    
    def prepare_dataset(self) -> Dict[str, Dataset]:
        """Load and preprocess datasets with advanced features."""
        logger.info("Preparing datasets")
        
        # Load dataset
        if self.data_config.dataset_path:
            if self.data_config.dataset_path.endswith('.json'):
                with open(self.data_config.dataset_path, 'r') as f:
                    data = json.load(f)
                dataset = Dataset.from_list(data)
            elif self.data_config.dataset_path.endswith('.jsonl'):
                dataset = Dataset.from_json(self.data_config.dataset_path)
            else:
                dataset = load_dataset(self.data_config.dataset_path)
        elif self.data_config.dataset_name:
            dataset = load_dataset(
                self.data_config.dataset_name,
                self.data_config.dataset_config
            )
        else:
            raise ValueError("Either dataset_path or dataset_name must be provided")
        
        # Handle dataset splits
        if isinstance(dataset, dict):
            train_dataset = dataset.get(self.data_config.train_split)
            eval_dataset = dataset.get(self.data_config.eval_split)
        else:
            # Split single dataset
            split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
        
        # Apply data filtering and preprocessing
        if self.data_config.filter_duplicates:
            train_dataset = train_dataset.filter(
                lambda x: len(set(x.values())) == len(x.values())
            )
        
        # Sample data if specified
        if self.data_config.sample_ratio:
            train_size = int(len(train_dataset) * self.data_config.sample_ratio)
            train_dataset = train_dataset.select(range(train_size))
        
        # Tokenize datasets
        def tokenize_function(examples):
            # Handle different input formats
            if 'text' in examples:
                texts = examples['text']
            elif 'input' in examples and 'output' in examples:
                texts = [f"{inp}\n{out}" for inp, out in zip(examples['input'], examples['output'])]
            else:
                raise ValueError("Dataset must contain 'text' or 'input'/'output' columns")
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=self.data_config.truncation,
                padding=False,  # We'll pad dynamically
                max_length=self.data_config.max_seq_length,
                return_overflowing_tokens=False,
            )
            
            # Create labels (for causal LM, labels = input_ids)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Apply tokenization
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train dataset"
        )
        
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Tokenizing eval dataset"
            )
        
        # Filter by length
        def filter_by_length(example):
            length = len(example['input_ids'])
            return self.data_config.min_length <= length <= self.data_config.max_length
        
        train_dataset = train_dataset.filter(filter_by_length)
        if eval_dataset:
            eval_dataset = eval_dataset.filter(filter_by_length)
        
        # Add length column for grouping
        if self.training_config.group_by_length:
            train_dataset = train_dataset.map(
                lambda x: {"length": len(x["input_ids"])},
                desc="Adding length column"
            )
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Eval dataset size: {len(eval_dataset)}")
        
        return {
            "train": train_dataset,
            "eval": eval_dataset
        }
    
    def create_trainer(self, model: nn.Module, datasets: Dict[str, Dataset]) -> Trainer:
        """Create optimized trainer with advanced features."""
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            run_name=self.training_config.run_name or self.experiment_name,
            
            # Training parameters
            num_train_epochs=self.training_config.num_train_epochs,
            max_steps=self.training_config.max_steps,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            
            # Optimization
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            adam_beta1=self.training_config.adam_beta1,
            adam_beta2=self.training_config.adam_beta2,
            adam_epsilon=self.training_config.adam_epsilon,
            max_grad_norm=self.training_config.max_grad_norm,
            optim=self.training_config.optim,
            
            # Learning rate scheduling
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            warmup_ratio=self.training_config.warmup_ratio,
            warmup_steps=self.training_config.warmup_steps,
            
            # Evaluation and saving
            evaluation_strategy=self.training_config.evaluation_strategy,
            eval_steps=self.training_config.eval_steps,
            save_strategy=self.training_config.save_strategy,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            
            # Performance optimizations
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            dataloader_pin_memory=self.training_config.dataloader_pin_memory,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            tf32=self.training_config.tf32,
            
            # Logging
            logging_steps=self.training_config.logging_steps,
            report_to=self.training_config.report_to,
            
            # Advanced features
            group_by_length=self.training_config.group_by_length,
            length_column_name=self.training_config.length_column_name,
            remove_unused_columns=self.training_config.remove_unused_columns,
        )
        
        # Add NEFTune if specified
        if self.training_config.neftune_noise_alpha:
            training_args.neftune_noise_alpha = self.training_config.neftune_noise_alpha
        
        # Create data collator
        data_collator = AdvancedDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.data_config.max_seq_length
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["eval"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        return trainer
    
    def train(self) -> Dict[str, Any]:
        """Execute the complete training pipeline."""
        try:
            logger.info("Starting Advanced QLoRA Training Pipeline")
            
            # Load model and tokenizer
            model, tokenizer = self.load_model_and_tokenizer()
            
            # Setup LoRA
            model = self.setup_lora(model)
            
            # Prepare datasets
            datasets = self.prepare_dataset()
            
            # Create trainer
            trainer = self.create_trainer(model, datasets)
            self.trainer = trainer
            
            # Check for existing checkpoints
            last_checkpoint = None
            if os.path.isdir(self.training_config.output_dir):
                last_checkpoint = get_last_checkpoint(self.training_config.output_dir)
                if last_checkpoint:
                    logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            
            # Start training
            logger.info("Beginning training...")
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
            
            # Save final model
            trainer.save_model()
            trainer.save_state()
            
            # Log final metrics
            metrics = train_result.metrics
            logger.info("Training completed successfully!")
            logger.info(f"Final metrics: {metrics}")
            
            # Save training summary
            summary = {
                "experiment_name": self.experiment_name,
                "model_name": self.model_config.model_name_or_path,
                "final_metrics": metrics,
                "config": {
                    "model": self.model_config.__dict__,
                    "lora": self.lora_config.__dict__,
                    "data": self.data_config.__dict__,
                    "training": self.training_config.__dict__
                }
            }
            
            summary_path = os.path.join(self.training_config.output_dir, "training_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            return summary
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def evaluate(self) -> Dict[str, float]:
        """Run comprehensive evaluation."""
        if not self.trainer:
            raise ValueError("Model must be trained first")
        
        logger.info("Running evaluation...")
        eval_results = self.trainer.evaluate()
        
        logger.info(f"Evaluation results: {eval_results}")
        return eval_results

def main():
    """Main training script with CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced QLoRA Fine-Tuning System")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B",
                       help="Model name or path")
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Trust remote code")
    
    # Data arguments  
    parser.add_argument("--dataset_path", type=str, help="Path to dataset file")
    parser.add_argument("--dataset_name", type=str, help="HuggingFace dataset name")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # System arguments
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--experiment_name", type=str,
                       help="Experiment name for tracking")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create configurations
    model_config = ModelConfig(
        model_name_or_path=args.model_name,
        trust_remote_code=args.trust_remote_code
    )
    
    lora_config = LoRAConfig(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    )
    
    data_config = DataConfig(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        max_seq_length=args.max_seq_length
    )
    
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Initialize trainer
    trainer = AdvancedQLoRATrainer(
        model_config=model_config,
        lora_config=lora_config,
        data_config=data_config,
        training_config=training_config,
        experiment_name=args.experiment_name
    )
    
    # Run training
    results = trainer.train()
    
    logger.info("Training pipeline completed successfully!")
    return results

if __name__ == "__main__":
    main()