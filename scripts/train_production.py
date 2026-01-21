#!/usr/bin/env python3
"""
Production training script for Advanced QLoRA System.
Optimized for enterprise deployment with comprehensive error handling.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch
from transformers import set_seed
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from advanced_qlora_trainer import (
    ModelConfig, LoRAConfig, DataConfig, TrainingConfig,
    AdvancedQLoRATrainer, setup_logging
)
from utils.model_utils import ModelCompatibilityManager, MemoryProfiler
from utils.data_utils import DatasetLoader, DatasetProcessor
from utils.training_utils import TrainingMonitor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_environment() -> Dict[str, Any]:
    """Validate training environment and return system info."""
    info = {
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info['gpu_memory'] = MemoryProfiler.get_gpu_memory_info()
        
        # Check for common issues
        if info['gpu_memory']['available_gb'] < 4:
            logger.warning("Low GPU memory available, consider using more aggressive quantization")
    
    logger.info(f"Environment validation: {json.dumps(info, indent=2)}")
    return info

def create_configs_from_dict(config_dict: Dict[str, Any]) -> tuple:
    """Create configuration objects from dictionary."""
    
    # Model configuration
    model_config = ModelConfig(**config_dict.get('model', {}))
    
    # LoRA configuration
    lora_config = LoRAConfig(**config_dict.get('lora', {}))
    
    # Data configuration
    data_config = DataConfig(**config_dict.get('data', {}))
    
    # Training configuration
    training_config = TrainingConfig(**config_dict.get('training', {}))
    
    return model_config, lora_config, data_config, training_config

def setup_experiment_directory(output_dir: str, experiment_name: str) -> str:
    """Setup experiment directory with proper structure."""
    exp_dir = Path(output_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "metrics").mkdir(exist_ok=True)
    (exp_dir / "configs").mkdir(exist_ok=True)
    
    return str(exp_dir)

def main():
    """Main training function with comprehensive error handling."""
    parser = argparse.ArgumentParser(description="Advanced QLoRA Production Training")
    
    # Configuration
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--experiment_name", type=str,
                       help="Experiment name (overrides config)")
    parser.add_argument("--output_dir", type=str,
                       help="Output directory (overrides config)")
    
    # Model overrides
    parser.add_argument("--model_name", type=str,
                       help="Model name (overrides config)")
    parser.add_argument("--dataset_path", type=str,
                       help="Dataset path (overrides config)")
    
    # Training overrides
    parser.add_argument("--learning_rate", type=float,
                       help="Learning rate (overrides config)")
    parser.add_argument("--num_epochs", type=int,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--batch_size", type=int,
                       help="Batch size (overrides config)")
    
    # System settings
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str,
                       help="Resume training from checkpoint")
    
    # Validation and testing
    parser.add_argument("--validate_only", action="store_true",
                       help="Only validate configuration without training")
    parser.add_argument("--dry_run", action="store_true",
                       help="Perform dry run without actual training")
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        setup_logging(args.log_level)
        logger.info("Starting Advanced QLoRA Production Training")
        
        # Set random seed
        set_seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")
        
        # Validate environment
        env_info = validate_environment()
        
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config_dict = load_config(args.config)
        
        # Apply command line overrides
        if args.model_name:
            config_dict.setdefault('model', {})['model_name_or_path'] = args.model_name
        if args.dataset_path:
            config_dict.setdefault('data', {})['dataset_path'] = args.dataset_path
        if args.learning_rate:
            config_dict.setdefault('training', {})['learning_rate'] = args.learning_rate
        if args.num_epochs:
            config_dict.setdefault('training', {})['num_train_epochs'] = args.num_epochs
        if args.batch_size:
            config_dict.setdefault('training', {})['per_device_train_batch_size'] = args.batch_size
        
        # Create configuration objects
        model_config, lora_config, data_config, training_config = create_configs_from_dict(config_dict)
        
        # Setup experiment directory
        experiment_name = args.experiment_name or training_config.run_name or "qlora_experiment"
        output_dir = args.output_dir or training_config.output_dir
        exp_dir = setup_experiment_directory(output_dir, experiment_name)
        training_config.output_dir = exp_dir
        
        # Save configuration
        config_save_path = Path(exp_dir) / "configs" / "training_config.yaml"
        with open(config_save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logger.info(f"Saved configuration to {config_save_path}")
        
        # Validation mode
        if args.validate_only:
            logger.info("Configuration validation completed successfully")
            return
        
        # Initialize training monitor
        monitor = TrainingMonitor(exp_dir)
        
        # Create trainer
        logger.info("Initializing Advanced QLoRA Trainer")
        trainer = AdvancedQLoRATrainer(
            model_config=model_config,
            lora_config=lora_config,
            data_config=data_config,
            training_config=training_config,
            experiment_name=experiment_name
        )
        
        # Dry run mode
        if args.dry_run:
            logger.info("Dry run mode - loading model and data without training")
            model, tokenizer = trainer.load_model_and_tokenizer()
            datasets = trainer.prepare_dataset()
            logger.info(f"Dry run completed successfully")
            logger.info(f"Model: {type(model).__name__}")
            logger.info(f"Tokenizer: {type(tokenizer).__name__}")
            logger.info(f"Train dataset size: {len(datasets['train'])}")
            if datasets['eval']:
                logger.info(f"Eval dataset size: {len(datasets['eval'])}")
            return
        
        # Run training
        logger.info("Starting training pipeline")
        results = trainer.train()
        
        # Generate final report
        report = monitor.generate_report()
        report_path = Path(exp_dir) / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results: {json.dumps(results['final_metrics'], indent=2)}")
        logger.info(f"Full report saved to: {report_path}")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error("Full traceback:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()