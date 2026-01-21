import os
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl,
    EarlyStoppingCallback, get_scheduler
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import PeftModel
import psutil
import GPUtil

logger = logging.getLogger(__name__)

class AdvancedTrainingCallback(TrainerCallback):
    """Enhanced callback with comprehensive monitoring and optimization."""
    
    def __init__(
        self,
        log_system_stats: bool = True,
        memory_cleanup_steps: int = 100,
        gradient_clip_logging: bool = True
    ):
        self.log_system_stats = log_system_stats
        self.memory_cleanup_steps = memory_cleanup_steps
        self.gradient_clip_logging = gradient_clip_logging
        self.step_times = []
        self.memory_usage = []
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each training step."""
        self.step_start_time = time.time()
        
        # Log system stats periodically
        if self.log_system_stats and state.global_step % args.logging_steps == 0:
            self._log_system_stats(state.global_step)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        # Record step time
        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)
        
        # Memory cleanup
        if state.global_step % self.memory_cleanup_steps == 0:
            self._cleanup_memory()
        
        # Log gradient norms if enabled
        if self.gradient_clip_logging and hasattr(kwargs.get('model'), 'named_parameters'):
            self._log_gradient_norms(kwargs['model'], state.global_step)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Enhanced logging with additional metrics."""
        if logs and self.step_times:
            # Add timing metrics
            recent_times = self.step_times[-args.logging_steps:]
            logs['step_time_avg'] = np.mean(recent_times)
            logs['step_time_std'] = np.std(recent_times)
            logs['steps_per_second'] = 1.0 / np.mean(recent_times)
            
            # Add memory metrics
            if torch.cuda.is_available():
                logs['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
                logs['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
    
    def _log_system_stats(self, step: int):
        """Log comprehensive system statistics."""
        try:
            # CPU and RAM usage
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            logger.info(f"Step {step} - CPU: {cpu_percent:.1f}%, RAM: {memory.percent:.1f}%")
            
            # GPU usage
            if torch.cuda.is_available():
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    logger.info(f"GPU {i}: {gpu.memoryUtil*100:.1f}% mem{gpu.load*100:.1f}% utilization")
        
        except Exception as e:
            logger.warning(f"Could not log system stats: {e}")
    
    def _cleanup_memory(self):
        """Perform memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _log_gradient_norms(self, model: nn.Module, step: int):
        """Log gradient norms for monitoring training stability."""
        try:
            total_norm = 0.0
            param_count = 0
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                logger.debug(f"Step {step} - Gradient norm: {total_norm:.4f}")
        
        except Exception as e:
            logger.warning(f"Could not log gradient norms: {e}")

class SavePeftModelCallback(TrainerCallback):
    """Enhanced PEFT model saving with versioning and metadata."""
    
    def __init__(self, save_full_model: bool = False):
        self.save_full_model = save_full_model
    
    def on_save(self, args, state, control, **kwargs):
        """Save PEFT model with enhanced metadata."""
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        
        # Save PEFT adapter
        kwargs["model"].save_pretrained(peft_model_path)
        
        # Save training metadata
        metadata = {
            'global_step': state.global_step,
            'epoch': state.epoch,
            'learning_rate': state.log_history[-1].get('learning_rate', 0) if state.log_history else 0,
            'train_loss': state.log_history[-1].get('train_loss', 0) if state.log_history else 0,
            'timestamp': time.time(),
            'model_type': type(kwargs["model"]).__name__
        }
        
        metadata_path = os.path.join(checkpoint_folder, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved PEFT checkpoint to {peft_model_path}")
        
        # Optionally save full model
        if self.save_full_model:
            full_model_path = os.path.join(checkpoint_folder, "full_model")
            kwargs["model"].save_pretrained(full_model_path)
            logger.info(f"Saved full model to {full_model_path}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Save final model with completion marker."""
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        
        # Create completion marker
        touch(os.path.join(args.output_dir, 'completed'))
        
        # Save final model
        self.on_save(args, state, control, **kwargs)

class DynamicBatchSizeCallback(TrainerCallback):
    """Dynamically adjust batch size based on memory usage."""
    
    def __init__(
        self,
        target_memory_utilization: float = 0.85,
        adjustment_frequency: int = 100,
        min_batch_size: int = 1,
        max_batch_size: int = 32
    ):
        self.target_memory_utilization = target_memory_utilization
        self.adjustment_frequency = adjustment_frequency
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize batch size tracking."""
        self.current_batch_size = args.per_device_train_batch_size
    
    def on_step_end(self, args, state, control, **kwargs):
        """Monitor memory and adjust batch size if needed."""
        if state.global_step % self.adjustment_frequency == 0:
            self._adjust_batch_size(args, state)
    
    def _adjust_batch_size(self, args, state):
        """Adjust batch size based on memory utilization."""
        if not torch.cuda.is_available():
            return
        
        try:
            memory_utilization = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory
            
            if memory_utilization > self.target_memory_utilization and self.current_batch_size > self.min_batch_size:
                # Reduce batch size
                new_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
                logger.info(f"Reducing batch size from {self.current_batch_size} to {new_batch_size} (memory: {memory_utilization:.2%})")
                self.current_batch_size = new_batch_size
                args.per_device_train_batch_size = new_batch_size
            
            elif memory_utilization < self.target_memory_utilization * 0.7 and self.current_batch_size < self.max_batch_size:
                # Increase batch size
                new_batch_size = min(self.max_batch_size, self.current_batch_size * 2)
                logger.info(f"Increasing batch size from {self.current_batch_size} to {new_batch_size} (memory: {memory_utilization:.2%})")
                self.current_batch_size = new_batch_size
                args.per_device_train_batch_size = new_batch_size
        
        except Exception as e:
            logger.warning(f"Could not adjust batch size: {e}")

class LossSpikesCallback(TrainerCallback):
    """Monitor and handle loss spikes during training."""
    
    def __init__(
        self,
        spike_threshold: float = 2.0,
        patience: int = 3,
        recovery_lr_factor: float = 0.5
    ):
        self.spike_threshold = spike_threshold
        self.patience = patience
        self.recovery_lr_factor = recovery_lr_factor
        self.loss_history = []
        self.spike_count = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Monitor training loss for spikes."""
        if logs and 'train_loss' in logs:
            current_loss = logs['train_loss']
            self.loss_history.append(current_loss)
            
            # Keep only recent history
            if len(self.loss_history) > 20:
                self.loss_history = self.loss_history[-20:]
            
            # Check for loss spike
            if len(self.loss_history) >= 5:
                recent_avg = np.mean(self.loss_history[-5:-1])
                if current_loss > recent_avg * self.spike_threshold:
                    self.spike_count += 1
                    logger.warning(f"Loss spike detected: {current_loss:.4f} vs recent avg {recent_avg:.4f}")
                    
                    if self.spike_count >= self.patience:
                        logger.warning("Multiple loss spikes detected, reducing learning rate")
                        self._reduce_learning_rate(kwargs.get('optimizer'))
                        self.spike_count = 0
    
    def _reduce_learning_rate(self, optimizer):
        """Reduce learning rate to recover from loss spikes."""
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] *= self.recovery_lr_factor
                logger.info(f"Reduced learning rate from {old_lr:.2e} to {param_group['lr']:.2e}")

class AdvancedTrainer(Trainer):
    """Enhanced trainer with additional optimizations and features."""
    
    def __init__(self, *args, **kwargs):
        # Extract custom parameters
        self.use_gradient_accumulation_optimization = kwargs.pop('use_gradient_accumulation_optimization', True)
        self.use_mixed_precision_optimization = kwargs.pop('use_mixed_precision_optimization', True)
        self.use_compile = kwargs.pop('use_compile', False)
        
        super().__init__(*args, **kwargs)
        
        # Apply optimizations
        if self.use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Enhanced loss computation with stability improvements."""
        # Standard loss computation
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        outputs = model(**inputs)
        
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        
        if labels is not None:
            if self.label_smoother is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        else:
            if isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                loss = outputs[0]
        
        # Loss stability check
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN or Inf loss detected, skipping this batch")
            loss = torch.tensor(0.0, requires_grad=True, device=loss.device)
        
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """Enhanced training step with additional optimizations."""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Use automatic mixed precision if enabled
        if self.use_amp:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # We keep track of the loss at each epoch
            loss = loss / self.args.gradient_accumulation_steps
        
        # Backward pass with gradient accumulation optimization
        if self.use_gradient_accumulation_optimization:
            self._optimized_backward(loss)
        else:
            if self.use_amp:
                self.scaler.scale(loss).backward()
            elif self.deepspeed:
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()
        
        return loss.detach()
    
    def _optimized_backward(self, loss):
        """Optimized backward pass for better memory efficiency."""
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Clear intermediate activations to save memory
        if hasattr(self.model, 'gradient_checkpointing') and self.model.gradient_checkpointing:
            torch.cuda.empty_cache()
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """Enhanced prediction step with better error handling."""
        try:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("OOM during prediction, clearing cache and retrying with smaller batch")
                torch.cuda.empty_cache()
                # Try with reduced batch size
                if len(inputs['input_ids']) > 1:
                    # Split batch in half
                    mid = len(inputs['input_ids']) // 2
                    inputs1 = {k: v[:mid] for k, v in inputs.items()}
                    inputs2 = {k: v[mid:] for k, v in inputs.items()}
                    
                    result1 = super().prediction_step(model, inputs1, prediction_loss_only, ignore_keys)
                    result2 = super().prediction_step(model, inputs2, prediction_loss_only, ignore_keys)
                    
                    # Combine results
                    if prediction_loss_only:
                        return ((result1[0] + result2[0]) / 2, None, None)
                    else:
                        combined_predictions = torch.cat([result1[1], result2[1]], dim=0)
                        combined_labels = torch.cat([result1[2], result2[2]], dim=0) if result1[2] is not None else None
                        return ((result1[0] + result2[0]) / 2, combined_predictions, combined_labels)
                else:
                    # Single sample, skip it
                    logger.warning("Skipping single sample due to OOM")
                    return (torch.tensor(0.0), None, None)
            else:
                raise e

class OptimizedTrainingArguments(TrainingArguments):
    """Enhanced training arguments with additional optimizations."""
    
    def __init__(self, *args, **kwargs):
        # Custom optimization flags
        self.use_gradient_accumulation_optimization = kwargs.pop('use_gradient_accumulation_optimization', True)
        self.use_mixed_precision_optimization = kwargs.pop('use_mixed_precision_optimization', True)
        self.use_compile = kwargs.pop('use_compile', False)
        self.dynamic_batch_size = kwargs.pop('dynamic_batch_size', False)
        self.loss_spike_detection = kwargs.pop('loss_spike_detection', True)
        
        super().__init__(*args, **kwargs)

def create_optimized_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    training_args=None,
    data_collator=None,
    compute_metrics=None,
    callbacks=None
) -> AdvancedTrainer:
    """Create an optimized trainer with all enhancements."""
    
    # Default callbacks
    default_callbacks = [
        AdvancedTrainingCallback(
            log_system_stats=True,
            memory_cleanup_steps=100,
            gradient_clip_logging=True
        ),
        SavePeftModelCallback(save_full_model=False)
    ]
    
    # Add optional callbacks based on training args
    if hasattr(training_args, 'dynamic_batch_size') and training_args.dynamic_batch_size:
        default_callbacks.append(DynamicBatchSizeCallback())
    
    if hasattr(training_args, 'loss_spike_detection') and training_args.loss_spike_detection:
        default_callbacks.append(LossSpikesCallback())
    
    # Add early stopping if eval dataset is provided
    if eval_dataset is not None:
        default_callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            )
        )
    
    # Combine with user callbacks
    all_callbacks = default_callbacks + (callbacks or [])
    
    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=all_callbacks,
        use_gradient_accumulation_optimization=getattr(training_args, 'use_gradient_accumulation_optimization', True),
        use_mixed_precision_optimization=getattr(training_args, 'use_mixed_precision_optimization', True),
        use_compile=getattr(training_args, 'use_compile', False)
    )
    
    return trainer

class TrainingMonitor:
    """Comprehensive training monitoring and analysis."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_history = []
        self.system_stats = []
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log training metrics with timestamp."""
        entry = {
            'step': step,
            'timestamp': time.time(),
            **metrics
        }
        self.metrics_history.append(entry)
        
        # Save to file periodically
        if step % 100 == 0:
            self.save_metrics()
    
    def log_system_stats(self, step: int):
        """Log system performance statistics."""
        stats = {
            'step': step,
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
        }
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'gpu_utilization': GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 0
            })
        
        self.system_stats.append(stats)
    
    def save_metrics(self):
        """Save metrics to files."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save training metrics
        metrics_file = os.path.join(self.output_dir, 'training_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Save system stats
        stats_file = os.path.join(self.output_dir, 'system_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(self.system_stats, f, indent=2)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        if not self.metrics_history:
            return {}
        
        # Calculate training statistics
        final_metrics = self.metrics_history[-1]
        
        # Loss progression
        train_losses = [m.get('train_loss', 0) for m in self.metrics_history if 'train_loss' in m]
        eval_losses = [m.get('eval_loss', 0) for m in self.metrics_history if 'eval_loss' in m]
        
        report = {
            'training_summary': {
                'total_steps': final_metrics.get('step', 0),
                'final_train_loss': train_losses[-1] if train_losses else None,
                'final_eval_loss': eval_losses[-1] if eval_losses else None,
                'best_eval_loss': min(eval_losses) if eval_losses else None,
                'training_time_hours': (final_metrics['timestamp'] - self.metrics_history[0]['timestamp']) / 3600
            },
            'loss_progression': {
                'train_loss_reduction': (train_losses[0] - train_losses[-1]) / train_losses[0] if len(train_losses) > 1 else 0,
                'eval_loss_reduction': (eval_losses[0] - eval_losses[-1]) / eval_losses[0] if len(eval_losses) > 1 else 0
            },
            'system_performance': self._analyze_system_performance()
        }
        
        return report
    
    def _analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze system performance during training."""
        if not self.system_stats:
            return {}
        
        cpu_usage = [s['cpu_percent'] for s in self.system_stats]
        memory_usage = [s['memory_percent'] for s in self.system_stats]
        
        analysis = {
            'avg_cpu_usage': np.mean(cpu_usage),
            'max_cpu_usage': np.max(cpu_usage),
            'avg_memory_usage': np.mean(memory_usage),
            'max_memory_usage': np.max(memory_usage)
        }
        
        if any('gpu_memory_allocated_gb' in s for s in self.system_stats):
            gpu_memory = [s.get('gpu_memory_allocated_gb', 0) for s in self.system_stats]
            analysis.update({
                'avg_gpu_memory_gb': np.mean(gpu_memory),
                'max_gpu_memory_gb': np.max(gpu_memory)
            })
        
        return analysis