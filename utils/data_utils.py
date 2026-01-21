"""
Advanced data processing utilities for QLoRA fine-tuning.
Handles multiple dataset formats and implements efficient preprocessing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer
import torch
from torch.utils.data import DataLoader
import numpy as np

logger = logging.getLogger(__name__)

class DatasetProcessor:
    """Advanced dataset processing with support for multiple formats."""
    
    SUPPORTED_FORMATS = {
        'alpaca': {
            'required_fields': ['instruction', 'input', 'output'],
            'template': "Below is an instruction that describes a task{input_part}. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}{input_section}\n\n### Response:\n{output}"
        },
        'sharegpt': {
            'required_fields': ['conversations'],
            'template': None  # Custom processing
        },
        'oasst': {
            'required_fields': ['text'],
            'template': "{text}"
        },
        'dolly': {
            'required_fields': ['instruction', 'context', 'response'],
            'template': "### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
        },
        'self_instruct': {
            'required_fields': ['prompt', 'completion'],
            'template': "{prompt}\n\n{completion}"
        },
        'custom': {
            'required_fields': ['input', 'output'],
            'template': "{input}\n\n{output}"
        }
    }
    
    @classmethod
    def detect_format(cls, dataset: Dataset) -> str:
        """Auto-detect dataset format based on column names."""
        columns = set(dataset.column_names)
        
        for format_name, format_info in cls.SUPPORTED_FORMATS.items():
            required_fields = set(format_info['required_fields'])
            if required_fields.issubset(columns):
                logger.info(f"Detected dataset format: {format_name}")
                return format_name
        
        logger.warning("Could not detect dataset format, using custom format")
        return 'custom'
    
    @classmethod
    def process_alpaca_format(cls, example: Dict[str, Any]) -> Dict[str, str]:
        """Process Alpaca format data."""
        instruction = example['instruction']
        input_text = example.get('input', '').strip()
        output = example['output']
        
        if input_text:
            input_part = ", paired with an input that provides further context"
            input_section = f"\n\n### Input:\n{input_text}"
        else:
            input_part = ""
            input_section = ""
        
        formatted_text = cls.SUPPORTED_FORMATS['alpaca']['template'].format(
            input_part=input_part,
            instruction=instruction,
            input_section=input_section,
            output=output
        )
        
        return {
            'text': formatted_text,
            'input': f"Below is an instruction that describes a task{input_part}. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}{input_section}\n\n### Response:",
            'output': output
        }
    
    @classmethod
    def process_sharegpt_format(cls, example: Dict[str, Any]) -> Dict[str, str]:
        """Process ShareGPT format data."""
        conversations = example['conversations']
        
        formatted_parts = []
        input_parts = []
        output_parts = []
        
        for i, conv in enumerate(conversations):
            role = conv.get('from', conv.get('role', 'unknown'))
            content = conv.get('value', conv.get('content', ''))
            
            if role in ['human', 'user']:
                formatted_parts.append(f"### Human:\n{content}")
                input_parts.append(content)
            elif role in ['gpt', 'assistant']:
                formatted_parts.append(f"### Assistant:\n{content}")
                output_parts.append(content)
        
        full_text = '\n\n'.join(formatted_parts)
        
        # For training, we typically want the last assistant response as output
        if output_parts:
            input_text = '\n\n'.join(formatted_parts[:-1]) + '\n\n### Assistant:'
            output_text = output_parts[-1]
        else:
            input_text = full_text
            output_text = ""
        
        return {
            'text': full_text,
            'input': input_text,
            'output': output_text
        }
    
    @classmethod
    def process_oasst_format(cls, example: Dict[str, Any]) -> Dict[str, str]:
        """Process OpenAssistant format data."""
        text = example['text']
        
        # Try to split into input/output if possible
        if '### Assistant:' in text:
            parts = text.split('### Assistant:')
            if len(parts) == 2:
                input_text = parts[0].strip() + '### Assistant:'
                output_text = parts[1].strip()
            else:
                input_text = text
                output_text = ""
        else:
            input_text = text
            output_text = ""
        
        return {
            'text': text,
            'input': input_text,
            'output': output_text
        }
    
    @classmethod
    def process_dolly_format(cls, example: Dict[str, Any]) -> Dict[str, str]:
        """Process Dolly format data."""
        instruction = example['instruction']
        context = example.get('context', '').strip()
        response = example['response']
        
        if context:
            formatted_text = cls.SUPPORTED_FORMATS['dolly']['template'].format(
                instruction=instruction,
                context=context,
                response=response
            )
            input_text = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:"
        else:
            formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            input_text = f"### Instruction:\n{instruction}\n\n### Response:"
        
        return {
            'text': formatted_text,
            'input': input_text,
            'output': response
        }
    
    @classmethod
    def process_self_instruct_format(cls, example: Dict[str, Any]) -> Dict[str, str]:
        """Process Self-Instruct format data."""
        prompt = example['prompt']
        completion = example['completion']
        
        formatted_text = cls.SUPPORTED_FORMATS['self_instruct']['template'].format(
            prompt=prompt,
            completion=completion
        )
        
        return {
            'text': formatted_text,
            'input': prompt,
            'output': completion
        }
    
    @classmethod
    def process_custom_format(cls, example: Dict[str, Any]) -> Dict[str, str]:
        """Process custom format data."""
        if 'input' in example and 'output' in example:
            input_text = example['input']
            output_text = example['output']
            formatted_text = f"{input_text}\n\n{output_text}"
        elif 'text' in example:
            formatted_text = example['text']
            input_text = formatted_text
            output_text = ""
        else:
            # Fallback: concatenate all fields
            formatted_text = ' '.join(str(v) for v in example.values())
            input_text = formatted_text
            output_text = ""
        
        return {
            'text': formatted_text,
            'input': input_text,
            'output': output_text
        }
    
    @classmethod
    def process_dataset(cls, dataset: Dataset, format_name: Optional[str] = None) -> Dataset:
        """Process dataset according to detected or specified format."""
        if format_name is None:
            format_name = cls.detect_format(dataset)
        
        # Select processing function
        process_func = {
            'alpaca': cls.process_alpaca_format,
            'sharegpt': cls.process_sharegpt_format,
            'oasst': cls.process_oasst_format,
            'dolly': cls.process_dolly_format,
            'self_instruct': cls.process_self_instruct_format,
            'custom': cls.process_custom_format
        }.get(format_name, cls.process_custom_format)
        
        # Apply processing
        processed_dataset = dataset.map(
            process_func,
            desc=f"Processing {format_name} format",
            remove_columns=dataset.column_names
        )
        
        return processed_dataset

class AdvancedTokenizer:
    """Enhanced tokenization with conversation-aware processing."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        truncation_strategy: str = "right",
        padding_strategy: str = "max_length"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.padding_strategy = padding_strategy
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tokenize_for_training(self, examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        """
        Tokenize examples for causal language modeling training.
        Handles input/output separation for proper loss calculation.
        """
        inputs = examples['input']
        outputs = examples['output']
        
        # Tokenize inputs and outputs separately
        tokenized_inputs = self.tokenizer(
            inputs,
            truncation=False,
            padding=False,
            add_special_tokens=True,
            return_attention_mask=False
        )
        
        tokenized_outputs = self.tokenizer(
            outputs,
            truncation=False,
            padding=False,
            add_special_tokens=False,
            return_attention_mask=False
        )
        
        # Combine input and output tokens
        combined_input_ids = []
        combined_labels = []
        
        for input_ids, output_ids in zip(tokenized_inputs['input_ids'], tokenized_outputs['input_ids']):
            # Combine input and output
            full_input_ids = input_ids + output_ids + [self.tokenizer.eos_token_id]
            
            # Create labels (ignore input tokens for loss calculation)
            labels = [-100] * len(input_ids) + output_ids + [self.tokenizer.eos_token_id]
            
            # Truncate if necessary
            if len(full_input_ids) > self.max_length:
                if self.truncation_strategy == "right":
                    full_input_ids = full_input_ids[:self.max_length]
                    labels = labels[:self.max_length]
                elif self.truncation_strategy == "left":
                    # Keep the output, truncate input
                    output_length = len(output_ids) + 1  # +1 for eos_token
                    if output_length < self.max_length:
                        input_length = self.max_length - output_length
                        truncated_input = input_ids[-input_length:] if input_length > 0 else []
                        full_input_ids = truncated_input + output_ids + [self.tokenizer.eos_token_id]
                        labels = [-100] * len(truncated_input) + output_ids + [self.tokenizer.eos_token_id]
                    else:
                        # Output itself is too long, truncate it
                        full_input_ids = output_ids[:self.max_length-1] + [self.tokenizer.eos_token_id]
                        labels = output_ids[:self.max_length-1] + [self.tokenizer.eos_token_id]
            
            combined_input_ids.append(full_input_ids)
            combined_labels.append(labels)
        
        return {
            'input_ids': combined_input_ids,
            'labels': combined_labels
        }
    
    def create_attention_masks(self, input_ids: List[List[int]]) -> List[List[int]]:
        """Create attention masks for input sequences."""
        attention_masks = []
        
        for ids in input_ids:
            # Create attention mask (1 for real tokens, 0 for padding)
            mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in ids]
            attention_masks.append(mask)
        
        return attention_masks

class DataCollatorForSupervisedDataset:
    """
    Enhanced data collator for supervised fine-tuning.
    Handles dynamic padding and proper label masking.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        pad_to_multiple_of: Optional[int] = 8,
        return_tensors: str = "pt"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
    
    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract sequences
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances]
        
        # Determine batch max length
        batch_max_len = min(
            max(len(seq) for seq in input_ids),
            self.max_length
        )
        
        # Pad to multiple if specified (for tensor core optimization)
        if self.pad_to_multiple_of:
            batch_max_len = (
                (batch_max_len + self.pad_to_multiple_of - 1) 
                // self.pad_to_multiple_of 
                * self.pad_to_multiple_of
            )
        
        # Pad sequences
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for seq_input_ids, seq_labels in zip(input_ids, labels):
            # Truncate if necessary
            seq_input_ids = seq_input_ids[:batch_max_len]
            seq_labels = seq_labels[:batch_max_len]
            
            # Calculate padding length
            pad_length = batch_max_len - len(seq_input_ids)
            
            # Pad input_ids
            padded_seq_input_ids = seq_input_ids + [self.tokenizer.pad_token_id] * pad_length
            padded_input_ids.append(padded_seq_input_ids)
            
            # Pad labels (use -100 for padding tokens to ignore in loss)
            padded_seq_labels = seq_labels + [-100] * pad_length
            padded_labels.append(padded_seq_labels)
            
            # Create attention mask
            attention_mask = [1] * len(seq_input_ids) + [0] * pad_length
            attention_masks.append(attention_mask)
        
        # Convert to tensors
        batch = {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
        }
        
        return batch

class DatasetLoader:
    """Unified dataset loader supporting multiple sources and formats."""
    
    @staticmethod
    def load_from_file(file_path: str) -> Dataset:
        """Load dataset from local file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        if file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return Dataset.from_list(data)
            else:
                raise ValueError("JSON file must contain a list of examples")
        
        elif file_path.suffix == '.jsonl':
            return Dataset.from_json(str(file_path))
        
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            return Dataset.from_pandas(df)
        
        elif file_path.suffix == '.parquet':
            return Dataset.from_parquet(str(file_path))
        
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    @staticmethod
    def load_from_hub(
        dataset_name: str,
        config_name: Optional[str] = None,
        split: Optional[str] = None,
        streaming: bool = False
    ) -> Union[Dataset, DatasetDict]:
        """Load dataset from Hugging Face Hub."""
        return load_dataset(
            dataset_name,
            config_name,
            split=split,
            streaming=streaming
        )
    
    @staticmethod
    def create_train_eval_split(
        dataset: Dataset,
        eval_size: Union[float, int] = 0.1,
        seed: int = 42
    ) -> DatasetDict:
        """Create train/eval split from a single dataset."""
        if isinstance(eval_size, float):
            if not 0 < eval_size < 1:
                raise ValueError("eval_size as float must be between 0 and 1")
        elif isinstance(eval_size, int):
            if eval_size >= len(dataset):
                raise ValueError("eval_size as int must be less than dataset size")
            eval_size = eval_size / len(dataset)
        
        split_dataset = dataset.train_test_split(
            test_size=eval_size,
            seed=seed,
            shuffle=True
        )
        
        return DatasetDict({
            'train': split_dataset['train'],
            'eval': split_dataset['test']
        })

class DataQualityChecker:
    """Quality control and filtering for training data."""
    
    @staticmethod
    def check_text_quality(text: str) -> Dict[str, Any]:
        """Analyze text quality metrics."""
        if not isinstance(text, str):
            return {'valid': False, 'reason': 'not_string'}
        
        # Basic checks
        if len(text.strip()) == 0:
            return {'valid': False, 'reason': 'empty'}
        
        if len(text) < 10:
            return {'valid': False, 'reason': 'too_short'}
        
        if len(text) > 50000:  # Very long texts might be problematic
            return {'valid': False, 'reason': 'too_long'}
        
        # Character diversity check
        unique_chars = len(set(text.lower()))
        if unique_chars < 10:
            return {'valid': False, 'reason': 'low_diversity'}
        
        # Repetition check (simple)
        words = text.split()
        if len(words) > 10:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio < 0.3:
                return {'valid': False, 'reason': 'high_repetition'}
        
        return {
            'valid': True,
            'length': len(text),
            'word_count': len(words),
            'unique_chars': unique_chars,
            'repetition_ratio': repetition_ratio if len(words) > 10 else 1.0
        }
    
    @staticmethod
    def filter_dataset(dataset: Dataset, min_quality_score: float = 0.5) -> Dataset:
        """Filter dataset based on quality metrics."""
        def quality_filter(example):
            text = example.get('text', '')
            quality = DataQualityChecker.check_text_quality(text)
            return quality['valid']
        
        filtered_dataset = dataset.filter(quality_filter, desc="Filtering low-quality examples")
        
        logger.info(f"Filtered dataset: {len(dataset)} -> {len(filtered_dataset)} examples")
        return filtered_dataset
    
    @staticmethod
    def remove_duplicates(dataset: Dataset, column: str = 'text') -> Dataset:
        """Remove duplicate examples based on specified column."""
        seen = set()
        
        def is_unique(example):
            text = example[column]
            text_hash = hash(text)
            if text_hash in seen:
                return False
            seen.add(text_hash)
            return True
        
        unique_dataset = dataset.filter(is_unique, desc="Removing duplicates")
        
        logger.info(f"Removed duplicates: {len(dataset)} -> {len(unique_dataset)} examples")
        return unique_dataset