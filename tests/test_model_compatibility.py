#!/usr/bin/env python3
"""
Comprehensive tests for model compatibility and issue resolution.
Tests all supported model architectures and common problems.
"""

import unittest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_utils import (
    ModelCompatibilityManager, QuantizationOptimizer, 
    MemoryProfiler, find_all_linear_names, ModelValidator
)

class TestModelCompatibilityManager(unittest.TestCase):
    """Test model compatibility fixes and detection."""
    
    def test_model_family_detection(self):
        """Test model family detection from names."""
        test_cases = [
            ("meta-llama/Llama-3.1-8B", "llama"),
            ("Qwen/Qwen2.5-7B", "qwen"),
            ("mistralai/Mistral-7B-v0.1", "mistral"),
            ("google/gemma-2-9b", "gemma"),
            ("microsoft/phi-3-mini", "phi"),
            ("unknown-model/test", "llama")  # fallback
        ]
        
        for model_name, expected_family in test_cases:
            with self.subTest(model_name=model_name):
                detected = ModelCompatibilityManager.detect_model_family(model_name)
                self.assertEqual(detected, expected_family)
    
    def test_model_fixes_configuration(self):
        """Test that model fixes are properly configured."""
        for family, fixes in ModelCompatibilityManager.MODEL_FIXES.items():
            with self.subTest(family=family):
                self.assertIsInstance(fixes, dict)
                self.assertIn("config_updates", fixes)
                self.assertIsInstance(fixes["config_updates"], dict)
    
    def test_apply_model_fixes(self):
        """Test applying model fixes to a mock model."""
        # Create mock model
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.use_cache = True
        
        # Apply fixes
        fixed_model = ModelCompatibilityManager.apply_model_fixes(
            mock_model, "meta-llama/Llama-3.1-8B"
        )
        
        # Check that use_cache was set to False
        self.assertEqual(fixed_model.config.use_cache, False)

class TestQuantizationOptimizer(unittest.TestCase):
    """Test quantization optimization utilities."""
    
    def test_bnb_config_optimization(self):
        """Test BitsAndBytes configuration optimization."""
        # Test with different memory scenarios
        test_cases = [
            (4.0, "7b"),    # Low memory
            (16.0, "13b"),  # Medium memory
            (32.0, "30b"),  # High memory
        ]
        
        for available_memory, model_size in test_cases:
            with self.subTest(memory=available_memory, size=model_size):
                config = QuantizationOptimizer.optimize_bnb_config(
                    f"test-model-{model_size}", available_memory
                )
                
                self.assertIsInstance(config, dict)
                # Should have either 4-bit or 8-bit quantization
                self.assertTrue(
                    config.get("load_in_4bit", False) or 
                    config.get("load_in_8bit", False)
                )

class TestMemoryProfiler(unittest.TestCase):
    """Test memory profiling utilities."""
    
    def test_gpu_memory_info(self):
        """Test GPU memory information retrieval."""
        memory_info = MemoryProfiler.get_gpu_memory_info()
        
        self.assertIsInstance(memory_info, dict)
        self.assertIn("total_gb", memory_info)
        self.assertIn("available_gb", memory_info)
        
        if torch.cuda.is_available():
            self.assertGreater(memory_info["total_gb"], 0)
        else:
            self.assertEqual(memory_info["total_gb"], 0)
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    def test_optimize_batch_size(self, mock_device_count, mock_cuda_available):
        """Test batch size optimization."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        batch_size = MemoryProfiler.optimize_batch_size(
            mock_model, mock_tokenizer, max_seq_length=2048
        )
        
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)
        self.assertLessEqual(batch_size, 8)

class TestLinearLayerDetection(unittest.TestCase):
    """Test linear layer detection for LoRA targeting."""
    
    def test_find_linear_names_basic(self):
        """Test finding linear layer names in a simple model."""
        # Create a simple model with linear layers
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 20)
                self.layer2 = nn.Linear(20, 30)
                self.lm_head = nn.Linear(30, 1000)
        
        model = SimpleModel()
        linear_names = find_all_linear_names(model, exclude_lm_head=True)
        
        self.assertIsInstance(linear_names, list)
        self.assertIn("layer1", linear_names)
        self.assertIn("layer2", linear_names)
        self.assertNotIn("lm_head", linear_names)
    
    def test_find_linear_names_with_lm_head(self):
        """Test finding linear layer names including lm_head."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 20)
                self.lm_head = nn.Linear(20, 1000)
        
        model = SimpleModel()
        linear_names = find_all_linear_names(model, exclude_lm_head=False)
        
        self.assertIn("layer1", linear_names)
        self.assertIn("lm_head", linear_names)

class TestModelValidator(unittest.TestCase):
    """Test model validation utilities."""
    
    def test_validate_model_config_basic(self):
        """Test basic model configuration validation."""
        # Create mock model and tokenizer
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.vocab_size = 1000
        mock_model.config.use_cache = True
        
        mock_tokenizer = Mock()
        mock_tokenizer.__len__ = Mock(return_value=1000)
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        
        issues = ModelValidator.validate_model_config(mock_model, mock_tokenizer)
        
        self.assertIsInstance(issues, list)
        # Should detect missing pad_token and use_cache=True
        self.assertTrue(any("pad_token" in issue for issue in issues))
        self.assertTrue(any("use_cache" in issue for issue in issues))
    
    def test_fix_common_issues(self):
        """Test automatic fixing of common issues."""
        # Create mock model and tokenizer with issues
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.use_cache = True
        mock_model.resize_token_embeddings = Mock()
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.__len__ = Mock(return_value=1000)
        
        issues = ["missing pad_token", "use_cache should be False"]
        
        fixed_model, fixed_tokenizer, fixed_issues = ModelValidator.fix_common_issues(
            mock_model, mock_tokenizer, issues
        )
        
        self.assertIsInstance(fixed_issues, list)
        self.assertGreater(len(fixed_issues), 0)
        self.assertEqual(fixed_model.config.use_cache, False)
        self.assertEqual(fixed_tokenizer.pad_token, "</s>")

class TestModelArchitectureSupport(unittest.TestCase):
    """Test support for different model architectures."""
    
    def test_supported_architectures(self):
        """Test that all claimed architectures are properly configured."""
        from utils.model_utils import ModelCompatibilityManager
        
        supported_families = [
            "llama", "qwen", "mistral", "gemma", "phi"
        ]
        
        for family in supported_families:
            with self.subTest(family=family):
                self.assertIn(family, ModelCompatibilityManager.MODEL_FIXES)
                
                fixes = ModelCompatibilityManager.MODEL_FIXES[family]
                self.assertIn("config_updates", fixes)
                self.assertIsInstance(fixes["config_updates"], dict)

class TestErrorHandling(unittest.TestCase):
    """Test error handling and recovery mechanisms."""
    
    def test_memory_profiler_no_gpu(self):
        """Test memory profiler behavior when no GPU is available."""
        with patch('torch.cuda.is_available', return_value=False):
            memory_info = MemoryProfiler.get_gpu_memory_info()
            
            self.assertEqual(memory_info["total_gb"], 0)
            self.assertEqual(memory_info["available_gb"], 0)
    
    def test_model_fixes_with_missing_attributes(self):
        """Test model fixes when model is missing expected attributes."""
        # Create mock model without some expected attributes
        mock_model = Mock()
        mock_model.config = Mock()
        
        # Remove some attributes that fixes might expect
        del mock_model.config.use_cache
        
        # Should not raise an error
        try:
            ModelCompatibilityManager.apply_model_fixes(
                mock_model, "meta-llama/Llama-3.1-8B"
            )
        except AttributeError:
            self.fail("apply_model_fixes raised AttributeError unexpectedly")

def run_compatibility_tests():
    """Run all compatibility tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestModelCompatibilityManager,
        TestQuantizationOptimizer,
        TestMemoryProfiler,
        TestLinearLayerDetection,
        TestModelValidator,
        TestModelArchitectureSupport,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_compatibility_tests()
    sys.exit(0 if success else 1)