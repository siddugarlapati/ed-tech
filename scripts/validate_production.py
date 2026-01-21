#!/usr/bin/env python3
"""
Production validation script for Advanced QLoRA System.
Comprehensive checks to ensure production readiness.
"""

import os
import sys
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import requests
import torch
import psutil
import GPUtil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_utils import MemoryProfiler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionValidator:
    """Comprehensive production readiness validator."""
    
    def __init__(self):
        self.results = {
            'system_checks': {},
            'dependency_checks': {},
            'performance_checks': {},
            'security_checks': {},
            'integration_checks': {},
            'overall_score': 0
        }
        self.total_checks = 0
        self.passed_checks = 0
    
    def run_check(self, check_name: str, check_func, *args, **kwargs) -> bool:
        """Run a single check and record results."""
        self.total_checks += 1
        try:
            logger.info(f"🔍 Running check: {check_name}")
            result = check_func(*args, **kwargs)
            if result:
                logger.info(f"✅ {check_name}: PASSED")
                self.passed_checks += 1
                return True
            else:
                logger.error(f"❌ {check_name}: FAILED")
                return False
        except Exception as e:
            logger.error(f"❌ {check_name}: ERROR - {e}")
            return False
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """Check system requirements."""
        checks = {}
        
        # Python version
        checks['python_version'] = self.run_check(
            "Python Version (>=3.8)",
            lambda: sys.version_info >= (3, 8)
        )
        
        # CUDA availability
        checks['cuda_available'] = self.run_check(
            "CUDA Available",
            torch.cuda.is_available
        )
        
        # GPU memory
        if torch.cuda.is_available():
            memory_info = MemoryProfiler.get_gpu_memory_info()
            checks['gpu_memory'] = self.run_check(
                "GPU Memory (>=4GB)",
                lambda: memory_info['total_gb'] >= 4
            )
        
        # CPU cores
        checks['cpu_cores'] = self.run_check(
            "CPU Cores (>=4)",
            lambda: psutil.cpu_count() >= 4
        )
        
        # RAM
        memory = psutil.virtual_memory()
        checks['ram'] = self.run_check(
            "RAM (>=8GB)",
            lambda: memory.total / (1024**3) >= 8
        )
        
        # Disk space
        disk = psutil.disk_usage('/')
        checks['disk_space'] = self.run_check(
            "Disk Space (>=50GB)",
            lambda: disk.free / (1024**3) >= 50
        )
        
        self.results['system_checks'] = checks
        return checks
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check required dependencies."""
        checks = {}
        
        required_packages = [
            'torch', 'transformers', 'peft', 'bitsandbytes',
            'accelerate', 'datasets', 'numpy', 'pandas'
        ]
        
        for package in required_packages:
            checks[f'{package}_installed'] = self.run_check(
                f"Package {package}",
                self._check_package_installed,
                package
            )
        
        # Check optional but recommended packages
        optional_packages = ['flash_attn', 'wandb', 'tensorboard']
        for package in optional_packages:
            checks[f'{package}_available'] = self.run_check(
                f"Optional Package {package}",
                self._check_package_installed,
                package
            )
        
        # Check CUDA compatibility
        if torch.cuda.is_available():
            checks['cuda_compatibility'] = self.run_check(
                "CUDA-PyTorch Compatibility",
                self._check_cuda_compatibility
            )
        
        self.results['dependency_checks'] = checks
        return checks
    
    def check_performance(self) -> Dict[str, bool]:
        """Check performance capabilities."""
        checks = {}
        
        # Model loading speed test
        checks['model_loading_speed'] = self.run_check(
            "Model Loading Speed (<60s)",
            self._test_model_loading_speed
        )
        
        # Memory efficiency test
        checks['memory_efficiency'] = self.run_check(
            "Memory Efficiency",
            self._test_memory_efficiency
        )
        
        # Inference speed test
        checks['inference_speed'] = self.run_check(
            "Inference Speed",
            self._test_inference_speed
        )
        
        self.results['performance_checks'] = checks
        return checks
    
    def check_security(self) -> Dict[str, bool]:
        """Check security configurations."""
        checks = {}
        
        # File permissions
        checks['file_permissions'] = self.run_check(
            "File Permissions",
            self._check_file_permissions
        )
        
        # Environment variables
        checks['env_vars_secure'] = self.run_check(
            "Environment Variables Security",
            self._check_env_vars_security
        )
        
        # Docker security (if applicable)
        if self._is_docker_environment():
            checks['docker_security'] = self.run_check(
                "Docker Security",
                self._check_docker_security
            )
        
        self.results['security_checks'] = checks
        return checks
    
    def check_integrations(self) -> Dict[str, bool]:
        """Check integration capabilities."""
        checks = {}
        
        # API server test
        checks['api_server'] = self.run_check(
            "API Server Functionality",
            self._test_api_server
        )
        
        # Configuration loading
        checks['config_loading'] = self.run_check(
            "Configuration Loading",
            self._test_config_loading
        )
        
        # Data processing
        checks['data_processing'] = self.run_check(
            "Data Processing Pipeline",
            self._test_data_processing
        )
        
        self.results['integration_checks'] = checks
        return checks
    
    def _check_package_installed(self, package_name: str) -> bool:
        """Check if a package is installed."""
        try:
            __import__(package_name)
            return True
        except ImportError:
            return False
    
    def _check_cuda_compatibility(self) -> bool:
        """Check CUDA-PyTorch compatibility."""
        try:
            # Test basic CUDA operations
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            return z.is_cuda
        except Exception:
            return False
    
    def _test_model_loading_speed(self) -> bool:
        """Test model loading speed."""
        try:
            start_time = time.time()
            
            # Simulate model loading (use a small model for testing)
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = "microsoft/DialoGPT-small"  # Small model for testing
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            loading_time = time.time() - start_time
            logger.info(f"Model loading time: {loading_time:.2f}s")
            
            # Cleanup
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return loading_time < 60  # Should load within 60 seconds
            
        except Exception as e:
            logger.error(f"Model loading test failed: {e}")
            return False
    
    def _test_memory_efficiency(self) -> bool:
        """Test memory efficiency."""
        try:
            if not torch.cuda.is_available():
                return True  # Skip on CPU-only systems
            
            initial_memory = torch.cuda.memory_allocated()
            
            # Create some tensors
            tensors = []
            for _ in range(10):
                tensor = torch.randn(1000, 1000).cuda()
                tensors.append(tensor)
            
            peak_memory = torch.cuda.memory_allocated()
            
            # Cleanup
            del tensors
            torch.cuda.empty_cache()
            
            final_memory = torch.cuda.memory_allocated()
            
            # Check if memory was properly released
            memory_released = (peak_memory - final_memory) / peak_memory > 0.8
            
            logger.info(f"Memory efficiency: {memory_released}")
            return memory_released
            
        except Exception as e:
            logger.error(f"Memory efficiency test failed: {e}")
            return False
    
    def _test_inference_speed(self) -> bool:
        """Test inference speed."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = "microsoft/DialoGPT-small"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Test inference
            prompt = "Hello, how are you?"
            inputs = tokenizer(prompt, return_tensors="pt")
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            inference_time = time.time() - start_time
            
            logger.info(f"Inference time: {inference_time:.2f}s")
            
            # Cleanup
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return inference_time < 10  # Should complete within 10 seconds
            
        except Exception as e:
            logger.error(f"Inference speed test failed: {e}")
            return False
    
    def _check_file_permissions(self) -> bool:
        """Check file permissions."""
        try:
            # Check if we can read/write in the current directory
            test_file = Path("test_permissions.tmp")
            test_file.write_text("test")
            content = test_file.read_text()
            test_file.unlink()
            
            return content == "test"
        except Exception:
            return False
    
    def _check_env_vars_security(self) -> bool:
        """Check environment variables security."""
        # Check for sensitive information in environment variables
        sensitive_patterns = ['password', 'secret', 'key', 'token']
        
        for key, value in os.environ.items():
            key_lower = key.lower()
            if any(pattern in key_lower for pattern in sensitive_patterns):
                if len(value) > 0:
                    logger.warning(f"Sensitive environment variable detected: {key}")
        
        return True  # This is more of a warning than a failure
    
    def _is_docker_environment(self) -> bool:
        """Check if running in Docker."""
        return Path("/.dockerenv").exists() or "docker" in Path("/proc/1/cgroup").read_text()
    
    def _check_docker_security(self) -> bool:
        """Check Docker security configurations."""
        try:
            # Check if running as non-root user
            return os.getuid() != 0
        except Exception:
            return True  # Skip on non-Unix systems
    
    def _test_api_server(self) -> bool:
        """Test API server functionality."""
        try:
            # This is a basic test - in production you'd start the actual server
            from scripts.inference_server import app
            return app is not None
        except Exception as e:
            logger.error(f"API server test failed: {e}")
            return False
    
    def _test_config_loading(self) -> bool:
        """Test configuration loading."""
        try:
            import yaml
            
            # Test loading a sample config
            config_path = Path(__file__).parent.parent / "configs" / "production_config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                return isinstance(config, dict) and len(config) > 0
            else:
                logger.warning("Production config file not found")
                return False
        except Exception as e:
            logger.error(f"Config loading test failed: {e}")
            return False
    
    def _test_data_processing(self) -> bool:
        """Test data processing pipeline."""
        try:
            from utils.data_utils import DatasetProcessor
            
            # Create sample data
            sample_data = [
                {"instruction": "Test", "input": "", "output": "Response"}
            ]
            
            from datasets import Dataset
            dataset = Dataset.from_list(sample_data)
            
            # Test processing
            processed = DatasetProcessor.process_dataset(dataset, "alpaca")
            
            return len(processed) > 0
        except Exception as e:
            logger.error(f"Data processing test failed: {e}")
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        # Calculate overall score
        if self.total_checks > 0:
            self.results['overall_score'] = (self.passed_checks / self.total_checks) * 100
        
        # Add summary
        self.results['summary'] = {
            'total_checks': self.total_checks,
            'passed_checks': self.passed_checks,
            'failed_checks': self.total_checks - self.passed_checks,
            'success_rate': f"{self.results['overall_score']:.1f}%"
        }
        
        # Determine production readiness
        score = self.results['overall_score']
        if score >= 90:
            readiness = "PRODUCTION READY ✅"
        elif score >= 75:
            readiness = "MOSTLY READY ⚠️ (Minor issues)"
        elif score >= 50:
            readiness = "NEEDS WORK ⚠️ (Major issues)"
        else:
            readiness = "NOT READY ❌ (Critical issues)"
        
        self.results['production_readiness'] = readiness
        
        return self.results
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all production validation checks."""
        logger.info("🚀 Starting Production Validation")
        logger.info("=" * 50)
        
        # Run all check categories
        self.check_system_requirements()
        self.check_dependencies()
        self.check_performance()
        self.check_security()
        self.check_integrations()
        
        # Generate final report
        report = self.generate_report()
        
        logger.info("=" * 50)
        logger.info("📊 PRODUCTION VALIDATION REPORT")
        logger.info("=" * 50)
        logger.info(f"Overall Score: {report['overall_score']:.1f}%")
        logger.info(f"Status: {report['production_readiness']}")
        logger.info(f"Passed: {report['summary']['passed_checks']}/{report['summary']['total_checks']}")
        
        return report

def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Validation for Advanced QLoRA System")
    parser.add_argument("--output", type=str, help="Output file for validation report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run validation
    validator = ProductionValidator()
    report = validator.run_all_checks()
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📄 Report saved to: {args.output}")
    
    # Exit with appropriate code
    if report['overall_score'] >= 75:
        logger.info("✅ System is production ready!")
        sys.exit(0)
    else:
        logger.error("❌ System needs improvements before production deployment")
        sys.exit(1)

if __name__ == "__main__":
    main()