#!/usr/bin/env python3
"""
Quick system validation test for Advanced QLoRA System.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all critical imports."""
    print("🔍 Testing system imports...")
    
    try:
        # Test core dataclasses
        from advanced_qlora_trainer import ModelConfig, LoRAConfig, DataConfig, TrainingConfig
        print("✅ Core configuration classes imported successfully")
        
        # Test trainer class
        from advanced_qlora_trainer import AdvancedQLoRATrainer
        print("✅ Main trainer class imported successfully")
        
        # Test utility modules
        from utils.model_utils import ModelCompatibilityManager
        print("✅ Model utilities imported successfully")
        
        from utils.data_utils import DatasetProcessor
        print("✅ Data utilities imported successfully")
        
        from utils.training_utils import AdvancedTrainer
        print("✅ Training utilities imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from advanced_qlora_trainer import ModelConfig, LoRAConfig, DataConfig, TrainingConfig
        
        # Test configuration creation
        model_config = ModelConfig(model_name_or_path="test-model")
        lora_config = LoRAConfig(r=16, alpha=32)
        data_config = DataConfig(max_seq_length=512)
        training_config = TrainingConfig(output_dir="./test_output")
        
        print("✅ Configuration objects created successfully")
        
        # Test model registry
        from utils.model_utils import ModelCompatibilityManager
        model_type = ModelCompatibilityManager.detect_model_family("meta-llama/Llama-3.1-8B")
        assert model_type == "llama"
        print("✅ Model compatibility detection working")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False

def test_file_structure():
    """Test file structure completeness."""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "advanced_qlora_trainer.py",
        "requirements.txt",
        "setup.py",
        "configs/production_config.yaml",
        "configs/edtech_config.yaml",
        "utils/model_utils.py",
        "utils/data_utils.py", 
        "utils/training_utils.py",
        "scripts/train_production.py",
        "scripts/inference_server.py",
        "scripts/validate_production.py",
        "examples/train_llama3_custom.py",
        "examples/train_qwen_multilingual.py",
        "examples/edtech_tutor_bot.py",
        "examples/edtech_content_generator.py",
        "docker/Dockerfile",
        "docker/docker-compose.yml",
        "tests/test_model_compatibility.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def main():
    """Run all validation tests."""
    print("🚀 Advanced QLoRA System Validation")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Import System", test_imports),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 SYSTEM IS READY FOR DEPLOYMENT!")
        return True
    else:
        print("⚠️  System needs fixes before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)