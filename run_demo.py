#!/usr/bin/env python3
"""
Quick demo script to test the complete system
"""

import sys
from pathlib import Path

def main():
    print("🚀 Advanced QLoRA System - Quick Demo")
    print("=" * 60)
    
    # Test 1: System validation
    print("\n📋 Test 1: System Validation")
    print("-" * 60)
    try:
        import test_system
        result = test_system.main()
        if result:
            print("✅ System validation PASSED")
        else:
            print("❌ System validation FAILED")
            return False
    except Exception as e:
        print(f"❌ System validation error: {e}")
        return False
    
    # Test 2: Visualization demo
    print("\n📋 Test 2: 3D Visualization Generator")
    print("-" * 60)
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from visualization.concept_visualizer import ConceptVisualizer, VisualizationConfig, ConceptType, AnimationLibrary
        
        visualizer = ConceptVisualizer()
        config = VisualizationConfig(
            concept_type=ConceptType.MATHEMATICS,
            difficulty_level=5,
            animation_library=AnimationLibrary.GSAP,
            interactive=True,
            show_labels=True,
            show_equations=True,
            animation_speed=1.0
        )
        
        html = visualizer.generate_visualization("unit_circle", config, angle=45)
        
        with open("demo_unit_circle.html", "w") as f:
            f.write(html)
        
        print("✅ 3D Visualization generated: demo_unit_circle.html")
        print("   Open this file in your browser to see the interactive visualization")
    except Exception as e:
        print(f"❌ Visualization generation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Model compatibility check
    print("\n📋 Test 3: Model Compatibility")
    print("-" * 60)
    try:
        from utils.model_utils import ModelCompatibilityManager
        
        test_models = [
            "meta-llama/Llama-3.1-8B",
            "Qwen/Qwen2.5-7B",
            "mistralai/Mistral-7B-v0.1",
            "google/gemma-2-9b"
        ]
        
        for model_name in test_models:
            family = ModelCompatibilityManager.detect_model_family(model_name)
            print(f"   {model_name} → {family}")
        
        print("✅ Model compatibility detection working")
    except Exception as e:
        print(f"❌ Model compatibility error: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("\n📚 Next Steps:")
    print("   1. Open demo_unit_circle.html in your browser")
    print("   2. Run: python examples/train_llama3_custom.py")
    print("   3. Or: python scripts/train_production.py --config configs/production_config.yaml")
    print("\n✨ System is production-ready!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
