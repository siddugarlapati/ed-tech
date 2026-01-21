#!/usr/bin/env python3
"""
Multi-Subject Model Training System
Trains specialized models for MBBS, Engineering, and all educational fields
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

sys.path.append(str(Path(__file__).parent))

from advanced_qlora_trainer import (
    ModelConfig, LoRAConfig, DataConfig, TrainingConfig,
    AdvancedQLoRATrainer, setup_logging
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubjectDatasetGenerator:
    """Generate training datasets for different subjects."""
    
    @staticmethod
    def generate_mbbs_dataset() -> List[Dict]:
        """Generate MBBS medical education dataset."""
        return [
            # Anatomy
            {
                "instruction": "Explain the structure and function of the human heart.",
                "input": "",
                "output": "The human heart is a muscular organ with four chambers: two atria (upper chambers) and two ventricles (lower chambers). The right side pumps deoxygenated blood to the lungs, while the left side pumps oxygenated blood to the body. The heart has four valves (tricuspid, pulmonary, mitral, and aortic) that ensure unidirectional blood flow."
            },
            {
                "instruction": "What are the layers of the skin?",
                "input": "",
                "output": "The skin has three main layers: 1) Epidermis - the outermost protective layer, 2) Dermis - contains blood vessels, nerves, hair follicles, and sweat glands, 3) Hypodermis (subcutaneous layer) - contains fat tissue for insulation and cushioning."
            },
            # Physiology
            {
                "instruction": "Explain the process of respiration.",
                "input": "",
                "output": "Respiration involves: 1) Ventilation - breathing in oxygen and out carbon dioxide, 2) External respiration - gas exchange in the lungs between alveoli and blood, 3) Internal respiration - gas exchange between blood and body tissues, 4) Cellular respiration - cells use oxygen to produce ATP energy."
            },
            # Pharmacology
            {
                "instruction": "What is the mechanism of action of aspirin?",
                "input": "",
                "output": "Aspirin (acetylsalicylic acid) works by irreversibly inhibiting cyclooxygenase (COX) enzymes, particularly COX-1 and COX-2. This prevents the synthesis of prostaglandins and thromboxanes, resulting in anti-inflammatory, analgesic, antipyretic, and antiplatelet effects."
            },
            # Pathology
            {
                "instruction": "Explain the pathophysiology of Type 2 Diabetes.",
                "input": "",
                "output": "Type 2 Diabetes develops due to insulin resistance and relative insulin deficiency. Cells become less responsive to insulin, requiring more insulin to maintain normal glucose levels. Over time, pancreatic beta cells cannot produce enough insulin, leading to hyperglycemia. Risk factors include obesity, sedentary lifestyle, genetics, and age."
            }
        ]
    
    @staticmethod
    def generate_engineering_dataset() -> List[Dict]:
        """Generate Engineering education dataset."""
        return [
            # Mechanical Engineering
            {
                "instruction": "Explain the first law of thermodynamics.",
                "input": "",
                "output": "The first law of thermodynamics states that energy cannot be created or destroyed, only converted from one form to another. Mathematically: ΔU = Q - W, where ΔU is the change in internal energy, Q is heat added to the system, and W is work done by the system. This is also known as the law of conservation of energy."
            },
            {
                "instruction": "What is the difference between stress and strain?",
                "input": "",
                "output": "Stress is the internal force per unit area within a material (σ = F/A), measured in Pascals. Strain is the deformation or change in dimension relative to the original dimension (ε = ΔL/L), dimensionless. Stress causes strain, and their relationship defines material properties like Young's modulus (E = σ/ε)."
            },
            # Electrical Engineering
            {
                "instruction": "Explain Ohm's Law and its applications.",
                "input": "",
                "output": "Ohm's Law states that voltage (V) across a conductor is directly proportional to current (I) flowing through it: V = IR, where R is resistance. Applications include: circuit analysis, power calculations (P = VI = I²R = V²/R), designing resistor networks, and troubleshooting electrical systems."
            },
            # Computer Science Engineering
            {
                "instruction": "Explain the time complexity of binary search.",
                "input": "",
                "output": "Binary search has O(log n) time complexity. It works by repeatedly dividing the search space in half. With each comparison, it eliminates half of the remaining elements. For n elements, it takes at most log₂(n) comparisons. For example, searching 1 million elements requires only about 20 comparisons."
            },
            # Civil Engineering
            {
                "instruction": "What is the purpose of reinforcement in concrete?",
                "input": "",
                "output": "Reinforcement (typically steel bars or mesh) is added to concrete because concrete is strong in compression but weak in tension. Steel reinforcement provides tensile strength, prevents cracking, increases ductility, and allows concrete to resist bending moments and shear forces. This creates reinforced concrete, suitable for beams, columns, and slabs."
            }
        ]

    
    @staticmethod
    def generate_mathematics_dataset() -> List[Dict]:
        """Generate Mathematics dataset."""
        return [
            {
                "instruction": "Explain the Pythagorean theorem with an example.",
                "input": "",
                "output": "The Pythagorean theorem states that in a right triangle, a² + b² = c², where c is the hypotenuse and a, b are the other sides. Example: If a = 3 and b = 4, then c² = 9 + 16 = 25, so c = 5. This forms a 3-4-5 right triangle."
            },
            {
                "instruction": "What is the derivative of sin(x)?",
                "input": "",
                "output": "The derivative of sin(x) is cos(x). This can be proven using the limit definition of derivatives. It's a fundamental result in calculus used extensively in physics, engineering, and mathematical analysis."
            },
            {
                "instruction": "Explain integration by parts.",
                "input": "",
                "output": "Integration by parts is based on the product rule: ∫u dv = uv - ∫v du. Choose u (to differentiate) and dv (to integrate) strategically. Example: ∫x·eˣ dx, let u=x, dv=eˣdx, then du=dx, v=eˣ, giving x·eˣ - ∫eˣ dx = x·eˣ - eˣ + C = eˣ(x-1) + C."
            }
        ]
    
    @staticmethod
    def generate_physics_dataset() -> List[Dict]:
        """Generate Physics dataset."""
        return [
            {
                "instruction": "Explain Newton's Second Law of Motion.",
                "input": "",
                "output": "Newton's Second Law states that Force equals mass times acceleration: F = ma. This means the acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass. A 10 kg object with 20 N force will accelerate at 2 m/s²."
            },
            {
                "instruction": "What is the difference between AC and DC current?",
                "input": "",
                "output": "DC (Direct Current) flows in one direction with constant voltage, like batteries. AC (Alternating Current) periodically reverses direction, typically 50-60 Hz in power grids. AC is easier to transform to different voltages and transmit over long distances, while DC is used in electronics and batteries."
            }
        ]
    
    @staticmethod
    def generate_chemistry_dataset() -> List[Dict]:
        """Generate Chemistry dataset."""
        return [
            {
                "instruction": "Explain the concept of pH.",
                "input": "",
                "output": "pH measures the acidity or basicity of a solution on a scale of 0-14. pH = -log[H⁺], where [H⁺] is hydrogen ion concentration. pH 7 is neutral (pure water), <7 is acidic (more H⁺), >7 is basic/alkaline (more OH⁻). Each pH unit represents a 10-fold change in acidity."
            },
            {
                "instruction": "What is an oxidation-reduction reaction?",
                "input": "",
                "output": "Redox reactions involve transfer of electrons between species. Oxidation is loss of electrons (increase in oxidation state), reduction is gain of electrons (decrease in oxidation state). Example: 2Na + Cl₂ → 2NaCl. Sodium is oxidized (0 to +1), chlorine is reduced (0 to -1). Remember: OIL RIG (Oxidation Is Loss, Reduction Is Gain)."
            }
        ]
    
    @staticmethod
    def generate_biology_dataset() -> List[Dict]:
        """Generate Biology dataset."""
        return [
            {
                "instruction": "Explain the process of photosynthesis.",
                "input": "",
                "output": "Photosynthesis converts light energy into chemical energy: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂. It occurs in two stages: 1) Light reactions (in thylakoids) - capture light energy, split water, produce ATP and NADPH, 2) Calvin cycle (in stroma) - use ATP and NADPH to fix CO₂ into glucose."
            },
            {
                "instruction": "What is DNA replication?",
                "input": "",
                "output": "DNA replication is the process of copying DNA before cell division. Steps: 1) Helicase unwinds the double helix, 2) Primase adds RNA primers, 3) DNA polymerase adds nucleotides to the 3' end, 4) Leading strand is continuous, lagging strand forms Okazaki fragments, 5) Ligase joins fragments. Result: two identical DNA molecules."
            }
        ]


def train_subject_model(subject: str, dataset: List[Dict], base_model: str) -> str:
    """Train a model for a specific subject."""
    
    logger.info(f"Training {subject} model...")
    
    # Expand dataset
    expanded_dataset = dataset * 20  # Repeat for more training data
    
    # Save dataset
    dataset_path = f"datasets/{subject}_dataset.json"
    os.makedirs("datasets", exist_ok=True)
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(expanded_dataset, f, indent=2, ensure_ascii=False)
    
    # Configure model
    model_config = ModelConfig(
        model_name_or_path=base_model,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Configure LoRA
    lora_config = LoRAConfig(
        r=32,
        alpha=32,
        dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,
        use_dora=False,
        target_modules=None,
        modules_to_save=None
    )
    
    # Configure data
    data_config = DataConfig(
        dataset_path=dataset_path,
        max_seq_length=1024,
        truncation=True,
        padding="max_length",
        min_length=10,
        max_length=2048,
        filter_duplicates=True,
        sample_ratio=None
    )
    
    # Configure training
    training_config = TrainingConfig(
        output_dir=f"./models/{subject}_model",
        run_name=f"{subject}_specialized",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        tf32=True,
        logging_steps=10,
        report_to=["tensorboard"],
        group_by_length=True,
        remove_unused_columns=False,
        neftune_noise_alpha=None,
    )
    
    # Initialize trainer
    trainer = AdvancedQLoRATrainer(
        model_config=model_config,
        lora_config=lora_config,
        data_config=data_config,
        training_config=training_config,
        experiment_name=f"{subject}_specialized"
    )
    
    # Train
    results = trainer.train()
    
    logger.info(f"{subject} model training completed!")
    return training_config.output_dir


def main():
    """Train all subject-specific models."""
    
    setup_logging("INFO")
    logger.info("Starting Multi-Subject Model Training")
    
    generator = SubjectDatasetGenerator()
    
    # Define subjects and their datasets
    subjects = {
        "mbbs": {
            "dataset": generator.generate_mbbs_dataset(),
            "model": "meta-llama/Llama-3.1-8B",
            "description": "Medical education (Anatomy, Physiology, Pharmacology, Pathology)"
        },
        "engineering": {
            "dataset": generator.generate_engineering_dataset(),
            "model": "meta-llama/Llama-3.1-8B",
            "description": "Engineering (Mechanical, Electrical, Civil, Computer Science)"
        },
        "mathematics": {
            "dataset": generator.generate_mathematics_dataset(),
            "model": "meta-llama/Llama-3.1-8B",
            "description": "Mathematics (Algebra, Calculus, Geometry)"
        },
        "physics": {
            "dataset": generator.generate_physics_dataset(),
            "model": "meta-llama/Llama-3.1-8B",
            "description": "Physics (Mechanics, Electromagnetism, Thermodynamics)"
        },
        "chemistry": {
            "dataset": generator.generate_chemistry_dataset(),
            "model": "Qwen/Qwen2.5-7B",
            "description": "Chemistry (Organic, Inorganic, Physical)"
        },
        "biology": {
            "dataset": generator.generate_biology_dataset(),
            "model": "Qwen/Qwen2.5-7B",
            "description": "Biology (Cell Biology, Genetics, Ecology)"
        }
    }
    
    print("\n" + "="*60)
    print("Multi-Subject Model Training System")
    print("="*60)
    print(f"\nTraining {len(subjects)} specialized models:\n")
    
    for subject, info in subjects.items():
        print(f"  • {subject.upper()}: {info['description']}")
        print(f"    Base Model: {info['model']}")
        print(f"    Dataset Size: {len(info['dataset'])} examples")
        print()
    
    # Train each subject
    trained_models = {}
    for subject, info in subjects.items():
        print(f"\n{'='*60}")
        print(f"Training {subject.upper()} Model")
        print(f"{'='*60}\n")
        
        try:
            model_path = train_subject_model(
                subject,
                info['dataset'],
                info['model']
            )
            trained_models[subject] = model_path
            print(f"✅ {subject.upper()} model trained successfully!")
            print(f"   Saved to: {model_path}\n")
        except Exception as e:
            print(f"❌ Error training {subject} model: {e}\n")
    
    # Save model registry
    registry = {
        "trained_models": trained_models,
        "subjects": subjects,
        "training_date": str(Path(__file__).stat().st_mtime)
    }
    
    with open("model_registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\n✅ Successfully trained {len(trained_models)} models")
    print(f"📝 Model registry saved to: model_registry.json")
    print("\nTrained models:")
    for subject, path in trained_models.items():
        print(f"  • {subject}: {path}")
    
    print("\n🚀 Models ready for deployment!")
    print("   Use educational_ai_system.py to start teaching\n")


if __name__ == "__main__":
    main()
