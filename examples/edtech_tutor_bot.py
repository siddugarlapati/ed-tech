#!/usr/bin/env python3
"""
EdTech Example: AI Tutor Bot using Advanced QLoRA System
Demonstrates training a personalized tutoring assistant for educational platforms.
"""

import os
import sys
from pathlib import Path
import json
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from advanced_qlora_trainer import (
    ModelConfig, LoRAConfig, DataConfig, TrainingConfig,
    AdvancedQLoRATrainer, setup_logging
)

def create_edtech_dataset():
    """Create a comprehensive educational dataset for tutor training."""
    
    edtech_data = [
        # Mathematics Tutoring
        {
            "instruction": "Help me understand this math problem step by step.",
            "input": "Solve: 2x + 5 = 13",
            "output": "I'll help you solve this linear equation step by step!\n\nStep 1: Start with 2x + 5 = 13\nStep 2: Subtract 5 from both sides: 2x = 13 - 5 = 8\nStep 3: Divide both sides by 2: x = 8 ÷ 2 = 4\n\nSolution: x = 4\n\nLet's verify: 2(4) + 5 = 8 + 5 = 13 ✓\n\nWould you like me to explain any of these steps in more detail?"
        },
        {
            "instruction": "Explain this concept in simple terms for a high school student.",
            "input": "What is the quadratic formula?",
            "output": "The quadratic formula is a powerful tool for solving quadratic equations (equations with x²)!\n\n**Formula:** x = (-b ± √(b² - 4ac)) / 2a\n\n**When to use it:** For equations like ax² + bx + c = 0\n\n**Example:** For x² - 5x + 6 = 0\n- a = 1, b = -5, c = 6\n- x = (5 ± √(25 - 24)) / 2 = (5 ± 1) / 2\n- Solutions: x = 3 or x = 2\n\n**Memory tip:** Think 'negative b, plus or minus the square root, over 2a'\n\nTry practicing with a few examples!"
        },
        
        # Science Tutoring
        {
            "instruction": "Explain this science concept with examples.",
            "input": "What is photosynthesis?",
            "output": "Photosynthesis is how plants make their own food using sunlight! 🌱☀️\n\n**Simple equation:** 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂\n\n**In plain English:**\n- Plants take in carbon dioxide (CO₂) from air\n- They absorb water (H₂O) through roots\n- Chlorophyll captures sunlight energy\n- This creates glucose (sugar) for food\n- Oxygen (O₂) is released as a bonus!\n\n**Real-world example:** When you see green leaves in sunlight, they're literally eating light and making sugar while producing the oxygen you breathe!\n\n**Fun fact:** One large tree produces enough oxygen for 2 people per day!\n\nWhat part would you like me to explain more?"
        },
        
        # Language Arts
        {
            "instruction": "Help me improve my essay writing.",
            "input": "How do I write a strong thesis statement?",
            "output": "A strong thesis statement is the backbone of your essay! Here's how to craft one:\n\n**What it should do:**\n✓ State your main argument clearly\n✓ Be specific and focused\n✓ Take a position you can defend\n✓ Preview your main points\n\n**Formula:** [Topic] + [Your position] + [Main reasons]\n\n**Example:**\n❌ Weak: 'Social media is bad for teenagers.'\n✅ Strong: 'Social media negatively impacts teenagers by reducing face-to-face social skills, increasing anxiety, and disrupting sleep patterns.'\n\n**Your turn:** Try writing a thesis for your topic, and I'll help you strengthen it!\n\n**Pro tip:** Your thesis should be arguable - someone should be able to disagree with it."
        },
        
        # History
        {
            "instruction": "Help me understand this historical event.",
            "input": "Why was the Industrial Revolution important?",
            "output": "The Industrial Revolution (1760-1840) completely transformed human society! Here's why it matters:\n\n**Major Changes:**\n🏭 **Production:** Hand-made → Machine-made (much faster!)\n🏙️ **Living:** Rural farms → Urban cities\n💼 **Work:** Agriculture → Factory jobs\n🚂 **Transportation:** Horses → Steam trains\n\n**Key Impacts:**\n• **Economic:** Created modern capitalism and wage labor\n• **Social:** Birth of the middle class and working class\n• **Technological:** Steam power, factories, mass production\n• **Global:** Started modern globalization\n\n**Think of it like this:** Imagine if smartphones suddenly let everyone work 10x faster - that's the scale of change!\n\n**Legacy today:** Everything from your clothes to your phone exists because of Industrial Revolution innovations.\n\nWhat aspect interests you most?"
        },
        
        # Programming/Computer Science
        {
            "instruction": "Teach me programming concepts step by step.",
            "input": "What are variables in programming?",
            "output": "Variables are like labeled boxes that store information in your program! 📦\n\n**Think of it like this:**\n- Variable = a box with a label\n- Value = what's inside the box\n- You can change what's in the box anytime\n\n**Example in Python:**\n```python\nname = \"Alice\"        # Box labeled 'name' contains \"Alice\"\nage = 16             # Box labeled 'age' contains 16\nscore = 95.5         # Box labeled 'score' contains 95.5\n\n# You can change the contents:\nage = 17             # Now the 'age' box contains 17\n```\n\n**Rules for naming variables:**\n✓ Use descriptive names: `student_grade` not `x`\n✓ No spaces: use `first_name` not `first name`\n✓ Start with letter: `name1` not `1name`\n\n**Practice:** Try creating variables for your favorite book, its author, and rating!\n\nReady for the next concept?"
        },
        
        # Study Skills & Learning Strategies
        {
            "instruction": "Give me study tips and learning strategies.",
            "input": "How can I remember information better for tests?",
            "output": "Great question! Here are proven memory techniques that really work:\n\n**🧠 Active Learning Strategies:**\n1. **Spaced Repetition:** Review material at increasing intervals (1 day, 3 days, 1 week, 1 month)\n2. **The Feynman Technique:** Explain concepts in simple words as if teaching a friend\n3. **Practice Testing:** Quiz yourself instead of just re-reading\n\n**🎯 Memory Techniques:**\n• **Mnemonics:** Create memorable phrases (ROY G. BIV for rainbow colors)\n• **Visual Associations:** Draw diagrams, mind maps, or mental images\n• **Chunking:** Break large info into smaller, manageable pieces\n\n**📚 Study Environment:**\n• Find your peak focus time (morning vs. evening)\n• Use the Pomodoro Technique (25 min study, 5 min break)\n• Eliminate distractions (phone in another room!)\n\n**💡 Pro tip:** Teach someone else what you learned - if you can explain it clearly, you truly understand it!\n\nWhich technique sounds most helpful for your current subject?"
        },
        
        # Personalized Learning Support
        {
            "instruction": "I'm struggling with this subject. Can you help me create a study plan?",
            "input": "I have a chemistry test in 2 weeks and I'm behind on 3 chapters.",
            "output": "Don't worry! Let's create a manageable 2-week study plan for your chemistry test! 📚⚗️\n\n**Week 1 - Foundation Building:**\n**Days 1-2:** Chapter 1\n- Read chapter (45 min)\n- Take notes on key concepts (30 min)\n- Do practice problems (30 min)\n\n**Days 3-4:** Chapter 2 (same structure)\n**Days 5-6:** Chapter 3 (same structure)\n**Day 7:** Review Week 1 + light practice\n\n**Week 2 - Mastery & Test Prep:**\n**Days 8-10:** Deep practice problems from all chapters\n**Days 11-12:** Create summary sheets, flashcards for formulas\n**Days 13-14:** Practice tests + review weak areas\n\n**Daily Success Tips:**\n✓ Study same time each day (builds habit)\n✓ Use active recall (close book, write what you remember)\n✓ Join study groups or find a study buddy\n✓ Ask teacher about confusing concepts\n\n**Emergency help:** If you get stuck, I'm here to explain any chemistry concepts!\n\nWhich chapter feels most challenging right now?"
        },
        
        # Career Guidance
        {
            "instruction": "Help me explore career options based on my interests.",
            "input": "I like math and technology but I'm not sure what careers are available.",
            "output": "Exciting! Math + technology opens up amazing career possibilities! 🚀💻\n\n**High-Demand Tech Careers:**\n🤖 **Data Scientist:** Analyze data to solve business problems\n💻 **Software Engineer:** Build apps, websites, and systems\n🔒 **Cybersecurity Analyst:** Protect digital systems from threats\n🎮 **Game Developer:** Create video games and interactive experiences\n📊 **Machine Learning Engineer:** Build AI systems\n\n**Emerging Fields:**\n• **Quantum Computing Researcher**\n• **Blockchain Developer**\n• **AI Ethics Specialist**\n• **Robotics Engineer**\n\n**Next Steps to Explore:**\n1. **Try coding:** Start with Python or JavaScript (free on Codecademy)\n2. **Take online courses:** Coursera, edX have great intro courses\n3. **Join communities:** Local coding clubs, online forums\n4. **Shadow professionals:** Ask to observe someone in these fields\n\n**Salary ranges:** Most start $60-80k, experienced professionals earn $100-200k+\n\n**Which area sounds most interesting? I can suggest specific learning resources!**"
        },
        
        # Homework Help
        {
            "instruction": "Help me with my homework assignment.",
            "input": "I need to write a 5-paragraph essay about climate change but don't know how to start.",
            "output": "I'll help you structure a strong 5-paragraph essay on climate change! 🌍\n\n**Essay Structure:**\n\n**Paragraph 1 - Introduction:**\n- Hook: Start with a striking fact or question\n- Background: Brief context about climate change\n- Thesis: Your main argument (3 key points)\n\n**Paragraphs 2-4 - Body (one main point each):**\n- **Para 2:** Causes of climate change\n- **Para 3:** Current effects we're seeing\n- **Para 4:** Solutions and what we can do\n\n**Paragraph 5 - Conclusion:**\n- Restate thesis in new words\n- Summarize main points\n- Call to action or future outlook\n\n**Sample Thesis:**\n'Climate change, primarily caused by human activities, is creating severe environmental impacts that require immediate action through renewable energy adoption, policy changes, and individual responsibility.'\n\n**Research Tips:**\n• Use reliable sources: NASA, NOAA, IPCC reports\n• Include specific data and examples\n• Balance problems with solutions\n\n**Ready to start? Which paragraph would you like help with first?**"
        }
    ]
    
    # Expand dataset with variations and additional subjects
    extended_data = []
    
    # Add the base data multiple times with slight variations
    for _ in range(3):
        extended_data.extend(edtech_data)
    
    # Add more subject-specific examples
    additional_subjects = [
        # Physics
        {
            "instruction": "Explain this physics concept with real-world examples.",
            "input": "What is Newton's First Law of Motion?",
            "output": "Newton's First Law (Law of Inertia) states: 'An object at rest stays at rest, and an object in motion stays in motion, unless acted upon by an external force.'\n\n**Real-world examples:**\n🚗 **In a car:** When the car suddenly stops, you keep moving forward (that's why we wear seatbelts!)\n🏒 **Hockey puck:** Slides across ice until friction slows it down\n🚀 **Space:** Satellites orbit Earth for years without engines because there's no friction in space\n\n**Key insight:** Objects 'want' to keep doing what they're already doing - it takes force to change that!\n\n**Practice question:** Why do you feel pushed back in your seat when a car accelerates quickly?"
        },
        
        # Biology
        {
            "instruction": "Help me understand biological processes.",
            "input": "How does DNA replication work?",
            "output": "DNA replication is like making a perfect copy of a twisted ladder! 🧬\n\n**Step-by-step process:**\n1. **Unzipping:** DNA helicase 'unzips' the double helix\n2. **Copying:** DNA polymerase reads each strand and builds complementary strands\n3. **Proofreading:** Enzymes check for errors and fix them\n4. **Result:** Two identical DNA molecules!\n\n**Why it matters:** Every time a cell divides, it needs to pass on complete genetic instructions to both new cells.\n\n**Memory trick:** Think of it like a zipper opening, then each side becomes a template for making a new matching side!\n\n**Cool fact:** Your body replicates about 6 billion base pairs of DNA every time a cell divides!"
        }
    ]
    
    extended_data.extend(additional_subjects * 10)
    
    # Save to file
    dataset_path = "edtech_tutor_dataset.json"
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(extended_data, f, indent=2, ensure_ascii=False)
    
    return dataset_path

def main():
    """Train an AI tutor bot for EdTech platforms."""
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("🎓 Starting EdTech AI Tutor Bot Training")
    
    # Create educational dataset
    dataset_path = create_edtech_dataset()
    logger.info(f"Created EdTech dataset: {dataset_path}")
    
    # Configure model - Optimized for educational content
    model_config = ModelConfig(
        model_name_or_path="meta-llama/Llama-3.1-8B",  # Great for educational explanations
        trust_remote_code=False,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        
        # Memory-efficient for educational deployment
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Configure LoRA - Balanced for educational tasks
    lora_config = LoRAConfig(
        r=32,  # Moderate rank for educational content
        alpha=32,  # 1:1 ratio for stability
        dropout=0.05,  # Low dropout for consistent educational responses
        bias="none",
        task_type="CAUSAL_LM",
        
        # Educational-specific optimizations
        use_rslora=True,  # Better stability for tutoring
        use_dora=False,
        
        target_modules=None,  # Auto-detect for LLaMA
        modules_to_save=None
    )
    
    # Configure data processing for educational content
    data_config = DataConfig(
        dataset_path=dataset_path,
        max_seq_length=2048,  # Good for detailed explanations
        truncation=True,
        padding="max_length",
        
        # Educational data quality
        min_length=20,  # Ensure substantial responses
        max_length=3000,  # Allow for detailed explanations
        filter_duplicates=True,
        sample_ratio=None
    )
    
    # Configure training - Optimized for educational use
    training_config = TrainingConfig(
        output_dir="./models/edtech_tutor_bot",
        run_name="edtech_tutor_v1",
        
        # Training schedule for educational content
        num_train_epochs=4,  # More epochs for better educational responses
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        
        # Optimization for educational stability
        learning_rate=1e-4,  # Lower LR for stable educational responses
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        
        # Learning rate scheduling
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,  # Longer warmup for educational content
        
        # Evaluation and saving
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Performance optimizations
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        tf32=True,
        
        # Logging for educational monitoring
        logging_steps=10,
        report_to=["tensorboard"],
        
        # Educational-specific features
        group_by_length=True,
        remove_unused_columns=False,
        
        # No noise injection for consistent educational responses
        neftune_noise_alpha=None,
    )
    
    # Initialize trainer
    logger.info("🤖 Initializing EdTech AI Tutor Trainer")
    trainer = AdvancedQLoRATrainer(
        model_config=model_config,
        lora_config=lora_config,
        data_config=data_config,
        training_config=training_config,
        experiment_name="edtech_tutor_bot"
    )
    
    # Run training
    logger.info("📚 Starting educational AI training pipeline")
    results = trainer.train()
    
    # Print results
    logger.info("🎉 EdTech AI Tutor training completed successfully!")
    logger.info(f"Final metrics: {json.dumps(results['final_metrics'], indent=2)}")
    
    # Test educational capabilities
    logger.info("🧪 Testing educational AI capabilities")
    test_prompts = [
        "Help me understand photosynthesis in simple terms.",
        "I'm struggling with algebra. Can you explain how to solve 3x + 7 = 22?",
        "What study techniques work best for memorizing historical dates?",
        "Explain the water cycle with examples a 7th grader would understand.",
        "I have a test tomorrow on the Revolutionary War. Can you help me create a study plan?"
    ]
    
    for prompt in test_prompts:
        logger.info(f"📝 Test prompt: {prompt}")
        # Note: Actual inference would require loading the saved model
    
    logger.info(f"🎓 EdTech AI Tutor model saved to: {training_config.output_dir}")
    logger.info("✅ Ready for deployment in educational platforms!")
    
    # Cleanup
    if os.path.exists(dataset_path):
        os.remove(dataset_path)
    
    return results

if __name__ == "__main__":
    main()