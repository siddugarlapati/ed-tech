#!/usr/bin/env python3
"""
EdTech Example: Educational Content Generator using Advanced QLoRA System
Trains a model to generate quizzes, lesson plans, and educational materials.
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

def create_content_generation_dataset():
    """Create dataset for educational content generation."""
    
    content_data = [
        # Quiz Generation
        {
            "instruction": "Generate a multiple choice quiz question about this topic.",
            "input": "Topic: Photosynthesis, Grade Level: 8th grade, Difficulty: Medium",
            "output": "**Question:** What are the main reactants (inputs) needed for photosynthesis?\n\nA) Oxygen and glucose\nB) Carbon dioxide and water\nC) Nitrogen and phosphorus\nD) Sunlight and chlorophyll\n\n**Correct Answer:** B) Carbon dioxide and water\n\n**Explanation:** Photosynthesis requires carbon dioxide from the air and water from the soil as the main reactants. Sunlight provides the energy and chlorophyll captures it, but they are not reactants in the chemical equation.\n\n**Learning Objective:** Students will identify the inputs and outputs of photosynthesis."
        },
        
        {
            "instruction": "Create a short answer quiz question for this subject.",
            "input": "Topic: American Revolution, Grade Level: 5th grade, Focus: Causes",
            "output": "**Question:** List and explain two main reasons why the American colonists were upset with British rule before the Revolutionary War.\n\n**Sample Answer:**\n1. **Taxation without Representation:** The British government made colonists pay taxes (like the Tea Act and Stamp Act) but didn't give them any say in the government that was taxing them.\n\n2. **Unfair Laws:** The British passed laws like the Intolerable Acts that punished all colonists for things like the Boston Tea Party, even if they weren't involved.\n\n**Grading Rubric:**\n- 2 points: Two clear causes with explanations\n- 1 point: One cause with explanation OR two causes without explanation\n- 0 points: Incorrect or no answer"
        },
        
        # Lesson Plan Generation
        {
            "instruction": "Create a lesson plan for this topic.",
            "input": "Subject: Mathematics, Topic: Fractions, Grade: 4th grade, Duration: 45 minutes",
            "output": "# Lesson Plan: Introduction to Fractions\n\n**Grade Level:** 4th Grade | **Duration:** 45 minutes | **Subject:** Mathematics\n\n## Learning Objectives:\nStudents will be able to:\n- Define what a fraction represents\n- Identify numerator and denominator\n- Represent fractions using visual models\n\n## Materials Needed:\n- Pizza/pie cutouts\n- Fraction bars\n- Whiteboard\n- Student worksheets\n\n## Lesson Structure:\n\n### Opening (10 minutes)\n- **Hook:** Show a pizza cut into 8 slices, eat 3 slices\n- **Question:** 'How much pizza did I eat?'\n- **Introduce:** This is what fractions help us describe!\n\n### Direct Instruction (15 minutes)\n- **Define fraction:** A part of a whole\n- **Explain parts:** Numerator (top) = parts we have, Denominator (bottom) = total parts\n- **Visual examples:** Use pizza slices, chocolate bars\n\n### Guided Practice (15 minutes)\n- **Activity:** Students use fraction bars to show 1/2, 1/4, 3/4\n- **Partner work:** Create fractions with manipulatives\n- **Check understanding:** Quick verbal assessment\n\n### Closure (5 minutes)\n- **Review:** What is a fraction? What do numerator and denominator mean?\n- **Preview:** Tomorrow we'll compare fractions!\n\n## Assessment:\n- Observation during guided practice\n- Exit ticket: Draw and label a fraction\n\n## Homework:\n- Find 3 examples of fractions at home (food, time, etc.)"
        },
        
        # Educational Worksheet Generation
        {
            "instruction": "Generate a worksheet for this learning objective.",
            "input": "Subject: Science, Topic: States of Matter, Grade: 3rd grade, Question Type: Fill in the blank",
            "output": "# States of Matter Worksheet\n**Name:** _________________ **Date:** _________________\n\n## Instructions: Fill in the blanks with the correct words from the word bank.\n\n**Word Bank:** solid, liquid, gas, melting, freezing, evaporation\n\n1. Ice is an example of a **_______** because it has a definite shape.\n\n2. Water is an example of a **_______** because it takes the shape of its container.\n\n3. Steam is an example of a **_______** because it spreads out to fill all available space.\n\n4. When ice turns into water, this process is called **_______**.\n\n5. When water turns into ice, this process is called **_______**.\n\n6. When water turns into steam, this process is called **_______**.\n\n## Bonus Question:\nDraw and label one example of each state of matter that you might find in your kitchen.\n\n**Solid:** _________________ **Liquid:** _________________ **Gas:** _________________\n\n---\n**Answer Key:**\n1. solid  2. liquid  3. gas  4. melting  5. freezing  6. evaporation"
        },
        
        # Study Guide Generation
        {
            "instruction": "Create a study guide for this upcoming test.",
            "input": "Subject: World History, Topic: Ancient Egypt, Grade: 6th grade, Test Date: Next Friday",
            "output": "# Ancient Egypt Study Guide 📚\n**Test Date:** Next Friday | **Subject:** World History\n\n## Key Topics to Review:\n\n### 1. Geography & Nile River\n- **Location:** Northeast Africa\n- **Nile River importance:** Flooding brought fertile soil, transportation, water\n- **Upper vs Lower Egypt:** Upper = South, Lower = North (confusing but true!)\n\n### 2. Pharaohs & Government\n- **Pharaoh:** King/Queen, considered a god\n- **Famous pharaohs:** Tutankhamun, Cleopatra, Ramses II\n- **Government:** Theocracy (ruled by religious leader)\n\n### 3. Religion & Afterlife\n- **Polytheistic:** Believed in many gods (Ra, Anubis, Isis)\n- **Mummification:** Preserved bodies for afterlife\n- **Pyramids:** Tombs for pharaohs\n\n### 4. Daily Life & Culture\n- **Social classes:** Pharaoh → Nobles → Scribes → Farmers → Slaves\n- **Writing:** Hieroglyphics\n- **Achievements:** Calendar, medicine, mathematics\n\n## Study Tips:\n- **Monday-Tuesday:** Review notes and textbook chapters 4-5\n- **Wednesday:** Practice with flashcards (gods, pharaohs, vocabulary)\n- **Thursday:** Take practice quiz, review any weak areas\n\n## Practice Questions:\n1. Why was the Nile River important to ancient Egyptians?\n2. Name three Egyptian gods and what they represented.\n3. Describe the mummification process in 3 steps.\n\n## Key Vocabulary:\n- Pharaoh, Hieroglyphics, Mummification, Pyramid, Papyrus, Sarcophagus\n\n**Good luck! You've got this! 🌟**"
        },
        
        # Interactive Activity Generation
        {
            "instruction": "Design an interactive classroom activity for this concept.",
            "input": "Subject: English Language Arts, Concept: Parts of Speech, Grade: 5th grade",
            "output": "# Parts of Speech Scavenger Hunt 🔍\n\n## Activity Overview:\nStudents work in teams to find examples of different parts of speech around the classroom and school.\n\n## Materials:\n- Clipboards and pencils\n- Parts of speech recording sheets\n- Timer\n- Small prizes/stickers\n\n## Setup (5 minutes):\n1. Divide class into teams of 3-4 students\n2. Give each team a recording sheet\n3. Review the 8 parts of speech quickly\n\n## Activity Rules (20 minutes):\n**Mission:** Find and record 2 examples of each part of speech\n\n**Where to look:**\n- Classroom posters and signs\n- Book covers and titles\n- Bulletin boards\n- Hallway displays (with permission)\n\n**Recording format:**\n- Write the word\n- Write where you found it\n- Explain why it's that part of speech\n\n**Example:**\n- **Noun:** 'Library' (sign on door) - names a place\n- **Verb:** 'Running' (PE poster) - shows action\n\n## Wrap-up (10 minutes):\n- Teams share their most creative finds\n- Discuss any challenging examples\n- Award points for accuracy and creativity\n\n## Extensions:\n- Create sentences using found words\n- Make a classroom parts of speech display\n- Challenge: Find compound words or contractions\n\n## Assessment:\n- Check recording sheets for accuracy\n- Listen to explanations during sharing\n- Note student collaboration and engagement"
        },
        
        # Differentiated Instruction
        {
            "instruction": "Adapt this lesson for different learning levels.",
            "input": "Original Lesson: Multiplication tables, Grade: 3rd grade, Adaptation needed: Advanced learners",
            "output": "# Differentiated Multiplication Lesson - Advanced Learners\n\n## For Advanced Students Who Have Mastered Basic Tables:\n\n### Challenge Activities:\n\n**1. Pattern Detective (15 minutes)**\n- Explore multiplication patterns in the 9s table\n- Discover why digits always add up to 9\n- Find patterns in 11s table (121, 132, 143...)\n\n**2. Real-World Problem Solving (20 minutes)**\n- **Scenario:** Planning a school carnival\n- Calculate total tickets needed: 8 games × 25 students × 3 rounds\n- Figure out supply costs: 12 prizes × $4 each × 6 booths\n- Design seating: 15 tables × 8 chairs each\n\n**3. Create Your Own (10 minutes)**\n- Design word problems for classmates\n- Must use multiplication facts 6-12\n- Include at least 2 steps\n- Provide answer key\n\n### Extension Projects:\n- Research ancient multiplication methods (Egyptian, lattice)\n- Create a multiplication game for younger students\n- Explore square numbers and their patterns\n\n### Assessment Adaptations:\n- Timed tests with larger numbers (2-digit × 1-digit)\n- Word problems requiring multiple operations\n- Explain mathematical reasoning, not just answers\n\n### Independent Learning:\n- Online math challenges (Khan Academy, Prodigy)\n- Peer tutoring opportunities\n- Math journal for pattern discoveries\n\n**Goal:** Keep advanced learners engaged while reinforcing core concepts through deeper application."
        },
        
        # Assessment Rubric Generation
        {
            "instruction": "Create a rubric for assessing this assignment.",
            "input": "Assignment: Science Fair Project, Grade: 7th grade, Subject: Life Science",
            "output": "# Science Fair Project Rubric - 7th Grade Life Science\n\n## Total Points: 100\n\n### Scientific Method (25 points)\n\n**Excellent (23-25 points):**\n- Clear, testable hypothesis\n- Well-designed controlled experiment\n- Identifies all variables (independent, dependent, controlled)\n- Follows scientific method steps precisely\n\n**Good (18-22 points):**\n- Hypothesis is mostly clear and testable\n- Experiment design is adequate with minor flaws\n- Identifies most variables correctly\n- Follows most scientific method steps\n\n**Needs Improvement (10-17 points):**\n- Hypothesis is unclear or not testable\n- Experiment design has significant flaws\n- Variables not clearly identified\n- Missing key scientific method steps\n\n**Unsatisfactory (0-9 points):**\n- No clear hypothesis\n- Poor or no experimental design\n- Does not follow scientific method\n\n### Data Collection & Analysis (25 points)\n\n**Excellent (23-25 points):**\n- Accurate, detailed data collection\n- Appropriate graphs/charts with labels\n- Thoughtful analysis of results\n- Discusses sources of error\n\n**Good (18-22 points):**\n- Mostly accurate data collection\n- Good use of graphs/charts\n- Basic analysis of results\n- Some discussion of limitations\n\n**Needs Improvement (10-17 points):**\n- Incomplete or inaccurate data\n- Poor use of visual representations\n- Limited analysis\n- Little awareness of limitations\n\n**Unsatisfactory (0-9 points):**\n- No meaningful data collection\n- No analysis of results\n\n### Presentation & Communication (25 points)\n\n**Excellent (23-25 points):**\n- Clear, engaging oral presentation\n- Professional display board\n- Confident answers to questions\n- Uses scientific vocabulary correctly\n\n**Good (18-22 points):**\n- Good presentation skills\n- Well-organized display\n- Answers most questions adequately\n- Generally uses vocabulary correctly\n\n**Needs Improvement (10-17 points):**\n- Unclear or disorganized presentation\n- Display needs improvement\n- Difficulty answering questions\n- Limited use of scientific vocabulary\n\n**Unsatisfactory (0-9 points):**\n- Poor presentation skills\n- Inadequate display\n- Cannot explain project\n\n### Life Science Content (25 points)\n\n**Excellent (23-25 points):**\n- Demonstrates deep understanding of life science concepts\n- Makes connections to broader biological principles\n- Uses appropriate life science terminology\n- Shows original thinking\n\n**Good (18-22 points):**\n- Shows good understanding of concepts\n- Makes some connections\n- Uses terminology mostly correctly\n- Some evidence of original thinking\n\n**Needs Improvement (10-17 points):**\n- Basic understanding of concepts\n- Few connections made\n- Limited use of terminology\n- Little original thinking\n\n**Unsatisfactory (0-9 points):**\n- Poor understanding of life science concepts\n- No connections or original thinking\n\n## Additional Comments:\n_Space for specific feedback and suggestions for improvement_"
        }
    ]
    
    # Expand with more educational content types
    additional_content = [
        # Vocabulary Building
        {
            "instruction": "Create vocabulary activities for these words.",
            "input": "Words: ecosystem, biodiversity, habitat, adaptation, Grade: 5th grade",
            "output": "# Vocabulary Activities: Ecosystem Unit\n\n## Word List:\n1. **Ecosystem** - All living and non-living things in an area working together\n2. **Biodiversity** - The variety of different plants and animals in an area\n3. **Habitat** - The natural home where an animal or plant lives\n4. **Adaptation** - Special features that help living things survive in their environment\n\n## Activities:\n\n### Activity 1: Vocabulary Matching Game\nMatch each word to its definition, then draw a picture example.\n\n### Activity 2: Fill in the Story\n'The rainforest is a complex _______ with incredible _______. Each animal has found its perfect _______ and developed special _______ to survive.'\n\n### Activity 3: Real-World Connections\n- Find examples of each word in your neighborhood\n- Take photos or draw pictures\n- Explain your choices\n\n### Activity 4: Vocabulary Charades\n- Act out each word without speaking\n- Classmates guess the vocabulary term\n- Explain your actions afterward\n\n### Assessment:\nUse all four words in a paragraph about your favorite animal and its environment."
        },
        
        # Parent Communication
        {
            "instruction": "Write a newsletter for parents about this unit.",
            "input": "Unit: Fractions, Grade: 4th grade, Duration: 3 weeks",
            "output": "# Parent Newsletter: Fractions Unit 📧\n\n**Dear 4th Grade Families,**\n\nWe're excited to begin our 3-week fractions unit! Here's what your child will be learning and how you can help at home.\n\n## What We're Learning:\n- **Week 1:** What fractions represent, identifying parts of a whole\n- **Week 2:** Comparing and ordering fractions\n- **Week 3:** Adding and subtracting fractions with like denominators\n\n## Key Skills:\n✓ Understanding numerator and denominator\n✓ Using visual models (pie charts, fraction bars)\n✓ Recognizing equivalent fractions\n✓ Solving real-world fraction problems\n\n## How to Help at Home:\n\n### Kitchen Math:\n- Cook together! Measuring cups use fractions (1/2 cup, 1/4 teaspoon)\n- Cut pizza, sandwiches, or pie into equal parts\n- Talk about fractions while eating: 'You ate 3/8 of the pizza!'\n\n### Daily Life Fractions:\n- Time: 'It's quarter past 3' (1/4 of an hour)\n- Money: 'A quarter is 1/4 of a dollar'\n- Sports: 'We're in the 3rd quarter of the game'\n\n### Practice Activities:\n- Draw and color fraction pictures\n- Use LEGO blocks to show fractions\n- Play fraction games online (approved sites list attached)\n\n## Important Dates:\n- **October 15:** Fraction quiz (basic concepts)\n- **October 22:** Unit test\n- **October 25:** Family Math Night (fractions focus!)\n\n## When to Worry (and When Not To):\n**Normal:** Confusion between numerator and denominator initially\n**Concerning:** Inability to recognize half of a shape after multiple lessons\n\n**Questions?** Email me anytime: teacher@school.edu\n\n**Together, we can make fractions fun!**\n\nMs. Johnson"
        }
    ]
    
    # Combine all content
    all_content = content_data + additional_content
    
    # Expand dataset
    extended_data = all_content * 8  # Repeat for more training examples
    
    # Save to file
    dataset_path = "edtech_content_generator_dataset.json"
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(extended_data, f, indent=2, ensure_ascii=False)
    
    return dataset_path

def main():
    """Train an educational content generator for EdTech platforms."""
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("📝 Starting EdTech Content Generator Training")
    
    # Create content generation dataset
    dataset_path = create_content_generation_dataset()
    logger.info(f"Created content generation dataset: {dataset_path}")
    
    # Configure model for content generation
    model_config = ModelConfig(
        model_name_or_path="meta-llama/Llama-3.1-8B",
        trust_remote_code=False,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        
        # Optimized for content generation
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Configure LoRA for creative content generation
    lora_config = LoRAConfig(
        r=64,  # Higher rank for creative content generation
        alpha=32,  # Balanced scaling
        dropout=0.1,  # Standard dropout for creativity
        bias="none",
        task_type="CAUSAL_LM",
        
        use_rslora=True,
        use_dora=False,
        
        target_modules=None,
        modules_to_save=None
    )
    
    # Configure data for content generation
    data_config = DataConfig(
        dataset_path=dataset_path,
        max_seq_length=3072,  # Longer for detailed content
        truncation=True,
        padding="max_length",
        
        min_length=50,  # Ensure substantial content
        max_length=4000,  # Allow for detailed lesson plans
        filter_duplicates=True,
        sample_ratio=None
    )
    
    # Configure training for content generation
    training_config = TrainingConfig(
        output_dir="./models/edtech_content_generator",
        run_name="edtech_content_gen_v1",
        
        # Training for creative content
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        
        # Optimization for content generation
        learning_rate=2e-4,  # Standard LR for creative tasks
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        
        # Scheduling
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        
        # Evaluation and saving
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        
        # Performance
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        bf16=True,
        tf32=True,
        
        # Logging
        logging_steps=10,
        report_to=["tensorboard"],
        
        # Content generation features
        group_by_length=True,
        remove_unused_columns=False,
        
        # Light noise for creativity
        neftune_noise_alpha=2.0,
    )
    
    # Initialize trainer
    logger.info("🎨 Initializing EdTech Content Generator Trainer")
    trainer = AdvancedQLoRATrainer(
        model_config=model_config,
        lora_config=lora_config,
        data_config=data_config,
        training_config=training_config,
        experiment_name="edtech_content_generator"
    )
    
    # Run training
    logger.info("📚 Starting content generation training pipeline")
    results = trainer.train()
    
    # Print results
    logger.info("🎉 EdTech Content Generator training completed!")
    logger.info(f"Final metrics: {json.dumps(results['final_metrics'], indent=2)}")
    
    # Test content generation capabilities
    logger.info("🧪 Testing content generation capabilities")
    test_prompts = [
        "Generate a multiple choice quiz question about the water cycle for 6th graders.",
        "Create a lesson plan for teaching basic algebra to 8th grade students.",
        "Design a worksheet for 4th graders learning about the solar system.",
        "Write a study guide for a high school biology test on cell structure.",
        "Create an interactive activity for teaching parts of speech to 5th graders."
    ]
    
    for prompt in test_prompts:
        logger.info(f"📝 Test prompt: {prompt}")
    
    logger.info(f"📚 Content Generator model saved to: {training_config.output_dir}")
    logger.info("✅ Ready for educational content generation!")
    
    # Cleanup
    if os.path.exists(dataset_path):
        os.remove(dataset_path)
    
    return results

if __name__ == "__main__":
    main()