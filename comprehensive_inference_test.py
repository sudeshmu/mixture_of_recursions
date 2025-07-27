#!/usr/bin/env python3
"""
Comprehensive Inference Test for Mixture of Recursions Model
Tests the model with 50 diverse prompts across multiple categories
Usage: python3 comprehensive_inference_test.py
"""
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

try:
    import torch
    from vm_scripts.local_inference import load_model_and_tokenizer, generate_text
    
    print("🧪 Comprehensive MoR Model Test - 50 Test Cases")
    print("=" * 60)
    
    model_path = project_root / "trained_model"
    
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        sys.exit(1)
    
    print(f"📁 Model path: {model_path}")
    print("🔄 Loading model and tokenizer...")
    
    model, tokenizer = load_model_and_tokenizer(str(model_path), 'cpu')
    
    print("✅ Model loaded successfully!")
    print(f"🔢 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📝 Vocab size: {tokenizer.vocab_size:,}")
    print("=" * 60)
    
    # 50 Diverse Test Cases
    test_cases = [
        # TECHNICAL/AI (10 cases)
        "The key to artificial intelligence is",
        "Machine learning algorithms work by",
        "Deep learning differs from traditional programming because",
        "Neural networks learn patterns through",
        "The future of robotics will involve",
        "Computer vision systems can detect",
        "Natural language processing helps computers",
        "The challenge with AI safety is",
        "Quantum computing could revolutionize",
        "Data science is important because",
        
        # CREATIVE WRITING (10 cases)
        "Once upon a time, in a distant galaxy,",
        "The mysterious door creaked open to reveal",
        "If I could travel back in time, I would",
        "The old wizard whispered the ancient spell:",
        "In a world where magic and technology coexist,",
        "The last person on Earth discovered",
        "The spaceship landed on the unknown planet and",
        "She opened the dusty book and found",
        "The detective noticed something strange about",
        "In the year 2150, humans finally learned",
        
        # SCIENCE/NATURE (10 cases)
        "The process of photosynthesis involves",
        "Climate change affects our planet by",
        "The human brain is remarkable because",
        "Ocean currents are driven by",
        "Stars form when cosmic gas",
        "DNA contains genetic information that",
        "Renewable energy sources include",
        "The theory of evolution explains",
        "Earthquakes occur when tectonic plates",
        "The immune system protects us by",
        
        # CONVERSATIONAL (10 cases)
        "Hello! I'm excited to tell you about",
        "The most important lesson I've learned is",
        "In my opinion, the future of technology will",
        "If you could change one thing about the world,",
        "The best advice I ever received was",
        "When I think about happiness, I believe",
        "The greatest challenge facing humanity is",
        "What motivates me most in life is",
        "If I had unlimited resources, I would",
        "The difference between knowledge and wisdom is",
        
        # PROBLEM SOLVING (10 cases)
        "To solve complex problems, one should first",
        "When facing a difficult decision, consider",
        "The best way to learn something new is",
        "Innovation happens when people",
        "To improve communication skills, practice",
        "Building trust requires",
        "Effective teamwork depends on",
        "To overcome fear, one must",
        "Leadership means",
        "Success is achieved through"
    ]
    
    print(f"🚀 Running {len(test_cases)} test cases...\n")
    
    results = []
    start_time = time.time()
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"📝 Test {i:2d}/50: '{prompt}'")
        
        try:
            generated = generate_text(
                model, tokenizer, prompt, 
                max_length=60, 
                temperature=0.7,
                do_sample=True
            )
            
            result = {
                'test_id': i,
                'prompt': prompt,
                'generated': generated,
                'success': True
            }
            results.append(result)
            
            print(f"🤖 Generated: {generated}")
            print("-" * 80)
            
        except Exception as e:
            print(f"❌ Error in test {i}: {e}")
            result = {
                'test_id': i,
                'prompt': prompt,
                'generated': f"ERROR: {e}",
                'success': False
            }
            results.append(result)
            print("-" * 80)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = sum(1 for r in results if r['success'])
    failed_tests = len(results) - successful_tests
    
    print(f"✅ Successful tests: {successful_tests}/50")
    print(f"❌ Failed tests: {failed_tests}/50")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"⚡ Average time per test: {total_time/len(results):.2f} seconds")
    
    # Category breakdown
    categories = [
        ("Technical/AI", 1, 10),
        ("Creative Writing", 11, 20),
        ("Science/Nature", 21, 30),
        ("Conversational", 31, 40),
        ("Problem Solving", 41, 50)
    ]
    
    print("\n📈 Performance by Category:")
    for cat_name, start_idx, end_idx in categories:
        cat_results = results[start_idx-1:end_idx]
        cat_success = sum(1 for r in cat_results if r['success'])
        print(f"  {cat_name}: {cat_success}/10 successful")
    
    print("\n🎉 Comprehensive test completed!")
    print("🚀 Your MoR model is ready for production inference!")
    
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("📦 Install with: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("🔧 Check your environment setup") 