#!/usr/bin/env python3
"""
Comprehensive Inference Test for MoR Model from Hugging Face Hub
Tests the uploaded model with 50 diverse prompts across multiple categories
Usage: python3 comprehensive_hf_inference_test.py --model-name your-username/mixture-of-recursions-360m
"""
import sys
import time
import argparse
from pathlib import Path

def install_requirements():
    """Install required packages if not available."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        return True
    except ImportError:
        print("📦 Installing required packages...")
        import subprocess
        packages = ["torch", "transformers", "accelerate"]
        for package in packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        return True

def load_hf_model(model_name: str, device: str = "auto"):
    """Load model and tokenizer from Hugging Face Hub."""
    
    print(f"🔄 Loading model from Hugging Face: {model_name}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Load tokenizer
        print("  📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"    ✅ Tokenizer loaded (vocab_size: {tokenizer.vocab_size:,})")
        
        # Load model
        print("  🤖 Loading model...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device if torch.cuda.is_available() else "cpu",
            trust_remote_code=True  # Required for custom MoR models
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"    ✅ Model loaded ({param_count:,} parameters)")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"    ❌ Failed to load model: {e}")
        print("    💡 Make sure the model exists and you have access to it")
        raise

def generate_text_hf(model, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7, do_sample: bool = True):
    """Generate text using the HuggingFace model."""
    
    import torch
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Move to device if CUDA available
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            repetition_penalty=1.1,
            top_p=0.9
        )
    
    # Decode only the new tokens (excluding input)
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text.strip()

def run_comprehensive_test(model_name: str, device: str = "auto"):
    """Run comprehensive inference test on HuggingFace model."""
    
    print("🧪 Comprehensive MoR Model Test from HuggingFace - 50 Test Cases")
    print("=" * 70)
    print(f"🤗 Model: {model_name}")
    
    # Install requirements and load model
    install_requirements()
    model, tokenizer = load_hf_model(model_name, device)
    
    print("=" * 70)
    
    # 50 Diverse Test Cases (same as original)
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
            generated = generate_text_hf(
                model, tokenizer, prompt, 
                max_new_tokens=50, 
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
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
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
    
    # Model info
    print(f"\n🤗 Model Information:")
    print(f"  📍 HuggingFace Model: {model_name}")
    print(f"  🔢 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  📝 Vocab Size: {tokenizer.vocab_size:,}")
    
    # Device info
    import torch
    if torch.cuda.is_available():
        print(f"  🖥️  Device: GPU ({torch.cuda.get_device_name()})")
        print(f"  💾 GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    else:
        print(f"  🖥️  Device: CPU")
    
    print("\n🎉 Comprehensive HuggingFace model test completed!")
    
    if successful_tests == 50:
        print("🚀 Perfect! Your MoR model from HuggingFace is working excellently!")
    elif successful_tests >= 45:
        print("✨ Great! Your MoR model is working very well!")
    elif successful_tests >= 40:
        print("👍 Good! Your MoR model is working well!")
    else:
        print("⚠️  Some issues detected. Check the failed tests above.")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test MoR model from HuggingFace Hub")
    parser.add_argument(
        "--model-name",
        required=True,
        help="HuggingFace model name (e.g., 'username/mixture-of-recursions-360m')"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use ('auto', 'cpu', 'cuda', etc.)"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_comprehensive_test(args.model_name, args.device)
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 