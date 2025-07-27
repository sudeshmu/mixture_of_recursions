#!/usr/bin/env python3
"""
Test script to verify the MoR model loads and runs correctly
Usage: python3 test_inference.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

try:
    import torch
    from vm_scripts.local_inference import load_model_and_tokenizer, generate_text
    
    print("🔍 Testing MoR model setup...")
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
    
    # Test generation with diverse examples
    test_prompts = [
        # Technical/AI prompts
        "The key to artificial intelligence is",
        "Machine learning algorithms work by",
        "The difference between deep learning and traditional programming is",
        
        # Creative/storytelling prompts
        "Once upon a time, in a world where robots and humans lived together,",
        "The mysterious door at the end of the corridor led to",
        "If I could travel back in time, I would",
        
        # Factual/explanatory prompts
        "The process of photosynthesis involves",
        "Climate change affects our planet by",
        "The human brain is remarkable because",
        
        # Conversational prompts
        "Hello! I'm excited to tell you about",
        "The most important lesson I've learned is",
        "In my opinion, the future of technology will"
    ]
    
    print("\n🧪 Testing text generation...")
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        generated = generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7)
        print(f"🤖 Generated: {generated}")
    
    print("\n🎉 Model is ready for inference!")
    print("🚀 Run: python3 vm_scripts/local_inference.py --interactive")
    
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("📦 Install with: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("🔧 Check your environment setup") 