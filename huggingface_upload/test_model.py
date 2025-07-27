#!/usr/bin/env python3
"""
Test script to verify the MoR model works correctly after uploading to Hugging Face.

Usage:
    python test_model.py --model-name "your-username/mixture-of-recursions-360m"
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model_loading(model_name: str):
    """Test if the model can be loaded correctly."""
    
    print(f"🔍 Testing model loading: {model_name}")
    
    try:
        # Load tokenizer
        print("  📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"    ✅ Tokenizer loaded (vocab_size: {tokenizer.vocab_size:,})")
        
        # Load model
        print("  🤖 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"    ✅ Model loaded ({param_count:,} parameters)")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"    ❌ Failed to load model: {e}")
        raise

def test_text_generation(model, tokenizer):
    """Test text generation capabilities."""
    
    print("\n🧪 Testing text generation...")
    
    test_prompts = [
        "The key to artificial intelligence is",
        "Machine learning algorithms work by",
        "Once upon a time, in a world where robots and humans lived together,",
        "The future of technology will"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n  Test {i}/4:")
        print(f"    📝 Prompt: '{prompt}'")
        
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=inputs.input_ids.shape[1] + 30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = generated_text[len(prompt):].strip()
            
            print(f"    🤖 Generated: '{continuation}'")
            print("    ✅ Generation successful")
            
        except Exception as e:
            print(f"    ❌ Generation failed: {e}")
            raise

def test_model_info(model, tokenizer):
    """Display model information."""
    
    print("\n📊 Model Information:")
    
    # Model config
    config = model.config
    print(f"  🏗️  Architecture: {config.architectures[0] if config.architectures else 'Unknown'}")
    print(f"  📏 Hidden size: {config.hidden_size}")
    print(f"  🔢 Num layers: {config.num_hidden_layers}")
    print(f"  🧠 Num attention heads: {config.num_attention_heads}")
    print(f"  📖 Vocab size: {config.vocab_size:,}")
    print(f"  📐 Max position embeddings: {config.max_position_embeddings}")
    
    # Model size
    param_count = sum(p.numel() for p in model.parameters())
    param_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
    print(f"  ⚖️  Parameters: {param_count:,} ({param_size_mb:.1f} MB)")
    
    # Memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"  💾 GPU memory: {allocated:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description="Test MoR model from Hugging Face")
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name on Hugging Face (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip text generation tests"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting MoR model testing...")
    print(f"Model: {args.model_name}")
    
    try:
        # Test model loading
        model, tokenizer = test_model_loading(args.model_name)
        
        # Display model info
        test_model_info(model, tokenizer)
        
        # Test generation
        if not args.skip_generation:
            test_text_generation(model, tokenizer)
        else:
            print("\n⏭️  Skipping text generation tests")
        
        print("\n🎉 All tests passed! The model is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main() 