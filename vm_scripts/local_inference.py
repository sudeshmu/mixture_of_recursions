#!/usr/bin/env python3
"""
Local Inference Script for Mixture of Recursions Model
This script loads the trained model and runs inference locally.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def setup_paths():
    """Add project root to Python path for imports"""
    project_root = Path(__file__).parent.parent.absolute()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    return project_root

def load_model_and_tokenizer(model_path, device='auto'):
    """Load the trained model and tokenizer"""
    print(f"🔍 Loading model from: {model_path}")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    # Load tokenizer
    print("📝 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"⚠️  Failed to load tokenizer from HuggingFace, trying local...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("🧠 Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'),
            trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Trying to load with fallback settings...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map='cpu',
            trust_remote_code=True
        )
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Model device: {next(model.parameters()).device}")
    print(f"🔢 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, do_sample=True):
    """Generate text using the model"""
    print(f"💭 Generating text for prompt: '{prompt}'")
    
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        inputs = inputs.cuda()
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_text = generated_text[len(prompt):].strip()
    
    return new_text

def interactive_mode(model, tokenizer):
    """Run interactive chat mode"""
    print("\n🎮 Interactive Mode")
    print("Type 'quit' to exit, 'clear' to clear context")
    print("-" * 50)
    
    conversation_history = ""
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                conversation_history = ""
                print("🧹 Context cleared!")
                continue
            elif not user_input:
                continue
            
            # Add to conversation history
            if conversation_history:
                prompt = f"{conversation_history}\nHuman: {user_input}\nAssistant:"
            else:
                prompt = f"Human: {user_input}\nAssistant:"
            
            # Generate response
            response = generate_text(model, tokenizer, prompt, max_length=150)
            
            # Clean up response
            if "Human:" in response:
                response = response.split("Human:")[0].strip()
            
            print(f"🤖 Assistant: {response}")
            
            # Update conversation history (keep last 3 exchanges)
            conversation_history = f"{conversation_history}\nHuman: {user_input}\nAssistant: {response}"
            lines = conversation_history.strip().split('\n')
            if len(lines) > 6:  # Keep last 3 exchanges (6 lines)
                conversation_history = '\n'.join(lines[-6:])
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Local inference for Mixture of Recursions model")
    parser.add_argument("--model_path", type=str, default="trained_model", 
                       help="Path to the trained model directory")
    parser.add_argument("--prompt", type=str, 
                       help="Text prompt for generation")
    parser.add_argument("--max_length", type=int, default=100,
                       help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = setup_paths()
    
    # Resolve model path
    if not os.path.isabs(args.model_path):
        args.model_path = project_root / args.model_path
    
    print("🚀 Mixture of Recursions - Local Inference")
    print(f"📁 Project root: {project_root}")
    print(f"🎯 Model path: {args.model_path}")
    print(f"🖥️  Device: {args.device}")
    print(f"🌡️  Temperature: {args.temperature}")
    print(f"🎲 Top-p: {args.top_p}")
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
        
        if args.interactive:
            # Interactive mode
            interactive_mode(model, tokenizer)
        elif args.prompt:
            # Single prompt mode
            generated_text = generate_text(
                model, tokenizer, args.prompt, 
                args.max_length, args.temperature, args.top_p
            )
            print(f"\n📝 Generated text:")
            print("-" * 50)
            print(f"{args.prompt}{generated_text}")
            print("-" * 50)
        else:
            # Demo mode with sample prompts
            print("\n🎯 Demo Mode - Testing with sample prompts")
            
            sample_prompts = [
                "The future of artificial intelligence is",
                "In a world where technology has advanced beyond our imagination,",
                "The most important lesson I learned was",
                "Once upon a time, in a distant galaxy,"
            ]
            
            for prompt in sample_prompts:
                print(f"\n📝 Prompt: {prompt}")
                generated = generate_text(model, tokenizer, prompt, 80)
                print(f"🤖 Generated: {generated}")
                print("-" * 70)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 