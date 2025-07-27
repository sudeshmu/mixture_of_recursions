#!/usr/bin/env python3
"""
Complete script to create a Hugging Face repository and upload the MoR model.

This script will:
1. Create a new repository on Hugging Face
2. Prepare all necessary model files
3. Upload the model with proper documentation
4. Verify the upload was successful

Usage:
    python create_repo.py --username "your-username" --model-name "mixture-of-recursions-360m"

Note: You'll be prompted for your Hugging Face token or you can pass it via --token
"""

import os
import sys
import argparse
import getpass
from pathlib import Path

def get_hf_token(token_arg: str = None) -> str:
    """Get Hugging Face token from argument or user input."""
    
    if token_arg:
        return token_arg
    
    # Check environment variable
    token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    if token:
        print("🔑 Using token from environment variable")
        return token
    
    # Prompt user
    print("🔑 Hugging Face token required for upload")
    print("   You can get your token from: https://huggingface.co/settings/tokens")
    token = getpass.getpass("   Enter your Hugging Face token: ")
    
    if not token:
        print("❌ Token is required for upload")
        sys.exit(1)
    
    return token

def validate_repo_name(username: str, model_name: str) -> str:
    """Validate and construct repository name."""
    
    # Basic validation
    if not username or not model_name:
        raise ValueError("Username and model name are required")
    
    # Remove any problematic characters
    username = username.strip().replace(' ', '-').lower()
    model_name = model_name.strip().replace(' ', '-').lower()
    
    repo_name = f"{username}/{model_name}"
    
    print(f"📝 Repository will be created as: {repo_name}")
    return repo_name

def main():
    parser = argparse.ArgumentParser(
        description="Create and upload MoR model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python create_repo.py --username "johndoe" --model-name "mor-360m"
    python create_repo.py --username "myorg" --model-name "mixture-of-recursions-360m" --private
    python create_repo.py --username "user" --model-name "mor-llama" --token "hf_xxx"
        """
    )
    
    parser.add_argument(
        "--username",
        required=True,
        help="Your Hugging Face username or organization name"
    )
    parser.add_argument(
        "--model-name", 
        required=True,
        help="Name for the model repository (e.g., 'mixture-of-recursions-360m')"
    )
    parser.add_argument(
        "--token",
        help="Hugging Face authentication token (will prompt if not provided)"
    )
    parser.add_argument(
        "--model-dir",
        default="../trained_model",
        help="Path to the trained model directory (default: ../trained_model)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    parser.add_argument(
        "--description",
        help="Custom description for the repository"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare files but don't upload (for testing)"
    )
    
    args = parser.parse_args()
    
    print("🚀 MoR Model Upload to Hugging Face")
    print("=" * 50)
    
    try:
        # Validate inputs
        repo_name = validate_repo_name(args.username, args.model_name)
        
        # Get token
        if not args.dry_run:
            token = get_hf_token(args.token)
        else:
            token = "dry-run-token"
            print("🧪 Dry run mode - no actual upload will occur")
        
        # Validate model directory
        model_dir = Path(args.model_dir).resolve()
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            print("   Please ensure your trained model exists at the specified path")
            sys.exit(1)
        
        print(f"📁 Using model from: {model_dir}")
        
        # Check for required files
        required_files = ["pytorch_model.bin", "config.json"]
        missing_files = []
        for file_name in required_files:
            if not (model_dir / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"❌ Missing required files: {', '.join(missing_files)}")
            print("   Please ensure your model training completed successfully")
            sys.exit(1)
        
        # Import and run upload script
        print("\n📦 Preparing for upload...")
        
        current_dir = Path(__file__).parent
        upload_script = current_dir / "upload_to_hf.py"
        
        if not upload_script.exists():
            print(f"❌ Upload script not found: {upload_script}")
            sys.exit(1)
        
        # Prepare command
        cmd_args = [
            sys.executable, 
            str(upload_script),
            "--repo-name", repo_name,
            "--token", token,
            "--model-dir", str(model_dir),
            "--commit-message", f"Upload {args.model_name} MoR model"
        ]
        
        if args.private:
            cmd_args.append("--private")
        
        if args.dry_run:
            print("🧪 Dry run - would execute:")
            print("   " + " ".join(cmd_args[:4] + ["--token", "[HIDDEN]"] + cmd_args[6:]))
            print("\n✅ Dry run completed successfully")
            return
        
        # Execute upload
        print("🚀 Starting upload process...")
        import subprocess
        result = subprocess.run(cmd_args, capture_output=False)
        
        if result.returncode == 0:
            print(f"\n🎉 Success! Your model is now available at:")
            print(f"   https://huggingface.co/{repo_name}")
            print(f"\n💡 To use your model:")
            print(f"   from transformers import AutoTokenizer, AutoModelForCausalLM")
            print(f"   tokenizer = AutoTokenizer.from_pretrained('{repo_name}')")
            print(f"   model = AutoModelForCausalLM.from_pretrained('{repo_name}')")
            
            # Offer to test the model
            test_script = current_dir / "test_model.py"
            if test_script.exists():
                print(f"\n🧪 To test your uploaded model:")
                print(f"   python test_model.py --model-name '{repo_name}'")
        else:
            print(f"\n❌ Upload failed with exit code {result.returncode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Upload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 