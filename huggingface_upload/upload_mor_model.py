#!/usr/bin/env python3
"""
Upload MoR model to Hugging Face Hub.
This script uploads your trained MoR model to Hugging Face using a secure token prompt.

Usage:
    python upload_mor_model.py --username your-username --model-name mixture-of-recursions-360m
"""

import os
import sys
import json
import shutil
import subprocess
import getpass
from pathlib import Path

def get_hf_token():
    """Get HuggingFace token securely."""
    # Check environment variable first
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    if token:
        print("🔑 Using token from environment variable")
        return token
    
    # Prompt user securely
    print("🔑 HuggingFace token required for upload")
    print("   Get your token from: https://huggingface.co/settings/tokens")
    print("   Make sure it has 'write' permissions")
    token = getpass.getpass("   Enter your HuggingFace token: ")
    
    if not token:
        print("❌ Token is required for upload")
        sys.exit(1)
    
    return token

def test_token(token):
    """Test if the HuggingFace token is valid."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f"✅ Token valid for user: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Token validation failed: {e}")
        return False

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    requirements = [
        "huggingface_hub>=0.17.0",
        "transformers>=4.35.0",
        "torch>=2.0.0"
    ]
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")
            return False
    
    print("✅ All packages installed successfully")
    return True

def prepare_upload_directory(model_dir: Path, upload_dir: Path, repo_name: str):
    """Prepare the upload directory with all necessary files."""
    
    print(f"📁 Preparing upload directory: {upload_dir}")
    
    # Clean and create upload directory
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model files
    model_files = [
        "pytorch_model.bin",
        "config.json",
        "generation_config.json"
    ]
    
    for file_name in model_files:
        source_file = model_dir / file_name
        if source_file.exists():
            shutil.copy2(source_file, upload_dir / file_name)
            print(f"  ✅ Copied {file_name}")
        else:
            print(f"  ⚠️  {file_name} not found")
    
    # Update config.json for HF compatibility
    config_path = upload_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Add MoR-specific configuration
        config.update({
            "model_type": "mor_llama",
            "auto_map": {
                "AutoConfig": "modeling_mor.MoRConfig",
                "AutoModelForCausalLM": "modeling_mor.MoRLlamaForCausalLM"
            },
            "architectures": ["MoRLlamaForCausalLM"],
            "custom_model": True,
            "mor_enabled": True
        })
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("  ✅ Updated config.json for HF compatibility")
    
    # Create tokenizer config if missing
    tokenizer_config_path = upload_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        tokenizer_config = {
            "tokenizer_class": "LlamaTokenizer",
            "bos_token": "<s>",
            "eos_token": "</s>", 
            "unk_token": "<unk>",
            "model_max_length": 1024
        }
        
        with open(tokenizer_config_path, 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        
        print("  ✅ Created tokenizer_config.json")
    
    # Copy supporting files from this directory
    support_files = [
        "README.md",
        "modeling_mor.py",
        "requirements.txt"
    ]
    
    for file_name in support_files:
        source_file = Path(__file__).parent / file_name
        if source_file.exists():
            shutil.copy2(source_file, upload_dir / file_name)
            print(f"  ✅ Copied {file_name}")
    
    print(f"✅ Upload directory prepared with all files")

def upload_to_huggingface(repo_name: str, upload_dir: Path, token: str):
    """Upload the model to Hugging Face Hub."""
    
    try:
        from huggingface_hub import create_repo, upload_folder
    except ImportError:
        print("❌ huggingface_hub not installed. Installing...")
        if not install_requirements():
            return False
        from huggingface_hub import create_repo, upload_folder
    
    print(f"🚀 Uploading to Hugging Face: {repo_name}")
    
    try:
        # Create repository
        print("  📝 Creating repository...")
        create_repo(
            repo_id=repo_name,
            token=token,
            exist_ok=True,
            repo_type="model"
        )
        print(f"  ✅ Repository {repo_name} created/verified")
        
        # Upload files
        print("  📤 Uploading files...")
        upload_folder(
            folder_path=str(upload_dir),
            repo_id=repo_name,
            token=token,
            commit_message="Upload MoR (Mixture-of-Recursions) model",
            repo_type="model"
        )
        
        print(f"  ✅ Upload completed successfully!")
        print(f"  🔗 Model available at: https://huggingface.co/{repo_name}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Upload failed: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload MoR model to Hugging Face")
    parser.add_argument("--username", required=True, help="Your HuggingFace username")
    parser.add_argument("--model-name", required=True, help="Model repository name")
    parser.add_argument("--model-dir", default="../trained_model", help="Path to trained model")
    
    args = parser.parse_args()
    
    # Construct repo name
    repo_name = f"{args.username}/{args.model_name}"
    
    print("🎯 MoR Model Upload to Hugging Face")
    print("=" * 50)
    print(f"Repository: {repo_name}")
    
    # Validate model directory
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        print("Please ensure your trained model exists.")
        return 1
    
    # Check for required files
    required_files = ["pytorch_model.bin", "config.json"]
    missing_files = [f for f in required_files if not (model_dir / f).exists()]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure your model training completed successfully.")
        return 1
    
    print(f"📁 Model directory: {model_dir}")
    
    # Prepare upload
    upload_dir = Path(__file__).parent / "temp_upload"
    
    try:
        # Get and validate HuggingFace token
        print("🔑 Getting HuggingFace token...")
        hf_token = get_hf_token()
        
        if not test_token(hf_token):
            print("❌ Invalid token. Please get a valid token from https://huggingface.co/settings/tokens")
            return 1
        
        prepare_upload_directory(model_dir, upload_dir, repo_name)
        
        # Upload to HuggingFace
        success = upload_to_huggingface(repo_name, upload_dir, hf_token)
        
        if success:
            print(f"\n🎉 Success! Your MoR model is now available!")
            print(f"🔗 https://huggingface.co/{repo_name}")
            print(f"\n💡 Usage:")
            print(f"from transformers import AutoTokenizer, AutoModelForCausalLM")
            print(f"tokenizer = AutoTokenizer.from_pretrained('{repo_name}')")
            print(f"model = AutoModelForCausalLM.from_pretrained('{repo_name}')")
            return 0
        else:
            print("\n❌ Upload failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Cleanup
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
            print("🧹 Cleaned up temporary files")

if __name__ == "__main__":
    exit(main()) 