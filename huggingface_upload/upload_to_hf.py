#!/usr/bin/env python3
"""
Script to upload the MoR (Mixture-of-Recursions) model to Hugging Face Hub.

Usage:
    python upload_to_hf.py --repo-name "your-username/mixture-of-recursions-360m" --token "your_hf_token"
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def setup_huggingface_hub():
    """Install and import required packages for Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
        from transformers import AutoTokenizer, AutoConfig
        return HfApi, create_repo, upload_folder, AutoTokenizer, AutoConfig
    except ImportError:
        print("Installing required packages...")
        os.system("pip install huggingface_hub transformers")
        from huggingface_hub import HfApi, create_repo, upload_folder
        from transformers import AutoTokenizer, AutoConfig
        return HfApi, create_repo, upload_folder, AutoTokenizer, AutoConfig

def prepare_model_files(source_model_dir: Path, target_dir: Path):
    """
    Prepare model files for Hugging Face upload.
    
    Args:
        source_model_dir: Path to the trained model directory
        target_dir: Path to the target upload directory
    """
    
    print(f"📁 Preparing model files from {source_model_dir} to {target_dir}")
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy essential model files
    essential_files = [
        "pytorch_model.bin",
        "config.json", 
        "generation_config.json"
    ]
    
    for file_name in essential_files:
        source_file = source_model_dir / file_name
        target_file = target_dir / file_name
        
        if source_file.exists():
            print(f"  ✅ Copying {file_name}")
            shutil.copy2(source_file, target_file)
        else:
            print(f"  ⚠️  Warning: {file_name} not found in source directory")
    
    # Update config.json to include MoR-specific information
    config_file = target_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Add MoR-specific fields
        config.update({
            "model_type": "mor_llama",
            "auto_map": {
                "AutoConfig": "modeling_mor.MoRConfig",
                "AutoModelForCausalLM": "modeling_mor.MoRLlamaForCausalLM"
            },
            "custom_model": True,
            "mor_enabled": True,
            "architectures": ["MoRLlamaForCausalLM"]
        })
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("  ✅ Updated config.json with MoR configuration")

def create_tokenizer_files(target_dir: Path, source_model_dir: Path):
    """
    Create or copy tokenizer files.
    
    Args:
        target_dir: Target directory for upload
        source_model_dir: Source model directory
    """
    
    print("🔤 Setting up tokenizer files...")
    
    # Check if tokenizer files exist in source
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json", 
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt"
    ]
    
    found_tokenizer = False
    for file_name in tokenizer_files:
        source_file = source_model_dir / file_name
        if source_file.exists():
            target_file = target_dir / file_name
            shutil.copy2(source_file, target_file)
            print(f"  ✅ Copied {file_name}")
            found_tokenizer = True
    
    if not found_tokenizer:
        print("  ⚠️  No tokenizer files found. Creating basic tokenizer config...")
        
        # Create a basic tokenizer config pointing to a compatible tokenizer
        tokenizer_config = {
            "tokenizer_class": "LlamaTokenizer",
            "auto_map": {
                "AutoTokenizer": ["transformers", "LlamaTokenizer"]
            },
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "model_max_length": 1024,
        }
        
        with open(target_dir / "tokenizer_config.json", 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        
        print("  ✅ Created basic tokenizer_config.json")

def upload_model(
    repo_name: str,
    upload_dir: Path,
    token: str,
    private: bool = False,
    commit_message: str = "Upload MoR model"
):
    """
    Upload the model to Hugging Face Hub.
    
    Args:
        repo_name: Repository name (e.g., "username/model-name")
        upload_dir: Directory containing files to upload
        token: Hugging Face token
        private: Whether to create a private repository
        commit_message: Commit message for the upload
    """
    
    HfApi, create_repo, upload_folder, AutoTokenizer, AutoConfig = setup_huggingface_hub()
    
    print(f"🚀 Uploading model to {repo_name}")
    
    # Initialize API
    api = HfApi(token=token)
    
    try:
        # Create repository
        print("  📝 Creating repository...")
        create_repo(
            repo_id=repo_name,
            token=token,
            private=private,
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
            commit_message=commit_message,
            repo_type="model"
        )
        
        print(f"  ✅ Successfully uploaded to https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"  ❌ Upload failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Upload MoR model to Hugging Face Hub")
    parser.add_argument(
        "--repo-name", 
        required=True,
        help="Repository name on Hugging Face (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Hugging Face authentication token"
    )
    parser.add_argument(
        "--model-dir",
        default="../trained_model",
        help="Path to the trained model directory (default: ../trained_model)"
    )
    parser.add_argument(
        "--upload-dir", 
        default="./hf_upload_temp",
        help="Temporary directory for preparing upload files (default: ./hf_upload_temp)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    parser.add_argument(
        "--commit-message",
        default="Upload MoR (Mixture-of-Recursions) model",
        help="Commit message for the upload"
    )
    
    args = parser.parse_args()
    
    # Convert paths
    model_dir = Path(args.model_dir).resolve()
    upload_dir = Path(args.upload_dir).resolve()
    
    # Validate model directory
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        print("Please ensure the trained model exists in the specified directory.")
        sys.exit(1)
    
    print(f"🎯 Starting MoR model upload process...")
    print(f"   Source model: {model_dir}")
    print(f"   Target repo: {args.repo_name}")
    print(f"   Upload dir: {upload_dir}")
    
    try:
        # Prepare upload directory
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        
        # Copy model files
        prepare_model_files(model_dir, upload_dir)
        
        # Setup tokenizer
        create_tokenizer_files(upload_dir, model_dir)
        
        # Copy README and model code
        readme_source = Path(__file__).parent / "README.md"
        if readme_source.exists():
            shutil.copy2(readme_source, upload_dir / "README.md")
            print("  ✅ Copied README.md")
        
        model_code_source = Path(__file__).parent / "modeling_mor.py"
        if model_code_source.exists():
            shutil.copy2(model_code_source, upload_dir / "modeling_mor.py")
            print("  ✅ Copied modeling_mor.py")
        
        # Upload to Hugging Face
        upload_model(
            repo_name=args.repo_name,
            upload_dir=upload_dir,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message
        )
        
        print(f"\n🎉 Successfully uploaded MoR model!")
        print(f"🔗 Model available at: https://huggingface.co/{args.repo_name}")
        print(f"💡 You can now use it with:")
        print(f'   from transformers import AutoTokenizer, AutoModelForCausalLM')
        print(f'   model = AutoModelForCausalLM.from_pretrained("{args.repo_name}")')
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        sys.exit(1)
        
    finally:
        # Cleanup temporary directory
        if upload_dir.exists() and upload_dir.name == "hf_upload_temp":
            shutil.rmtree(upload_dir)
            print("🧹 Cleaned up temporary files")

if __name__ == "__main__":
    main() 