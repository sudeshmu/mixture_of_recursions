#!/usr/bin/env python3
"""
Helper script to get a new HuggingFace token and upload the MoR model.

This script will:
1. Guide you to get a new HuggingFace token
2. Test the token
3. Upload your model

Usage: python get_token_and_upload.py --username your-username --model-name mixture-of-recursions-360m
"""

import argparse
import subprocess
import sys
from pathlib import Path

def print_token_instructions():
    """Print instructions for getting a new HuggingFace token."""
    
    print("\n🔑 How to get a new HuggingFace token:")
    print("=" * 50)
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Click 'New token'")
    print("3. Give it a name (e.g., 'MoR Model Upload')")
    print("4. Select 'Write' permission")
    print("5. Click 'Generate token'")
    print("6. Copy the token (starts with 'hf_')")
    print("=" * 50)
    print("\n⚠️  Important: The token will only be shown once!")
    print("📝 Make sure to copy it completely")
    print("")

def main():
    parser = argparse.ArgumentParser(description="Get new token and upload MoR model")
    parser.add_argument("--username", required=True, help="Your HuggingFace username")
    parser.add_argument("--model-name", required=True, help="Model repository name")
    
    args = parser.parse_args()
    
    print("🎯 MoR Model Upload Helper")
    print("=" * 40)
    print(f"Repository: {args.username}/{args.model_name}")
    
    # Check if upload script exists
    upload_script = Path(__file__).parent / "upload_mor_model.py"
    if not upload_script.exists():
        print("❌ upload_mor_model.py not found!")
        return 1
    
    # Print token instructions
    print_token_instructions()
    
    # Ask user to get token
    input("📋 Press Enter after you've got your new token...")
    
    print("\n🚀 Starting upload process...")
    print("   You'll be prompted for your token securely")
    print("   (Your token won't be stored permanently)")
    
    # Run upload script
    cmd = [sys.executable, str(upload_script), "--username", args.username, "--model-name", args.model_name]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n🎉 Upload completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Upload failed with exit code {e.returncode}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Upload cancelled by user")
        return 1

if __name__ == "__main__":
    exit(main()) 