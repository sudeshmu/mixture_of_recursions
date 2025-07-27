#!/bin/bash

# Setup and Upload MoR Model to Hugging Face
# Usage: ./setup_and_upload.sh your-username model-name

set -e  # Exit on any error

echo "🚀 MoR Model Hugging Face Upload Setup"
echo "======================================"

# Check if arguments are provided
if [ $# -lt 2 ]; then
    echo "❌ Usage: $0 <username> <model-name>"
    echo "   Example: $0 johndoe mixture-of-recursions-360m"
    exit 1
fi

USERNAME="$1"
MODEL_NAME="$2"
REPO_NAME="${USERNAME}/${MODEL_NAME}"

echo "📝 Repository: ${REPO_NAME}"

# Check if we're in the right directory
if [ ! -f "upload_mor_model.py" ]; then
    echo "❌ Please run this script from the huggingface_upload directory"
    exit 1
fi

# Check if trained model exists
if [ ! -d "../trained_model" ]; then
    echo "❌ Trained model directory not found: ../trained_model"
    echo "   Please ensure your model training completed successfully"
    exit 1
fi

# Check for required model files
if [ ! -f "../trained_model/pytorch_model.bin" ]; then
    echo "❌ pytorch_model.bin not found in trained_model directory"
    exit 1
fi

if [ ! -f "../trained_model/config.json" ]; then
    echo "❌ config.json not found in trained_model directory"
    exit 1
fi

echo "✅ Model files found"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -q huggingface_hub transformers torch

# Make upload script executable
chmod +x upload_mor_model.py

# Run the upload
echo "🚀 Starting upload process..."
python upload_mor_model.py --username "${USERNAME}" --model-name "${MODEL_NAME}"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Upload completed successfully!"
    echo "🔗 Your model is available at: https://huggingface.co/${REPO_NAME}"
    echo ""
    echo "💡 To use your model:"
    echo "   from transformers import AutoTokenizer, AutoModelForCausalLM"
    echo "   tokenizer = AutoTokenizer.from_pretrained('${REPO_NAME}')"
    echo "   model = AutoModelForCausalLM.from_pretrained('${REPO_NAME}')"
    echo ""
    echo "🧪 To test your model:"
    echo "   python test_model.py --model-name '${REPO_NAME}'"
else
    echo "❌ Upload failed!"
    exit 1
fi 