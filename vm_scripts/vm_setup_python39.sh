#!/bin/bash

# VM Setup Script for Mixture of Recursions Project using Python 3.9
# Uses /var/home/sudeshmu and existing Python 3.9

set -e  # Exit on any error

echo "🚀 Starting Mixture of Recursions setup with Python 3.9..."

# Configuration - using /var/home which has lots of space
PROJECT_DIR="/var/home/sudeshmu"
REPO_URL="https://github.com/raymin0223/mixture_of_recursions.git"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create project directory in /var/home where we have space
print_status "Creating project directory: $PROJECT_DIR"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Check if repository already exists
if [ -d "mixture_of_recursions" ]; then
    print_warning "Repository already exists, pulling latest changes..."
    cd mixture_of_recursions
    git pull
    cd ..
else
    print_status "Cloning repository..."
    git clone $REPO_URL
fi

cd mixture_of_recursions

# Create virtual environment using python3 (3.9)
print_status "Creating Python virtual environment with Python 3.9..."
python3 -m venv mor_venv

# Activate virtual environment
print_status "Activating virtual environment..."
source mor_venv/bin/activate

# Upgrade pip and setuptools
print_status "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Try to install PyTorch with CUDA support first (may need to use older versions)
print_status "Installing PyTorch with CUDA support (Python 3.9 compatible)..."
pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Try to install flash-attention (may not work with Python 3.9, but let's try)
print_status "Attempting to install flash-attention..."
pip install flash-attn --no-build-isolation || {
    print_warning "Flash-attention installation failed, continuing without it..."
    print_warning "Some models may be slower without flash-attention"
}

# Install transformers with compatible version
print_status "Installing transformers..."
pip install transformers==4.44.0

# Install basic requirements that are likely to work with Python 3.9
print_status "Installing basic requirements..."
pip install hydra-core omegaconf wandb tensorboard datasets accelerate peft zstandard boto3 smart_open bitsandbytes || {
    print_warning "Some packages failed to install, continuing..."
}

# Try deepspeed (may not work without development tools)
print_status "Attempting to install deepspeed..."
pip install deepspeed || {
    print_warning "DeepSpeed installation failed, you may need to use accelerate only"
}

# Install evaluation packages
print_status "Installing evaluation packages..."
pip install sacrebleu evaluate nltk rouge_score matplotlib more_itertools || {
    print_warning "Some evaluation packages failed to install"
}

# Install lm-evaluation-harness
print_status "Installing lm-evaluation-harness..."
if [ -d "lm-evaluation-harness" ]; then
    print_warning "lm-evaluation-harness already exists, pulling latest changes..."
    cd lm-evaluation-harness
    git pull
    cd ..
else
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
fi
cd lm-evaluation-harness
pip install -e . || {
    print_warning "LM evaluation harness installation failed, some evaluations may not work"
}
cd ..

# Create data directories
print_status "Creating data directories..."
mkdir -p data/mixture-of-recursions/hf_cache
mkdir -p data/mixture-of-recursions/hf_datasets
mkdir -p data/mixture-of-recursions/hf_models
mkdir -p data/mixture-of-recursions/results

# Create symbolic links
print_status "Creating symbolic links..."
ln -sf $(pwd)/data/mixture-of-recursions/* $(pwd)/ 2>/dev/null || true

# Create environment activation script
print_status "Creating environment activation script..."
cat > activate_mor.sh << 'EOF'
#!/bin/bash
cd /var/home/sudeshmu/mixture_of_recursions
source mor_venv/bin/activate
echo "🎯 Mixture of Recursions environment activated!"
echo "📁 Current directory: $(pwd)"
echo "🐍 Python version: $(python --version)"
echo "🔥 PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "⚡ CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
if python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null | grep -q True; then
    echo "🎮 GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "🎮 GPU names:"
    python -c 'import torch; [print(f"  GPU {i}: {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]'
fi
EOF
chmod +x activate_mor.sh

# Create symbolic link to make it accessible from /root/sudeshmu
print_status "Creating convenience link..."
mkdir -p /root/sudeshmu
ln -sf $PROJECT_DIR/mixture_of_recursions /root/sudeshmu/ 2>/dev/null || true

# Create a simple test script
print_status "Creating test script..."
cat > test_setup.py << 'EOF'
import sys
print("🧪 Testing setup...")
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    print("❌ PyTorch not installed properly")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError:
    print("❌ Transformers not installed properly")

try:
    import flash_attn
    print(f"Flash Attention version: {flash_attn.__version__}")
except ImportError:
    print("⚠️  Flash Attention not available (this is okay)")

print("✅ Setup test completed!")
EOF

# Run test
print_status "Running setup test..."
python test_setup.py

print_status "✅ Setup completed with Python 3.9!"
print_warning "Note: Using Python 3.9 instead of 3.12. Some features may have compatibility issues."
print_warning "Flash-attention may not be available, which could impact training speed."
print_status "🎯 To activate the environment, run: source /var/home/sudeshmu/mixture_of_recursions/activate_mor.sh"
print_status "🔗 Convenience link created: /root/sudeshmu/mixture_of_recursions -> /var/home/sudeshmu/mixture_of_recursions"
print_status "📚 To download datasets, navigate to the project directory and run: bash download_data.sh"
print_status "🚀 Ready to start training (with Python 3.9 limitations)!" 