#!/bin/bash

# Updated Training Script for Mixture of Recursions Project
# This script runs training on the VM with GPU support

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "${BLUE}[TRAINING]${NC} $1"
}

# Configuration - Updated for actual project location
PROJECT_DIR="/var/home/sudeshmu/mixture_of_recursions"
CONFIG_NAME="${1:-250720_pretrain_smollm-360m_rec2_middle_cycle_random_lr3e-3}"  # Use first argument or default
LAUNCHER="${2:-accelerate}"  # Use second argument or default to 'accelerate'
WANDB_MODE="${3:-offline}"   # Use third argument or default to 'offline'

print_header "🚀 Starting Mixture of Recursions Training"
print_status "📁 Project directory: $PROJECT_DIR"
print_status "⚙️  Configuration: $CONFIG_NAME"
print_status "🚀 Launcher: $LAUNCHER"
print_status "📊 WandB mode: $WANDB_MODE"

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    print_error "Project directory $PROJECT_DIR does not exist!"
    print_error "Please run the setup script first"
    exit 1
fi

# Navigate to project directory
cd $PROJECT_DIR

# Activate virtual environment
print_status "Activating virtual environment..."
source mor_venv/bin/activate

# Check GPU availability
print_status "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Set environment variables
export HYDRA_FULL_ERROR=1
export WANDB_MODE=$WANDB_MODE

# Get available GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
print_status "Available GPUs: $GPU_COUNT"

if [ "$GPU_COUNT" -eq 0 ]; then
    print_error "No GPUs detected! This training requires GPU support."
    exit 1
fi

# Set GPU indices based on available GPUs
if [ "$GPU_COUNT" -eq 1 ]; then
    GPU_INDICES="0"
    CUDA_VISIBLE_DEVICES="0"
elif [ "$GPU_COUNT" -eq 2 ]; then
    GPU_INDICES="0,1"
    CUDA_VISIBLE_DEVICES="0,1"
elif [ "$GPU_COUNT" -ge 4 ]; then
    GPU_INDICES="0,1,2,3"
    CUDA_VISIBLE_DEVICES="0,1,2,3"
else
    GPU_INDICES="0,1,2"
    CUDA_VISIBLE_DEVICES="0,1,2"
fi

print_status "Using GPUs: $GPU_INDICES"

# Check if config exists
if [ ! -f "conf/pretrain/${CONFIG_NAME}.yaml" ]; then
    print_warning "Configuration file conf/pretrain/${CONFIG_NAME}.yaml not found!"
    print_status "Available configurations:"
    ls conf/pretrain/*.yaml | head -5
    
    # Try to use the first available config
    FIRST_CONFIG=$(ls conf/pretrain/*.yaml | head -1 | xargs basename | sed 's/.yaml$//')
    print_warning "Using $FIRST_CONFIG as fallback..."
    CONFIG_NAME="$FIRST_CONFIG"
fi

# Create logs directory
mkdir -p logs

# Check if DeepSpeed is available
DEEPSPEED_AVAILABLE=false
if python -c "import deepspeed" 2>/dev/null; then
    DEEPSPEED_AVAILABLE=true
    print_status "DeepSpeed is available"
else
    print_warning "DeepSpeed not available, will use Accelerate only"
    if [ "$LAUNCHER" == "deepspeed" ]; then
        print_warning "Switching launcher from deepspeed to accelerate"
        LAUNCHER="accelerate"
    fi
fi

# Run training based on launcher choice
if [ "$LAUNCHER" == "deepspeed" ] && [ "$DEEPSPEED_AVAILABLE" == "true" ]; then
    print_header "🔥 Starting training with DeepSpeed..."
    
    TRAINING_CMD="deepspeed --include localhost:$GPU_INDICES --no_local_rank --master_port 25720 pretrain.py --config-name $CONFIG_NAME"
    
    print_status "Command: $TRAINING_CMD"
    
    # Log to file and display
    exec > >(tee -a logs/training_$(date +%Y%m%d_%H%M%S).log) 2>&1
    
    eval $TRAINING_CMD

else
    print_header "⚡ Starting training with Accelerate..."
    
    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    
    TRAINING_CMD="accelerate launch --config_file acc_configs/default_config.yaml --main_process_port 25720 pretrain.py --config-name $CONFIG_NAME"
    
    print_status "Command: $TRAINING_CMD"
    
    # Log to file and display
    exec > >(tee -a logs/training_$(date +%Y%m%d_%H%M%S).log) 2>&1
    
    eval $TRAINING_CMD

fi

print_header "✅ Training completed!"
print_status "📋 Check logs directory for training logs"
print_status "📊 Check results directory for outputs" 