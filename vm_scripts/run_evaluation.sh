#!/bin/bash

# Evaluation Script for Mixture of Recursions Model
# This script runs evaluation on the trained model

set -e

# Configuration
PROJECT_DIR="/var/home/sudeshmu/mixture_of_recursions"
CONFIG_NAME="250720_pretrain_smollm-360m_rec2_middle_cycle_random_lr3e-3"
TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa"
EVAL_OUTPUT_DIR="$PROJECT_DIR/results/eval"

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
    echo -e "${BLUE}[EVALUATION]${NC} $1"
}

# Main evaluation function
main() {
    print_header "🧪 Starting Mixture of Recursions Model Evaluation"
    
    print_status "📁 Project directory: $PROJECT_DIR"
    print_status "⚙️  Configuration: $CONFIG_NAME"
    print_status "📊 Tasks: $TASKS"
    
    # Check if project directory exists
    if [[ ! -d "$PROJECT_DIR" ]]; then
        print_error "Project directory not found: $PROJECT_DIR"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
    
    print_status "Activating virtual environment..."
    source mor_venv/bin/activate
    
    # Check if model exists
    MODEL_PATH="$PROJECT_DIR/results/pretrain/$CONFIG_NAME"
    if [[ ! -d "$MODEL_PATH" ]]; then
        print_error "Trained model not found: $MODEL_PATH"
        exit 1
    fi
    
    print_status "Checking GPU availability..."
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
    
    # Create evaluation output directory
    mkdir -p "$EVAL_OUTPUT_DIR"
    
    print_header "⚡ Starting evaluation..."
    
    # Run evaluation
    print_status "Command: python eval_fewshot.py --config-name $CONFIG_NAME"
    
    python eval_fewshot.py \
        --config-name "$CONFIG_NAME" \
        hydra.run.dir="$EVAL_OUTPUT_DIR" \
        || {
        print_error "Evaluation failed!"
        exit 1
    }
    
    print_header "✅ Evaluation completed!"
    print_status "📋 Check $EVAL_OUTPUT_DIR for evaluation results"
    print_status "📊 Check wandb logs for detailed metrics"
}

# Run main function
main "$@" 