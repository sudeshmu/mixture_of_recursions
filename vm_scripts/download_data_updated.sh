#!/bin/bash

# Updated Data Download Script for Mixture of Recursions Project
# This script downloads the required datasets

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
    echo -e "${BLUE}[DOWNLOAD]${NC} $1"
}

# Configuration - Updated for actual project location
PROJECT_DIR="/var/home/sudeshmu/mixture_of_recursions"

print_header "📚 Starting Dataset Download for Mixture of Recursions"

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

# Check available disk space
print_status "Checking available disk space..."
df -h /var/home

# Download FineWeb-Edu dataset
print_header "📥 Downloading FineWeb-Edu dataset..."
print_warning "This dataset is large and may take significant time and bandwidth!"

# Check if download script exists
if [ ! -f "lm_dataset/download_scripts/download_fineweb-edu-dedup.sh" ]; then
    print_error "Download script not found!"
    exit 1
fi

# Make download script executable
chmod +x lm_dataset/download_scripts/download_fineweb-edu-dedup.sh

# Run download
print_status "Starting download..."
bash lm_dataset/download_scripts/download_fineweb-edu-dedup.sh

# Download other optional datasets
print_header "📦 Downloading additional evaluation datasets..."

# Download FineWeb test set if script exists
if [ -f "lm_dataset/download_scripts/download_fineweb-test.sh" ]; then
    print_status "Downloading FineWeb test set..."
    chmod +x lm_dataset/download_scripts/download_fineweb-test.sh
    bash lm_dataset/download_scripts/download_fineweb-test.sh
fi

# Download language modeling datasets if script exists
if [ -f "lm_dataset/download_scripts/download_langauge_modeling_datasets.sh" ]; then
    print_status "Downloading language modeling datasets..."
    chmod +x lm_dataset/download_scripts/download_langauge_modeling_datasets.sh
    bash lm_dataset/download_scripts/download_langauge_modeling_datasets.sh
fi

# Download LM evaluation datasets
print_header "🔍 Downloading LM evaluation datasets..."
if [ -f "lm_eval/download_lm_eval_datasets.py" ]; then
    print_status "Running LM eval dataset download..."
    python lm_eval/download_lm_eval_datasets.py
fi

print_header "✅ Dataset download completed!"
print_status "📊 Checking final disk usage..."
df -h /var/home

print_status "🎯 Data download summary:"
print_status "  📁 Main datasets: $(du -sh hf_datasets 2>/dev/null || echo 'Not found')"
print_status "  🗃️  Cache: $(du -sh hf_cache 2>/dev/null || echo 'Not found')"
print_status "  📋 Total project size: $(du -sh . 2>/dev/null || echo 'Not found')"

print_header "🚀 Ready for training!"
print_status "To start training, run: bash /root/sudeshmu/run_training_updated.sh [config_name] [launcher] [wandb_mode]" 