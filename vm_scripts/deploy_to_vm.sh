#!/bin/bash

# Deployment Script for Mixture of Recursions VM Setup
# This script copies all necessary files to the VM

set -e

# Configuration - Use environment variables with defaults
VM_HOST="${VM_HOST:-user@your-vm-hostname}"
REMOTE_DIR="${REMOTE_DIR:-/home/user/mixture_of_recursions}"

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
    echo -e "${BLUE}[DEPLOY]${NC} $1"
}

# Check requirements
check_requirements() {
    print_header "🔍 Checking requirements..."
    
    # Check if VM_HOST is set properly
    if [[ "$VM_HOST" == "user@your-vm-hostname" ]]; then
        print_error "VM_HOST environment variable not set!"
        print_error "Please set it like: export VM_HOST='root@your-vm-hostname'"
        exit 1
    fi
    
    # Check SSH connectivity
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$VM_HOST" exit 2>/dev/null; then
        print_error "Cannot connect to VM: $VM_HOST"
        print_error "Please ensure:"
        print_error "  1. VM is accessible"
        print_error "  2. SSH key is properly configured"
        print_error "  3. VM_HOST environment variable is correct"
        exit 1
    fi
    
    print_status "✅ SSH connectivity verified"
}

# Copy scripts to VM
deploy_scripts() {
    print_header "📦 Deploying scripts to VM..."
    
    print_status "Creating remote directory: $REMOTE_DIR"
    ssh "$VM_HOST" "mkdir -p $REMOTE_DIR"
    
    print_status "Copying VM setup script..."
    scp vm_setup_python39.sh "$VM_HOST:$REMOTE_DIR/"
    
    print_status "Copying training scripts..."
    scp run_training_updated.sh "$VM_HOST:$REMOTE_DIR/"
    
    print_status "Copying data download script..."
    scp download_data_updated.sh "$VM_HOST:$REMOTE_DIR/"
    
    print_status "Copying evaluation script..."
    scp run_evaluation.sh "$VM_HOST:$REMOTE_DIR/"
    
    print_status "Copying fix scripts..."
    scp fix_cache_utils.py "$VM_HOST:$REMOTE_DIR/"
    scp fix_flash_attention.py "$VM_HOST:$REMOTE_DIR/"
    
    print_status "Making scripts executable..."
    ssh "$VM_HOST" "chmod +x $REMOTE_DIR/*.sh"
    
    print_status "✅ All scripts deployed successfully!"
}

# Show usage instructions
show_usage() {
    print_header "📋 Next Steps:"
    echo ""
    echo "1. Connect to your VM:"
    echo "   ssh $VM_HOST"
    echo ""
    echo "2. Run the setup script:"
    echo "   cd $REMOTE_DIR"
    echo "   bash vm_setup_python39.sh"
    echo ""
    echo "3. Download datasets:"
    echo "   bash download_data_updated.sh"
    echo ""
    echo "4. Start training:"
    echo "   bash run_training_updated.sh"
    echo ""
    echo "5. Run evaluation (after training):"
    echo "   bash run_evaluation.sh"
    echo ""
}

# Main function
main() {
    print_header "🚀 Mixture of Recursions VM Deployment"
    print_status "Target VM: $VM_HOST"
    print_status "Remote directory: $REMOTE_DIR"
    
    check_requirements
    deploy_scripts
    show_usage
    
    print_header "✅ Deployment completed!"
}

# Show help if requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Mixture of Recursions VM Deployment Script"
    echo ""
    echo "Usage:"
    echo "  export VM_HOST='root@your-vm-hostname'"
    echo "  export REMOTE_DIR='/path/to/remote/directory'  # optional"
    echo "  bash deploy_to_vm.sh"
    echo ""
    echo "Environment Variables:"
    echo "  VM_HOST      - SSH connection string (required)"
    echo "  REMOTE_DIR   - Remote directory path (optional, default: /home/user/mixture_of_recursions)"
    echo ""
    exit 0
fi

# Run main function
main "$@" 