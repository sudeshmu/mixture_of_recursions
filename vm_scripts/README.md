# Mixture of Recursions - VM Scripts

This directory contains all the scripts needed to deploy, setup, and run the Mixture of Recursions project on a remote VM with GPU support.

## 📁 Scripts Overview

| Script | Purpose | Description |
|--------|---------|-------------|
| `deploy_to_vm.sh` | Deployment | Copies all scripts to the VM |
| `vm_setup_python39.sh` | VM Setup | Sets up Python environment and dependencies |
| `run_training_updated.sh` | Training | Starts model training |
| `download_data_updated.sh` | Data | Downloads required datasets |
| `run_evaluation.sh` | Evaluation | Runs model evaluation |
| `local_inference.py` | Inference | Local inference script |
| `fix_cache_utils.py` | Fix | Fixes cache utilities compatibility |
| `fix_flash_attention.py` | Fix | Fixes Flash Attention compatibility |

## 🚀 Quick Start

### Prerequisites

- SSH access to a VM with NVIDIA GPUs
- SSH key-based authentication configured
- At least 100GB free disk space on VM

### Step 1: Set Environment Variables

```bash
# Set your VM connection details
export VM_HOST="root@your-vm-hostname"
export REMOTE_DIR="/var/home/username"  # Optional, has default
```

### Step 2: Deploy Scripts to VM

```bash
cd mixture_of_recursions/vm_scripts
bash deploy_to_vm.sh
```

### Step 3: Setup VM Environment

```bash
# SSH into your VM
ssh $VM_HOST

# Navigate to the project directory
cd $REMOTE_DIR

# Run the setup script
bash vm_setup_python39.sh
```

### Step 4: Download Datasets

```bash
bash download_data_updated.sh
```

### Step 5: Start Training

```bash
# Start training in background
nohup bash run_training_updated.sh > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Monitor progress
tail -f training_*.log
```

### Step 6: Run Evaluation (After Training)

```bash
bash run_evaluation.sh
```

## 📊 Local Inference

After training completes, copy the model to your local machine and run inference:

```bash
# Copy model from VM to local (run from local machine)
rsync -avz $VM_HOST:/var/home/username/mixture_of_recursions/results/pretrain/250720_pretrain_smollm-360m_rec2_middle_cycle_random_lr3e-3/ ./mixture_of_recursions/trained_model/

# Run local inference
cd mixture_of_recursions/vm_scripts
python local_inference.py --interactive
```

## 🛠️ Advanced Usage

### Custom Configuration

To use a different training configuration:

```bash
# Edit the config name in run_training_updated.sh
vim run_training_updated.sh
# Change CONFIG_NAME variable

# Or pass it as environment variable
CONFIG_NAME="your-config-name" bash run_training_updated.sh
```

### Monitoring Training

```bash
# Check GPU utilization
nvidia-smi -l 1

# Monitor training logs
tail -f training_*.log

# Check training progress
watch -n 30 'ls -la results/pretrain/*/checkpoint-*'
```

### Troubleshooting

#### Common Issues

1. **CUDA Out of Memory**
   - Reduce `per_device_train_batch_size` in config
   - Reduce `max_length` in config

2. **SSH Connection Issues**
   - Verify VM_HOST environment variable
   - Check SSH key authentication
   - Ensure VM is accessible

3. **Disk Space Issues**
   - The script uses `/var/home` for more space
   - Check available space: `df -h`

4. **Import Errors**
   - Run the fix scripts: `python fix_*.py`
   - Check Python environment activation

#### Debug Mode

```bash
# Run scripts with debug output
bash -x vm_setup_python39.sh
bash -x run_training_updated.sh
```

## 📁 Directory Structure (After Setup)

```
/var/home/username/mixture_of_recursions/
├── mor_venv/                      # Python virtual environment
├── data/                          # Downloaded datasets
├── results/
│   ├── pretrain/                  # Training outputs
│   └── eval/                      # Evaluation outputs
├── logs/                          # Training logs
├── wandb/                         # WandB offline runs
└── [project files...]            # Model code and configs
```

## 🔧 Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `VM_HOST` | Required | SSH connection string (user@hostname) |
| `REMOTE_DIR` | `/home/user/mixture_of_recursions` | Remote directory path |
| `CONFIG_NAME` | `250720_pretrain_smollm-360m_rec2_middle_cycle_random_lr3e-3` | Training configuration |
| `WANDB_MODE` | `offline` | WandB logging mode |

## 📝 Configuration Files

Key configuration files you might want to modify:

- `conf/pretrain/*.yaml` - Training configurations
- `acc_configs/default_config.yaml` - Accelerate configuration
- `ds_configs/*.config` - DeepSpeed configurations (optional)

## 🎯 Performance Tips

1. **GPU Memory Optimization**
   - Use appropriate batch sizes for your GPU memory
   - Enable gradient checkpointing for larger models
   - Consider using DeepSpeed for memory efficiency

2. **Training Speed**
   - Use all available GPUs
   - Optimize data loading with multiple workers
   - Use mixed precision training

3. **Storage Optimization**
   - Use fast SSD storage for datasets
   - Store checkpoints on high-capacity storage
   - Clean up old checkpoints periodically

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the log files for error messages
3. Ensure all dependencies are properly installed
4. Verify GPU compatibility and CUDA version

## 🔗 Related Files

- `../README.md` - Main project README
- `../requirements.txt` - Python dependencies
- `../conf/` - Configuration files
- `SUCCESS_REPORT.md` - Detailed training success report 