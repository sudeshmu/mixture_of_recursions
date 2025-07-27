# 🏠 Local Inference Guide - Mixture of Recursions

## 🎯 Overview

This guide covers running inference with your locally trained **Mixture of Recursions** model using the files in your `trained_model/` directory.

## 📋 Prerequisites

### ✅ **Required Files**
Your `trained_model/` directory should contain:
```
trained_model/
├── pytorch_model.bin (499MB)    # Main model weights
├── config.json                  # Model configuration
└── generation_config.json       # Generation parameters
```

### 📦 **Dependencies**
```bash
# Core requirements
pip install torch transformers accelerate omegaconf hydra-core

# Or install all dependencies
pip install -r requirements.txt
```

### 🐍 **Environment Setup**
```bash
# Option 1: Use existing virtual environment
source mor_env/bin/activate

# Option 2: Create new environment
python3 -m venv mor_env
source mor_env/bin/activate
pip install -r requirements.txt
```

## 🚀 Usage Methods

### 1. **🎮 Interactive Chat Mode** (Recommended)
```bash
python vm_scripts/local_inference.py --interactive
```

**Features:**
- Real-time conversation
- Context memory (last 3 exchanges)
- Commands: `clear` (reset), `quit` (exit)
- Smart response parsing

### 2. **📝 Single Prompt Generation**
```bash
# Basic usage
python vm_scripts/local_inference.py --prompt "Your prompt here"

# With custom parameters
python vm_scripts/local_inference.py \
    --prompt "Explain quantum computing" \
    --max_length 150 \
    --temperature 0.8 \
    --top_p 0.95
```

### 3. **🎯 Demo Mode**
```bash
python vm_scripts/local_inference.py
```
Tests with 4 sample prompts automatically.

### 4. **🧪 Testing & Validation**
```bash
# Basic functionality test
python test_inference.py

# Comprehensive 50-test suite
python comprehensive_inference_test.py
```

## ⚙️ Configuration Options

### **Generation Parameters**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_length` | 100 | Maximum generation length |
| `--temperature` | 0.7 | Sampling temperature (0.1-2.0) |
| `--top_p` | 0.9 | Top-p nucleus sampling |
| `--device` | auto | Device (auto/cpu/cuda) |

### **Example Commands**
```bash
# Conservative generation
python vm_scripts/local_inference.py \
    --prompt "The future of AI is" \
    --temperature 0.3 \
    --max_length 80

# Creative generation
python vm_scripts/local_inference.py \
    --prompt "Once upon a time" \
    --temperature 1.2 \
    --max_length 200

# Force CPU usage
python vm_scripts/local_inference.py \
    --prompt "Hello world" \
    --device cpu
```

## 🔍 Model Information

### **Architecture Details**
- **Base**: LlamaForCausalLM with MoR enhancements
- **Parameters**: 361,821,120 (~362M)
- **Vocabulary**: 49,152 tokens
- **Context Window**: 1,024 tokens
- **Attention**: 15 heads (5 KV heads - GQA)
- **Hidden Size**: 960
- **Layers**: 32

### **MoR Features**
- **Dynamic Recursion**: Adaptive computation depth per token
- **Efficient Inference**: Optimized resource utilization
- **KV Caching**: Recursion-wise caching strategy
- **Router Mechanisms**: Smart computation routing

## 🧪 Testing Your Setup

### **Quick Test**
```bash
python test_inference.py
```
Expected output: Model loads + generates text for 12 prompts

### **Comprehensive Test**
```bash
python comprehensive_inference_test.py
```
Expected output: 50/50 tests pass in ~84 seconds

### **Performance Benchmark**
- **Load Time**: ~10-15 seconds (CPU)
- **Generation Speed**: ~1.7 seconds per prompt
- **Memory Usage**: ~2-3GB RAM (CPU inference)

## 🔧 Troubleshooting

### **Common Issues**

#### ❌ "No module named 'torch'"
```bash
# Solution: Install dependencies
pip install torch transformers accelerate
```

#### ❌ "Model not found"
```bash
# Check model path
ls -la trained_model/
# Should show pytorch_model.bin (499MB)
```

#### ❌ "CUDA out of memory"
```bash
# Force CPU usage
python vm_scripts/local_inference.py --device cpu
```

#### ❌ "Attention mask warning"
This is normal - the model still generates correctly.

### **Performance Optimization**

#### **For Faster Inference:**
```bash
# Use GPU if available
python vm_scripts/local_inference.py --device cuda

# Lower precision (if CUDA available)
# Model automatically uses bfloat16 on GPU
```

#### **For Lower Memory:**
```bash
# Use CPU with lower precision
python vm_scripts/local_inference.py --device cpu
```

## 📊 Expected Results

### **Successful Setup Indicators:**
✅ Model loads without errors  
✅ Parameters: ~361M  
✅ Vocab size: 49,152  
✅ Generates text for all prompts  
✅ No crashes during inference  

### **Generation Characteristics:**
- Consistent token patterns
- Mix of natural language and specialized tokens
- Stable performance across prompt types
- Appropriate response lengths

## 🎯 Production Usage

### **Integration Example**
```python
#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from vm_scripts.local_inference import load_model_and_tokenizer, generate_text

# Load model once
model, tokenizer = load_model_and_tokenizer("trained_model", "cpu")

# Generate multiple times
prompts = ["Hello", "Explain AI", "Tell a story"]
for prompt in prompts:
    result = generate_text(model, tokenizer, prompt, max_length=50)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}\n")
```

## 📈 Next Steps

1. **✅ Validate Setup**: Run `test_inference.py`
2. **🎮 Try Interactive**: Use `--interactive` mode
3. **🔧 Optimize Parameters**: Experiment with temperature/length
4. **🚀 Build Applications**: Integrate into your projects
5. **📊 Monitor Performance**: Track generation quality

---

## 🔗 Related Guides

- **[🤗 HuggingFace Guide](HUGGINGFACE_INFERENCE_README.md)** - Cloud inference
- **[📊 Test Results](INFERENCE_TEST_RESULTS.md)** - Validation data
- **[🏠 Main Guide](INFERENCE_USAGE.md)** - Overview of all options

---

*Your MoR model is ready for local inference! 🚀* 