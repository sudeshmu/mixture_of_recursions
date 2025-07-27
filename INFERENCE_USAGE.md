# Mixture of Recursions (MoR) - Inference Usage Guide

## 🚀 Overview

This repository provides comprehensive inference capabilities for the **Mixture of Recursions** model, supporting both local inference and HuggingFace Hub deployment.

## 📋 Quick Start Options

### 🏠 **Local Inference**
Use your locally trained model directly:
```bash
# Interactive chat mode
python vm_scripts/local_inference.py --interactive

# Single prompt
python vm_scripts/local_inference.py --prompt "Your prompt here"

# Comprehensive testing
python comprehensive_inference_test.py
```

### 🤗 **HuggingFace Hub Inference**
Use the model deployed on HuggingFace Hub:
```bash
# Test from HuggingFace
python huggingface_upload/comprehensive_hf_inference_test.py --model-name sudeshmu/mixture-of-recursions-360m
```

## 📁 File Structure

```
mixture_of_recursions/
├── 📄 INFERENCE_USAGE.md                    # This file
├── 📄 LOCAL_INFERENCE_README.md             # Local inference guide
├── 📄 HUGGINGFACE_INFERENCE_README.md       # HuggingFace inference guide
├── 📄 INFERENCE_TEST_RESULTS.md             # Test results documentation
├── 🧪 comprehensive_inference_test.py       # Local model testing
├── 🧪 test_inference.py                     # Basic local testing
├── 🤗 huggingface_upload/
│   └── comprehensive_hf_inference_test.py   # HuggingFace model testing
├── 🎮 vm_scripts/local_inference.py         # Interactive inference
└── 📦 trained_model/                        # Local model files
    ├── pytorch_model.bin (499MB)
    ├── config.json
    └── generation_config.json
```

## 🎯 Choose Your Method

| Method | Use Case | Requirements |
|--------|----------|--------------|
| **🏠 Local** | Private, fast inference | Local model files |
| **🤗 HuggingFace** | Cloud access, sharing | Internet connection |

## 📖 Detailed Guides

- **[🏠 Local Inference Guide](LOCAL_INFERENCE_README.md)** - Complete local setup and usage
- **[🤗 HuggingFace Guide](HUGGINGFACE_INFERENCE_README.md)** - Cloud-based inference
- **[📊 Test Results](INFERENCE_TEST_RESULTS.md)** - Comprehensive validation results

## ⚡ Quick Test Commands

### Local Model Test:
```bash
# Basic functionality test
python test_inference.py

# Comprehensive 50-test validation
python comprehensive_inference_test.py
```

### HuggingFace Model Test:
```bash
# Test cloud model (replace with your model name)
python huggingface_upload/comprehensive_hf_inference_test.py --model-name YOUR_USERNAME/YOUR_MODEL_NAME
```

## 🔧 Requirements

**Core Dependencies:**
- `torch` - PyTorch framework
- `transformers` - HuggingFace transformers
- `accelerate` - Model loading optimization

**Install:**
```bash
pip install torch transformers accelerate
```

## 🎮 Interactive Features

Both local and HuggingFace inference support:
- **Interactive chat mode** with conversation memory
- **Custom generation parameters** (temperature, top-p, length)
- **Batch processing** capabilities
- **Comprehensive testing** with 50 diverse prompts

## 📊 Model Specifications

- **Architecture**: LlamaForCausalLM with MoR enhancements
- **Parameters**: ~362M (361,821,120)
- **Vocabulary**: 49,152 tokens
- **Context Length**: 1,024 tokens
- **Features**: Dynamic recursion, adaptive computation depth

## 🏆 Validation Status

✅ **100% Test Success Rate**  
✅ **50/50 Comprehensive Tests Passed**  
✅ **Production Ready**  
✅ **Zero Technical Failures**

## 🚀 Get Started

1. **Choose your method**: Local or HuggingFace
2. **Read the specific guide**: See detailed READMEs above
3. **Run tests**: Validate functionality
4. **Start inferencing**: Interactive or programmatic use

---

*For detailed instructions, see the method-specific README files linked above.* 