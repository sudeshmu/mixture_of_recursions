# 🤗 HuggingFace Inference Guide - Mixture of Recursions

## 🎯 Overview

This guide covers running inference with your **Mixture of Recursions** model deployed on HuggingFace Hub, enabling cloud-based access and sharing.

## 🚀 Quick Start

### **Test the Model**
```bash
python huggingface_upload/comprehensive_hf_inference_test.py --model-name sudeshmu/mixture-of-recursions-360m
```

### **Use in Your Code**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model from HuggingFace Hub
model_name = "sudeshmu/mixture-of-recursions-360m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Generate text
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## 📋 Prerequisites

### 📦 **Dependencies**
```bash
pip install torch transformers accelerate
```

### 🔐 **HuggingFace Account** (Optional)
- For private models or uploading: Create account at [huggingface.co](https://huggingface.co)
- For public models: No account needed

## 🎮 Usage Methods

### 1. **🧪 Comprehensive Testing**
```bash
# Test with 50 diverse prompts
python huggingface_upload/comprehensive_hf_inference_test.py --model-name sudeshmu/mixture-of-recursions-360m

# Test with custom parameters
python huggingface_upload/comprehensive_hf_inference_test.py \
    --model-name sudeshmu/mixture-of-recursions-360m \
    --device cuda
```

### 2. **📝 Python Integration**
```python
#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_mor_model(model_name="sudeshmu/mixture-of-recursions-360m"):
    """Load MoR model from HuggingFace Hub."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    """Generate text with the model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only new tokens
    input_length = inputs.input_ids.shape[1]
    new_tokens = outputs[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# Usage
model, tokenizer = load_mor_model()
result = generate_text(model, tokenizer, "Hello, I am")
print(result)
```

### 3. **🔧 Advanced Configuration**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Advanced loading with specific device/precision
model = AutoModelForCausalLM.from_pretrained(
    "sudeshmu/mixture-of-recursions-360m",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Automatic device placement
    low_cpu_mem_usage=True
)

# Advanced generation
outputs = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.95,
    do_sample=True,
    repetition_penalty=1.1,
    num_return_sequences=1
)
```

## 🔍 Model Information

### **HuggingFace Model Card**
- **Model Name**: `sudeshmu/mixture-of-recursions-360m`
- **Architecture**: LlamaForCausalLM with MoR enhancements
- **Parameters**: ~362M (361,821,120)
- **License**: Check model card on HuggingFace
- **Usage**: Research and experimentation

### **Key Features**
- ✅ **Dynamic Recursion**: Adaptive computation per token
- ✅ **Efficient Inference**: Optimized resource usage
- ✅ **Cloud Access**: Available via HuggingFace Hub
- ✅ **Easy Integration**: Standard transformers interface

## 📊 Testing & Validation

### **Comprehensive Test Suite**
The `comprehensive_hf_inference_test.py` script tests:
- 🤖 **Technical/AI prompts** (10 tests)
- 📝 **Creative writing** (10 tests)  
- 🧬 **Science/nature** (10 tests)
- 💬 **Conversational** (10 tests)
- 🧩 **Problem solving** (10 tests)

### **Expected Results**
```
✅ Successful tests: 50/50
⏱️  Total time: ~60-120 seconds (depending on connection/hardware)
📊 100% success rate across all categories
```

### **Performance Metrics**
- **GPU Inference**: ~0.5-1.0s per generation
- **CPU Inference**: ~1.5-3.0s per generation
- **Memory Usage**: 2-4GB depending on device/precision

## 🌐 Network & Hardware

### **Internet Requirements**
- **Initial Download**: ~500MB model weights
- **Inference**: No internet needed after download
- **Caching**: Models cached locally after first download

### **Hardware Recommendations**

| Device | Performance | Memory |
|--------|-------------|---------|
| **GPU (CUDA)** | ⭐⭐⭐⭐⭐ | 4-8GB VRAM |
| **CPU (Intel/AMD)** | ⭐⭐⭐ | 8-16GB RAM |
| **Apple Silicon** | ⭐⭐⭐⭐ | 8-16GB Unified Memory |

## 🔧 Troubleshooting

### **Common Issues**

#### ❌ "Repository not found"
```bash
# Check model name spelling
python huggingface_upload/comprehensive_hf_inference_test.py --model-name sudeshmu/mixture-of-recursions-360m
```

#### ❌ "Connection timeout"
```bash
# Try with longer timeout or check internet connection
export TRANSFORMERS_CACHE="/path/to/cache"
```

#### ❌ "CUDA out of memory"
```bash
# Force CPU usage
python huggingface_upload/comprehensive_hf_inference_test.py \
    --model-name sudeshmu/mixture-of-recursions-360m \
    --device cpu
```

#### ❌ "trust_remote_code error"
```python
# Always use trust_remote_code=True for MoR models
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True  # Required!
)
```

### **Performance Optimization**

#### **For GPU:**
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Faster inference
    device_map="auto",           # Auto GPU placement
    trust_remote_code=True
)
```

#### **For CPU:**
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # CPU compatibility
    device_map="cpu",           # Force CPU
    trust_remote_code=True
)
```

## 🌟 Advantages of HuggingFace Deployment

### **✅ Benefits:**
- **🌐 Universal Access**: Use from any machine with internet
- **🔄 Easy Sharing**: Share model with simple name
- **📦 No Local Storage**: No need for large local files
- **🔄 Version Control**: Model versioning and updates
- **📊 Usage Analytics**: Track model usage (if enabled)
- **🤝 Community**: Easy collaboration and sharing

### **⚠️ Considerations:**
- **🌐 Internet Required**: Initial download needs connection
- **🐌 First Load**: Slower first load (downloads model)
- **🔐 Privacy**: Public models are visible to all
- **💰 Costs**: Potential hosting costs for private models

## 🚀 Production Usage

### **Web Application Example**
```python
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_model():
    model_name = "sudeshmu/mixture-of-recursions-360m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer

st.title("MoR Text Generator")
model, tokenizer = load_model()

prompt = st.text_input("Enter your prompt:")
if prompt:
    result = generate_text(model, tokenizer, prompt)
    st.write(result)
```

### **API Service Example**
```python
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Load model once at startup
model_name = "sudeshmu/mixture-of-recursions-360m"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    result = generate_text(model, tokenizer, prompt)
    return jsonify({'generated_text': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 📈 Next Steps

1. **🧪 Test the Model**: Run the comprehensive test script
2. **🔧 Integrate**: Add to your applications
3. **⚙️ Optimize**: Tune parameters for your use case
4. **📊 Monitor**: Track performance and quality
5. **🚀 Deploy**: Use in production applications

---

## 🔗 Related Resources

- **[🏠 Local Inference](LOCAL_INFERENCE_README.md)** - Local model usage
- **[📊 Test Results](INFERENCE_TEST_RESULTS.md)** - Validation data
- **[🏠 Main Guide](INFERENCE_USAGE.md)** - Overview of all options
- **[🤗 HuggingFace Hub](https://huggingface.co/sudeshmu/mixture-of-recursions-360m)** - Model page

---

*Your MoR model is ready for cloud inference! 🌟* 