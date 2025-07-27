# MoR Model Hugging Face Upload Guide

This directory contains everything you need to upload your Mixture-of-Recursions (MoR) model to Hugging Face Hub.

## 📁 Files Created

```
huggingface_upload/
├── README.md                         # Comprehensive model documentation for HF
├── modeling_mor.py                   # Simplified MoR model implementation  
├── requirements.txt                  # HF-specific requirements
├── upload_mor_model.py              # Main upload script (secure token prompt)
├── get_token_and_upload.py          # Helper script with token instructions
├── comprehensive_hf_inference_test.py # Test script for uploaded HF model
├── upload_to_hf.py                  # Generic upload utility
├── test_model.py                    # Model testing script
├── create_repo.py                   # Interactive repository creation
├── setup_and_upload.sh              # Automated bash script
└── UPLOAD_GUIDE.md                  # This guide
```

## 🚀 Quick Start (Updated)

### Step 1: Get a Valid HuggingFace Token

The provided token appears to be invalid/expired. You need a fresh token:

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "MoR Model Upload")
4. Select **"Write" permission**
5. Copy the token (starts with `hf_`)

### Step 2: Upload Your Model

#### Option 1: Guided Upload (Easiest)

```bash
cd mixture_of_recursions/huggingface_upload
python get_token_and_upload.py --username your-username --model-name mixture-of-recursions-360m
```

This will:
- Show you token instructions
- Guide you through the upload
- Prompt securely for your token

#### Option 2: Direct Upload

```bash
cd mixture_of_recursions/huggingface_upload
python upload_mor_model.py --username your-username --model-name mixture-of-recursions-360m
```

You'll be prompted for your token securely.

### Step 3: Test Your Uploaded Model

Once uploaded successfully, test it:

```bash
python comprehensive_hf_inference_test.py --model-name your-username/mixture-of-recursions-360m
```

This will run 50 comprehensive test cases on your uploaded model!

## 🔑 Token Requirements

**Important**: Your HuggingFace token must have **"Write" permission** to create repositories and upload models.

- ✅ **Good**: Token with "Write" permission
- ❌ **Bad**: Token with only "Read" permission
- ❌ **Bad**: Expired or invalid token

## 🧪 Testing the Uploaded Model

The new `comprehensive_hf_inference_test.py` script:

- Loads your model directly from HuggingFace Hub
- Runs 50 diverse test cases (same as local tests)
- Works with any uploaded MoR model
- Provides detailed performance breakdown
- Supports both CPU and GPU inference

```bash
# Test your uploaded model
python comprehensive_hf_inference_test.py --model-name your-username/mixture-of-recursions-360m

# Test with specific device
python comprehensive_hf_inference_test.py --model-name your-username/mixture-of-recursions-360m --device cpu
```

## 🎯 What Gets Uploaded

Your repository will contain:

- **README.md**: Professional HF model card with usage examples
- **modeling_mor.py**: MoR model implementation for HF compatibility
- **config.json**: Updated model configuration with MoR settings
- **pytorch_model.bin**: Your trained model weights (~523MB)
- **generation_config.json**: Generation parameters
- **tokenizer_config.json**: Tokenizer configuration
- **requirements.txt**: Dependencies

## 💡 Using Your Uploaded Model

Once uploaded, anyone can use your model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "your-username/mixture-of-recursions-360m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True  # Required for MoR models
)

# Generate text
prompt = "The key to artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 🔒 Security Notes

- ✅ **Token Security**: Token is prompted securely, never stored permanently
- ✅ **Temporary Files**: Upload preparation files are cleaned up automatically
- ✅ **Model Weights**: Only necessary model files are uploaded
- ✅ **Trust Remote Code**: Required flag for custom MoR architecture

## 🛠️ Advanced Options

### Custom Model Directory
```bash
python upload_mor_model.py --username your-username --model-name mor-model --model-dir /path/to/model
```

### Private Repository
```bash
python create_repo.py --username your-username --model-name mor-model --private
```

### Environment Variable Token
```bash
export HF_TOKEN="your_valid_token_here"
python upload_mor_model.py --username your-username --model-name mixture-of-recursions-360m
```

## 🔍 Troubleshooting

### Common Issues

1. **Invalid Token**
   ```
   ❌ Token validation failed: Invalid user token
   ```
   **Solution**: Get a new token with "Write" permission

2. **Missing Model Files**
   ```
   ❌ pytorch_model.bin not found
   ```
   **Solution**: Ensure model training completed successfully

3. **Repository Already Exists**
   ```
   ❌ Repository already exists
   ```
   **Solution**: Use a different model name or delete existing repo

4. **Network Issues**
   ```
   ❌ Upload failed: Connection timeout
   ```
   **Solution**: Check internet connection, try again

### Getting Help

1. **Check model files**: `ls -la ../trained_model/`
2. **Verify token**: Visit https://huggingface.co/settings/tokens
3. **Test connection**: `python -c "from huggingface_hub import HfApi; print('✅ HF Hub connection OK')"`

## 📝 Repository Naming

**Good names:**
- `mixture-of-recursions-360m`
- `mor-llama-360m`
- `adaptive-recursion-model`

**Avoid:**
- Spaces in names
- Special characters
- Very long names

## 🎉 After Upload

1. **View your model**: https://huggingface.co/your-username/model-name
2. **Test generation**: Use the inference widget on HF
3. **Run comprehensive tests**: Use `comprehensive_hf_inference_test.py`
4. **Share**: Send the HF link to collaborators
5. **Update**: Push new versions with upload scripts

## 📚 Additional Resources

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub)
- [HF Token Management](https://huggingface.co/docs/hub/security-tokens)
- [MoR Paper](https://arxiv.org/abs/2507.10524)
- [Original MoR Repository](https://github.com/raymin0223/mixture_of_recursions)

---

**Need help?** Check the troubleshooting section or create an issue in the original repository. 