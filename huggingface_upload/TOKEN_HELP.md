# 🔑 Get Your New HuggingFace Token

## ❌ Current Token Issue

The token `hf_bDyFzztOUscZrrhvJrJpJSQgUzkTaQgAu` is **invalid/expired**.

Error: `401 Unauthorized - Invalid user token`

## ✅ How to Get a New Working Token

### Step 1: Go to HuggingFace Settings
🔗 **https://huggingface.co/settings/tokens**

### Step 2: Create New Token
1. Click **"New token"** button
2. **Name**: `MoR Model Upload` (or any name you like)
3. **Type**: Select **"Write"** ⚠️ (This is crucial!)
4. Click **"Generate token"**

### Step 3: Copy the New Token
- The token will start with `hf_`
- It will look like: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
- ⚠️ **Copy it immediately** - it's only shown once!

### Step 4: Use the New Token
Once you have your new token, run:

```bash
cd mixture_of_recursions/huggingface_upload
export HF_TOKEN="your_new_token_here"
python upload_mor_model.py --username sudeshmu --model-name mixture-of-recursions-360m
```

## 🚨 Common Token Issues

❌ **"Read" permission only** - Need "Write" permission  
❌ **Token expired** - Generate a new one  
❌ **Token revoked** - Generate a new one  
❌ **Wrong format** - Must start with `hf_`

## 🎯 What We're Uploading

- **Model**: `mixture_of_recursions/trained_model/pytorch_model.bin` (523MB)
- **Target**: `https://huggingface.co/sudeshmu/mixture-of-recursions-360m`
- **Files**: Model weights, config, README, and more

**Your model is ready to upload - just need a working token! 🚀** 