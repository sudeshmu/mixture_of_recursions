---
license: mit
library_name: transformers
tags:
- mixture-of-recursions
- adaptive-computation
- early-exiting
- llama
- language-model
- efficient-inference
base_model: microsoft/DialoGPT-medium
datasets:
- HuggingFaceTB/smollm-corpus
language:
- en
pipeline_tag: text-generation
model_type: llama
---

# Mixture-of-Recursions (MoR): Learning Dynamic Recursive Depths for Adaptive Token-Level Computation

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv:2507.10524-Green)](https://arxiv.org/abs/2507.10524)
[![GitHub](https://img.shields.io/badge/GitHub-mixture_of_recursions-blue)](https://github.com/raymin0223/mixture_of_recursions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## Model Description

This is a **Mixture-of-Recursions (MoR)** model that implements adaptive token-level computation through dynamic recursive depths. MoR addresses key bottlenecks in early-exiting techniques by introducing a unified framework that tackles both missing Key-Value (KV) cache problems and inefficient batched inference.

**Key Features:**
- 🚀 **Up to 2× greater inference throughput** compared to standard transformers at similar accuracy
- 🧠 **Dynamic routing mechanism** that assigns optimal recursion depth to each token
- 💾 **Recursion-wise KV caching strategy** that optimizes memory usage
- ⚡ **Efficient batched inference** through parameter sharing
- 🎯 **End-to-end trainable** architecture

### Model Details

- **Model Size**: 360M parameters  
- **Architecture**: Based on LLaMA with MoR modifications
- **Context Length**: 1024 tokens
- **Vocabulary Size**: 49,152 tokens
- **Hidden Size**: 960
- **Number of Layers**: 32
- **Attention Heads**: 15 (5 KV heads)
- **Training Data**: FineWeb-Edu deduplicated subset

## Quick Start

### Installation

```bash
pip install torch transformers accelerate
```

### Basic Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "your-username/mixture-of-recursions-360m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate text
prompt = "The key to artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### Advanced Usage with Custom Recursion

```python
# For advanced users: Access MoR-specific features
# Note: This requires the original MoR codebase for full functionality

from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_name)
# The model supports dynamic recursion depths through routing mechanisms
# See the original repository for complete MoR training and inference scripts
```

## Model Architecture

The MoR model introduces several key innovations over standard transformers:

### 1. Dynamic Routing Mechanism
- **Expert-choice routing**: Dynamically selects which tokens to process at each recursion depth
- **Token-choice routing**: Allows tokens to choose their optimal processing depth
- **Trainable routers**: End-to-end learning of routing decisions

### 2. Recursion-wise KV Caching
- Solves the missing KV cache problem in early-exiting models
- Selective KV pair storage for memory optimization
- Enables efficient parallel decoding

### 3. Parameter Sharing Strategies
- **Cycle sharing**: Enables tokens at different depths to be processed together
- **Middle cycle sharing**: Optimizes parameter utilization across recursion levels

## Training Details

- **Training Framework**: PyTorch with DeepSpeed/Accelerate
- **Hardware**: 4x H100/A100 GPUs
- **Optimization**: AdamW with cosine learning rate schedule
- **Mixed Precision**: bfloat16
- **Gradient Accumulation**: Multi-step accumulation for effective large batch training

## Performance

### Efficiency Gains
- **Throughput**: Up to 2× improvement over standard transformers
- **Memory**: Reduced memory requirements through optimized KV caching
- **Training**: Lower total FLOPs during training

### Accuracy Preservation
The model maintains competitive performance on standard benchmarks while providing significant efficiency improvements.

## Use Cases

- **Efficient text generation**: Ideal for applications requiring fast inference
- **Resource-constrained deployment**: Suitable for edge devices and mobile applications  
- **Real-time applications**: Chat systems, interactive AI assistants
- **Research**: Adaptive computation and early-exiting research

## Limitations

- Custom architecture requires specific handling for full MoR features
- Optimal performance achieved with the complete MoR training framework
- May require model-specific optimizations for deployment

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{bae2025mixtureofrecursionslearningdynamicrecursive,
    title={Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation}, 
    author={Sangmin Bae and Yujin Kim and Reza Bayat and Sungnyun Kim and Jiyoun Ha and Tal Schuster and Adam Fisch and Hrayr Harutyunyan and Ziwei Ji and Aaron Courville and Se-Young Yun},
    year={2025},
    eprint={2507.10524},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2507.10524}, 
}
```

## License

This model is released under the MIT License. See the LICENSE file for details.

## Authors

**Sangmin Bae**, **Yujin Kim**, **Reza Bayat**, Sungnyun Kim, Jiyoun Ha, Tal Schuster, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Aaron Courville, Se-Young Yun

*KAIST AI, Mila, Google Cloud, Google DeepMind, Google Research, Université de Montréal*

## Links

- 📄 [Paper](https://arxiv.org/abs/2507.10524)
- 💻 [GitHub Repository](https://github.com/raymin0223/mixture_of_recursions)
- 🤗 [Hugging Face Model](https://huggingface.co/your-username/mixture-of-recursions-360m)

---

*For complete training scripts, evaluation code, and advanced MoR features, please visit the [official GitHub repository](https://github.com/raymin0223/mixture_of_recursions).* 