#!/usr/bin/env python3
"""
Fix script for FlashAttentionKwargs compatibility issue
"""

import re

def fix_modeling_llama():
    file_path = "/var/home/sudeshmu/mixture_of_recursions/model/base_model/modeling_llama.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the problematic import with a try-except block and dummy class
    old_import = "from transformers.modeling_flash_attention_utils import FlashAttentionKwargs"
    
    new_import = """# FlashAttentionKwargs compatibility fix
try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
except ImportError:
    # Create dummy FlashAttentionKwargs for compatibility
    from typing_extensions import TypedDict
    class FlashAttentionKwargs(TypedDict, total=False):
        pass"""
    
    # Replace the import
    content = content.replace(old_import, new_import)
    
    # Write back the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Fixed FlashAttentionKwargs import in modeling_llama.py")

if __name__ == "__main__":
    fix_modeling_llama() 