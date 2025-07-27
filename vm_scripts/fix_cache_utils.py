#!/usr/bin/env python3
"""
Fix script for _static_cache_update compatibility issue
"""

def fix_cache_utils():
    file_path = "/var/home/sudeshmu/mixture_of_recursions/model/kv_caches/cache_utils.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the problematic import
    old_import = "from transformers.cache_utils import _static_cache_update"
    new_import = "# from transformers.cache_utils import _static_cache_update  # Removed for compatibility"
    
    content = content.replace(old_import, new_import)
    
    # Replace the function call with StaticCache.update method
    old_call = """        return _static_cache_update(
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
            key_states,
            value_states,
            cache_kwargs.get("cache_position"),
        )"""
    
    new_call = """        # Use StaticCache update method instead of _static_cache_update
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None
        if cache_position is not None:
            self.key_cache[layer_idx][:, :, cache_position] = key_states
            self.value_cache[layer_idx][:, :, cache_position] = value_states
        else:
            # Fallback: update at the current sequence position
            seq_len = key_states.shape[-2]
            self.key_cache[layer_idx][:, :, :seq_len] = key_states
            self.value_cache[layer_idx][:, :, :seq_len] = value_states
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]"""
    
    content = content.replace(old_call, new_call)
    
    # Write back the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Fixed _static_cache_update in cache_utils.py")

if __name__ == "__main__":
    fix_cache_utils() 