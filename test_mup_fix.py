#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/akito/Pollux')

import torch
import copy
import gc
from apps.Castor.model import Castor, ModelArgs, build_2B_Castor
from apps.Castor.modules.transformer import TransformerArgs
from mup import set_base_shapes

def test_mup_setup():
    print("Testing MuP setup...")
    
    # Build the main model
    model_args = build_2B_Castor().args
    model = Castor(model_args)
    
    print(f"Main model config:")
    print(f"  dim: {model_args.diffusion_model.dim}")
    print(f"  n_heads: {model_args.diffusion_model.n_heads}")
    print(f"  n_kv_heads: {model_args.diffusion_model.n_kv_heads}")
    
    # Create base model
    base_args = copy.deepcopy(model_args)
    base_args.diffusion_model.dim = 288
    base_args.diffusion_model.n_heads = 4
    base_args.diffusion_model.n_kv_heads = 2
    base_model = Castor(base_args)
    
    print(f"Base model config:")
    print(f"  dim: {base_args.diffusion_model.dim}")
    print(f"  n_heads: {base_args.diffusion_model.n_heads}")
    print(f"  n_kv_heads: {base_args.diffusion_model.n_kv_heads}")

    # Create delta model
    delta_args = copy.deepcopy(model_args)
    delta_args.diffusion_model.dim = 360
    delta_args.diffusion_model.n_heads = 5
    delta_args.diffusion_model.n_kv_heads = 3
    delta_model = Castor(delta_args)
    
    print(f"Delta model config:")
    print(f"  dim: {delta_args.diffusion_model.dim}")
    print(f"  n_heads: {delta_args.diffusion_model.n_heads}")
    print(f"  n_kv_heads: {delta_args.diffusion_model.n_kv_heads}")

    # Test MuP setup
    try:
        print("\nSetting base shapes...")
        set_base_shapes(model, base_model, delta=delta_model)
        print("✅ MuP setup successful!")
        
        # Test a forward pass
        print("\nTesting forward pass...")
        model.eval()
        
        # Create dummy batch
        batch = {
            'latent_code': torch.randn(2, 16, 32, 32),  # B, C, H, W
            'text_embedding': torch.randn(2, 128, 512),  # B, seq_len, dim
            'attention_mask': torch.ones(2, 128, dtype=torch.bool)
        }
        
        with torch.no_grad():
            output = model(batch)
            print(f"✅ Forward pass successful! Loss: {output.loss.item():.4f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    finally:
        # Cleanup
        del base_model, delta_model
        gc.collect()
    
    return True

if __name__ == "__main__":
    success = test_mup_setup()
    if success:
        print("\n🎉 All tests passed! The MuP fix is working correctly.")
    else:
        print("\n💥 Tests failed. There may still be issues with the MuP setup.") 