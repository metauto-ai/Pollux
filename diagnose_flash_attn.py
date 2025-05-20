#!/usr/bin/env python
"""
Diagnostic script for flash-attention issues
This script helps identify issues with flash-attention installation and imports.
"""

import os
import sys
import importlib.util
import subprocess
from pathlib import Path

def print_header(title):
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def check_module_exists(module_name):
    return importlib.util.find_spec(module_name) is not None

def run_cmd(cmd):
    print(f"Running: {cmd}")
    try:
        output = subprocess.check_output(cmd, shell=True, universal_newlines=True, stderr=subprocess.STDOUT)
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output

def main():
    print_header("FLASH ATTENTION DIAGNOSTIC TOOL")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check CUDA availability
    print("\nCUDA Environment:")
    cuda_home = os.environ.get("CUDA_HOME", "Not set")
    print(f"CUDA_HOME: {cuda_home}")
    
    # Check for PyTorch installation
    print("\nChecking PyTorch installation:")
    if check_module_exists("torch"):
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("PyTorch is not installed")
    
    # Check flash attention installation
    print("\nChecking flash-attention installation:")
    if check_module_exists("flash_attn"):
        try:
            import flash_attn
            print(f"flash-attn version: {flash_attn.__version__}")
            
            # Check if the interface module exists
            print("\nChecking flash_attn.flash_attn_interface:")
            if check_module_exists("flash_attn.flash_attn_interface"):
                from flash_attn.flash_attn_interface import flash_attn_varlen_func
                print("flash_attn.flash_attn_interface.flash_attn_varlen_func is available")
            else:
                print("flash_attn.flash_attn_interface module not found")
            
            # Check compatibility layer
            print("\nChecking flash_attn_interface compatibility layer:")
            if check_module_exists("flash_attn_interface"):
                import flash_attn_interface
                if hasattr(flash_attn_interface, "flash_attn_varlen_func"):
                    print("flash_attn_interface.flash_attn_varlen_func is available")
                else:
                    print("flash_attn_interface module exists but flash_attn_varlen_func is not available")
            else:
                print("flash_attn_interface compatibility layer not found")
                
            # Check flash-attention module file locations
            flash_attn_path = Path(flash_attn.__file__).parent
            print(f"\nflash-attn installation path: {flash_attn_path}")
            
            interface_files = list(flash_attn_path.glob("*interface*"))
            if interface_files:
                print("Interface files found:")
                for f in interface_files:
                    print(f"  {f}")
            else:
                print("No interface files found in flash-attn directory")
                
        except ImportError as e:
            print(f"Error importing flash-attn modules: {e}")
    else:
        print("flash-attn is not installed")
    
    # Check for common issues
    print_header("CHECKING FOR COMMON ISSUES")
    
    # Check Castor component.py import
    castor_component = "apps/Castor/modules/component.py"
    if os.path.exists(castor_component):
        print(f"\nChecking {castor_component} for import issues:")
        success, output = run_cmd(f"grep -n 'flash_attn' {castor_component} | head -5")
        if success:
            print(output)
        else:
            print(f"Error checking {castor_component}: {output}")
    else:
        print(f"\n{castor_component} not found")
    
    # Run pip check to look for dependency issues
    print("\nChecking for dependency issues:")
    success, output = run_cmd("pip check")
    if success:
        print("No dependency issues found")
    else:
        print("Dependency issues found:")
        print(output)
    
    # Provide recommendations
    print_header("RECOMMENDATIONS")
    print("""
1. If flash-attn is not installed:
   - Run: pip install flash-attn --no-build-isolation
   
2. If the compatibility layer is missing:
   - Create a file at site-packages/flash_attn_interface.py with:
     from flash_attn.flash_attn_interface import flash_attn_varlen_func
     
3. If Castor's component.py has the wrong import:
   - Update it to use: from flash_attn.flash_attn_interface import flash_attn_varlen_func
   - Or create the compatibility layer as in step 2
   
4. If there are CUDA errors:
   - Check CUDA version compatibility with installed PyTorch
   - Check if CUDA_HOME is set correctly
   
For more detailed diagnosis, check the flash-attention repository at:
https://github.com/Dao-AILab/flash-attention
    """)

if __name__ == "__main__":
    main() 