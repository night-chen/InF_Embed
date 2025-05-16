import argparse
import os
import sys
import glob
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from transformers import HfArgumentParser
from config import Arguments

def modify_update_args(file_path: str, new_args: Dict[str, Any]) -> None:
    """
    Directly modifies the update_args function in train_basic.py
    
    Args:
        file_path: Path to train_basic.py
        new_args: Dictionary of arguments to update, e.g. {'pooling': 'mean', 'num_train_epochs': 3}
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the update_args function
    start_idx = content.find('def update_args(args):')
    if start_idx == -1:
        raise ValueError("Could not find update_args function in the file")
    
    # Find the return statement
    return_idx = content.find('return args', start_idx)
    if return_idx == -1:
        raise ValueError("Could not find return statement in update_args function")
    
    # Extract the function body before the return
    function_body = content[start_idx:return_idx]
    
    # Create a modified function body by updating existing args
    lines = function_body.split('\n')
    modified_lines = []
    
    for line in lines:
        line_modified = False
        for key, value in new_args.items():
            # Check if this line sets the argument we want to modify
            pattern = rf'^\s*args\.{key}\s*='
            if re.match(pattern, line.strip()):
                # Replace the line with our new value
                indent = re.match(r'^\s*', line).group(0)
                if isinstance(value, str):
                    modified_lines.append(f'{indent}args.{key} = "{value}"')
                else:
                    modified_lines.append(f'{indent}args.{key} = {value}')
                line_modified = True
                break
        
        if not line_modified:
            modified_lines.append(line)
    
    updated_function_body = '\n'.join(modified_lines)
    
    # Replace the old function with the updated one
    updated_content = content[:start_idx] + updated_function_body + content[return_idx:]
    
    # Write back to the file
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Successfully updated update_args function in {file_path}")

def rename_epoch_folder(exp_args: Dict[str, Any]) -> None:
    """
    Rename the epoch folder after experiment completion
    
    Args:
        exp_args: Dictionary containing experiment parameters
    """
    # Find the epoch folder
    epoch_folders = glob.glob("./None/epoch_*")
    if not epoch_folders:
        print("Warning: No epoch folders found to rename")
        return
    
    # Sort by creation time (newest first)
    epoch_folders.sort(key=os.path.getctime, reverse=True)
    latest_folder = epoch_folders[0]
    
    # Extract epoch number from folder name
    epoch_match = re.search(r'epoch_(\d+\.\d+)', latest_folder)
    if not epoch_match:
        print(f"Warning: Could not extract epoch number from folder name: {latest_folder}")
        return
    
    # Create new folder name
    if "div_neg_batch" in exp_args:
        new_folder_name = f"{exp_args['model_type']}_{exp_args['model'].split('/')[-1]}_{exp_args['pooling']}_share_encoder_{exp_args['share_encoder']}_epoch_{exp_args['num_train_epochs']}_contrast_mode_{exp_args['contrast_mode']}_reverse_{exp_args['data_reverse']}_padding_{exp_args['padding_side']}_div_neg_batch_{exp_args['div_neg_batch']}"
    else:
        new_folder_name = f"{exp_args['model_type']}_{exp_args['model'].split('/')[-1]}_{exp_args['pooling']}_share_encoder_{exp_args['share_encoder']}_epoch_{exp_args['num_train_epochs']}_contrast_mode_{exp_args['contrast_mode']}_reverse_{exp_args['data_reverse']}_padding_{exp_args['padding_side']}"
    new_folder_path = os.path.join(os.path.dirname(latest_folder), new_folder_name)
    
    # Rename the folder
    os.rename(latest_folder, new_folder_path)
    print(f"Renamed folder: {latest_folder} -> {new_folder_path}")


def run_experiments(experiments: List[Dict[str, Any]], 
                    train_script_path: str = "run_train.py",
                    port_start: int = 29500) -> None:
    """
    Run multiple experiments with different parameters
    
    Args:
        experiments: List of dictionaries, each containing parameters for one experiment
        train_script_path: Path to train_basic.py
        port_start: Starting port number for accelerate
    """
    # Set to use only GPUs with indices 2-7
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
    
    for i, exp_args in enumerate(experiments):
        print(f"\n\n{'='*50}")
        print(f"Running experiment {i+1}/{len(experiments)}")
        print(f"Parameters: {exp_args}")
        print(f"{'='*50}\n")
        
        # Update the args in train_basic.py
        modify_update_args(train_script_path, exp_args)
        
        # Run the experiment
        port = port_start + i
        cmd = f"accelerate launch --main_process_port={port} {train_script_path} --do_train --model_name_or_path {exp_args['model']} --output_dir checkpoints/"
        print(f"Running command: {cmd}")
        os.system(cmd)
        
        time.sleep(10)
        # Rename the epoch folder after experiment completion
        rename_epoch_folder(exp_args)
        time.sleep(10)


if __name__ == "__main__":
    # Example usage
    experiments = [
        {"model_type": "basic", "model": "Qwen/Qwen2.5-1.5B", "pooling": "last", "share_encoder": True, "num_train_epochs": 2, "contrast_mode": "qk", "data_reverse": False, "padding_side": "left", "train_file": "aarontrinh02/ms_marco_synthetic_data"},
    ]
    run_experiments(experiments)