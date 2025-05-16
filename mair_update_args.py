#!/usr/bin/env python3

import os
import argparse
import subprocess
import time
import signal
import sys
from collections import deque
import threading
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get all folders in the checkpoints directory
checkpoint_base_path = "./checkpoints/"
checkpoint_rela_path = []

for folder in os.listdir(checkpoint_base_path):

    folder_path = os.path.join(checkpoint_base_path, folder)
    # Find the checkpoint with the highest number
    max_checkpoint = None
    max_checkpoint_num = -1
    for item in os.listdir(folder_path):
        if item.startswith("checkpoint-"):
            try:
                checkpoint_num = int(item.split("-")[1])
                if checkpoint_num > max_checkpoint_num:
                    max_checkpoint_num = checkpoint_num
                    max_checkpoint = item
            except ValueError:
                continue
    
    if max_checkpoint:
        checkpoint_rela_path.append((folder, max_checkpoint_num))
        logger.info(f"Found checkpoint {max_checkpoint} for {folder}")

# Sort the list for better readability
checkpoint_rela_path.sort()

# Print the found folders for debugging
logger.info(f"Found {len(checkpoint_rela_path)} checkpoint folders: {[path[0] for path in checkpoint_rela_path]}")

# Define checkpoints and their corresponding model names
CHECKPOINTS = [
    {
        "model_name_or_path": f"./checkpoints/{checkpoint_path}/checkpoint-{checkpoint_num}",
    } for checkpoint_path, checkpoint_num in checkpoint_rela_path
]

# Define all tasks
TASKS = [
    "Core_2017", "DD_2015", "DD_2016", "DD_2017", "FairRanking_2021", 
    "FairRanking_2022", "NeuCLIR-Tech_2023", "NeuCLIR_2022", 
    "NeuCLIR_2023", "ToT_2023", "ToT_2024", "InstructIR", 
    "ExcluIR", "Core17", "News21", "Robust04", "NevIR"
]

# Number of GPUs available
NUM_GPUS = 8  # Using GPUs 0-7

def create_task_queue():
    """Create a queue of all tasks for all checkpoints."""
    task_queue = deque()
    for checkpoint_info in CHECKPOINTS:
        for task in TASKS:
            task_queue.append({
                "model_name_or_path": checkpoint_info["model_name_or_path"],
                "task": task
            })
    return task_queue

def run_task(gpu_id, task_info):
    """Run a task on a specific GPU and wait for it to complete."""
    checkpoint = task_info["model_name_or_path"]
    # model_name = task_info["model_name"]
    task = task_info["task"]
    
    # Prepare the command
    cmd = [
        "python", "eval_mair.py",
        "--model_name_or_path", checkpoint,
        "--task", task,
        "--gpu_id", str(gpu_id)
    ]
    
    # Set the environment variable for GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    logger.info(f"Starting task {task} for model {checkpoint} on GPU {gpu_id}")
    
    # Run the process directly and wait for it to complete
    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
        
        # Real-time output handling
        for line in process.stdout:
            print(f"[GPU {gpu_id} - {task} - {checkpoint}] {line.strip()}")
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode == 0:
            logger.info(f"Completed task {task} for model {checkpoint} on GPU {gpu_id}")
            return True
        else:
            logger.error(f"Task {task} for model {checkpoint} on GPU {gpu_id} failed with return code {process.returncode}")
            return False
    except Exception as e:
        logger.error(f"Task {task} for model {checkpoint} on GPU {gpu_id} failed with error: {e}")
        return False

def handle_signal(sig, frame):
    """Handle termination signals."""
    logger.info("Received termination signal. Cleaning up...")
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Create task queue
    task_queue = create_task_queue()
    logger.info(f"Created task queue with {len(task_queue)} tasks")
    
    # Dictionary to track GPU availability
    gpu_available = {gpu_id: True for gpu_id in range(NUM_GPUS)}
    
    # Main loop
    while task_queue:
        # Check for available GPUs
        for gpu_id in range(NUM_GPUS):
            if gpu_available[gpu_id] and task_queue:
                # Mark GPU as busy
                gpu_available[gpu_id] = False
                
                # Get next task
                task_info = task_queue.popleft()
                
                # Run task in a separate thread to not block the main thread
                def process_task(gpu_id, task_info):
                    run_task(gpu_id, task_info)
                    # Mark GPU as available again
                    gpu_available[gpu_id] = True
                
                thread = threading.Thread(target=process_task, args=(gpu_id, task_info))
                thread.daemon = True
                thread.start()
                
                # Small delay to prevent resource conflicts
                time.sleep(2)
        
        # Wait a bit before checking again
        time.sleep(5)
        
        # If all GPUs are busy, wait longer
        if not any(gpu_available.values()):
            time.sleep(10)
    
    # Wait for all remaining tasks to complete
    while not all(gpu_available.values()):
        time.sleep(10)
    
    logger.info("All tasks completed!")
