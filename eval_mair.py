import logging
import os
import json
from datetime import datetime
import argparse
from functools import partial
from typing import Dict
import torch
from torch import nn
# import mteb
# from mteb import MTEB
from logger_config import logger, LoggerCallback
from transformers.utils.logging import enable_explicit_format
from transformers import HfArgumentParser, set_seed, AutoTokenizer, TrainingArguments
from transformers.trainer_callback import PrinterCallback
from accelerate import Accelerator
from models.basic_model import BasicModel, BasicAddModel
from models.map_model import MapModel, MapAddModel
from models.attention_model import AttentionModel, AttentionAddModel
from models.attention_map_model import AttentionMapModel, AttentionMapAddModel

from collators.basic_collator import BasicCollator
from collators.attention_collator import AttentionCollator
from dataloaders.basic_dataloader import MSMarcoDataLoader
from dataloaders.attention_dataloader import AttentionDataLoader

from trainers.basic_trainer import BasicTrainer
from trainers.map_trainer import MapTrainer
from trainers.attention_trainer import AttentionTrainer
from metrics import accuracy, batch_mrr
from config import Arguments, AcceleratorConfig
from evals.mair import eval_embedding
import wandb  # Add this import

os.environ["WANDB_PROJECT"] = "IF_Embed"  # name your W&B project
# os.environ["WANDB_LOG_MODEL"] = "biencoder-robust04"  # log all model checkpoints

# def update_args(args):
#     parser = HfArgumentParser((Arguments,))
#     hfargs: Arguments = parser.parse_args_into_dataclasses()[0]
#     for key, value in args.__dict__.items():
#         if key not in hfargs.__dict__:
#             # print(key, value)
#             setattr(hfargs, key, value)
#         else:
#             # print(key, value)
#             setattr(hfargs, key, value)
#     args = hfargs
#     args.remove_unused_columns = False
#     args.use_accelerator = True
#     args.bf16 = True
#     args.pooling = "cls"
#     args.per_device_train_batch_size = 4
#     args.share_encoder = False
#     args.model_type = "basic"
#     args.padding_side = "right"
#     args.report_to = 'wandb'
#     args.do_eval = False
#     args.per_device_test_batch_size = 128
#     args.num_workers = 16
#     args.data_reverse = False
#     args.div_neg_batch = 2
#     args.prompt_method = "none"
#     args.num_train_epochs = 1
#     args.task = "CQADupStack"
#     args.contrast_mode = "qk"
#     args.eval_output_dir = ""
#     return args


def main():

    parser = argparse.ArgumentParser(description='Baseline of training cross-encoder model.')
    parser.add_argument('--name', type=str, help='Name of the experiment.', default='IF_Embed')
    parser.add_argument('--log_dir', type=str, help='Directory to save logs.', default='./logs')
    parser.add_argument('--output_dir', type=str,  help='Output directory for model checkpoints and predictions.', default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--data_seed', type=int, default=42, help='Random seed for data loading.')
    parser.add_argument('--use_accelerator', action='store_true', help='Use accelerate for training.')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Cache directory for datasets and models.')
    parser.add_argument('--report_to', type=str, default='wandb', help='Reporting tool to use (wandb or tensorboard).')
    parser.add_argument('--gpu_id', type=int, default=0, help='Specific GPU ID to use for evaluation.')
    parser.add_argument('--task', type=str, default="CQADupStack", help='Task to evaluate on.')
    # data configurations
    parser.add_argument('--train_file', type=str, help='Path or huggingface name of the training data.', default="aarontrinh02/ms_marco_synthetic_data")
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples.')
    parser.add_argument('--p_max_len', type=int, default=512, help='Maximum length of passage input.')
    parser.add_argument('--q_max_len', type=int, default=256, help='Maximum length of query input.')

    # model configurations
    parser.add_argument('--model_name_or_path', type=str, help='Name of the model.', default="Qwen/Qwen2.5-1.5B")
    parser.add_argument('--share_encoder', action='store_true', help='Use the same encoder for query and passage.')
    parser.add_argument('--extract_first_n_layers', type=int, default=0, help='Number of layers to extract from the model.')
    parser.add_argument('--add_pooler', action='store_true', help='Add a linear pooler to the model.')

    # trainer configurations
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true', help='Whether to run evaluation.')
    parser.add_argument('--do_lora', action='store_true', help='Whether to use LoRA.')
    parser.add_argument('--batch_eval_metrics', type=bool, default=True, help='Whether to compute metrics during evaluation.')
    parser.add_argument('--eval_strategy', type=str, default='no', help='Evaluation strategy.')
    parser.add_argument('--eval_steps', type=int, default=2400, help='Evaluation steps.')
    parser.add_argument('--save_strategy', type=str, default='epoch', help='Save strategy.')
    # parser.add_argument('--save_steps', type=int, default=2400, help='Save steps.')
    parser.add_argument('--load_best_model_at_end', action='store_true', help='Load the best model at the end of training.')
    parser.add_argument('--full_determinism', action='store_true', help='Use full determinism.')
    parser.add_argument('--deepspeed_plugin', type=str, default=None, help='Path to deepspeed plugin config file.')
    parser.add_argument('--eval_use_gather_object', action='store_true', help='Use gradient checkpointing.')
    parser.add_argument('--save_only_model', action='store_true', help='Use gradient checkpointing.')
    parser.add_argument('--skip_memory_metrics', type=bool, default=True, help='Skip memory metrics.')
    parser.add_argument('--logging_steps', type=int, default=1, help='Logging steps.')
    args = parser.parse_args()
    
    # args = update_args(args)
    
    # Set specific GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    logger.info(f"Using GPU ID: {args.gpu_id}")
    

    for part in args.model_name_or_path.split('/'):
        if 'checkpoint-' in part:
            continue
        if any(model in part for model in ["Qwen2.5", "Llama-3.2", "ModernBERT", "e5", "gte"]):
            model_info = part
            break
    # Extract model_type (everything before the model name)
    if "gte" in model_info:
        model_type = model_info.split("_gte")[0]
        model_name_part = "gte" + model_info.split("gte")[1].split("_")[0]
    elif "_Qwen2.5" in model_info:
        model_type = model_info.split("_Qwen2.5")[0]
        model_name_part = "Qwen2.5" + model_info.split("_Qwen2.5")[1].split("_")[0]
    elif "_Llama-3.2" in model_info:
        model_type = model_info.split("_Llama-3.2")[0]
        model_name_part = "Llama-3.2" + model_info.split("_Llama-3.2")[1].split("_")[0]
    elif "_ModernBERT" in model_info:
        model_type = model_info.split("_ModernBERT")[0]
        model_name_part = "ModernBERT" + model_info.split("_ModernBERT")[1].split("_")[0]
    elif "_e5" in model_info:
        model_type = model_info.split("_e5")[0]
        model_name_part = "e5" + model_info.split("_e5")[1].split("_")[0]
    
    # Extract model name (including whether it has Instruct)
    model_name = model_name_part
    
    # Extract pooling type
    if "_last_" in model_info:
        pooling_type = "last"
    elif "_avg_" in model_info:
        pooling_type = "avg"
    elif "_cls_" in model_info:
        pooling_type = "cls"
    
    # Extract share_encoder
    if "_share_encoder_True_" in model_info:
        share_encoder = True
    else:
        share_encoder = False
    
    # Extract epoch
    if "_epoch_" in model_info:
        epoch_part = model_info.split("_epoch_")[1]
        epoch = int(epoch_part.split("_")[0])
    
    # Extract contrast_mode (everything after contrast_mode_)
    if "_contrast_mode_" in model_info:
        contrast_mode_part = model_info.split("_contrast_mode_")[1]
        contrast_mode = contrast_mode_part.split("_reverse_")[0]

    if "_reverse_True_" in model_info:
        reverse_mode = True
    else:
        reverse_mode = False

    if "e5" in model_name or "ModernBERT" in model_name:
        padding_side = "right"
    else:
        padding_side = "left"

    if "div_neg_batch_" in model_info:
        div_neg_batch = int(model_info.split("_div_neg_batch_")[1])
    else:
        div_neg_batch = None

    args.model_type = model_type
    args.model_name = model_name
    args.pooling = pooling_type
    args.share_encoder = share_encoder
    args.epoch = epoch
    args.contrast_mode = contrast_mode
    args.data_reverse = reverse_mode
    args.padding_side = padding_side
    args.div_neg_batch = div_neg_batch
    args.prompt_method = "none"
    args.l2_normalize = False
    args.add_pooler = False
    args.remove_unused_columns = False
    args.p_max_len = 512
    args.q_max_len = 256
    args.cache_dir = './cache'

    print(args)
        
    logger.info(f"Loading checkpoint with the following configuration:")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Base model: {model_name}")
    logger.info(f"Pooling: {pooling_type}")
    logger.info(f"Share encoder: {share_encoder}")
    logger.info(f"Trained for {epoch} epochs")
    logger.info(f"Contrast mode: {contrast_mode}")
    logger.info(f"Data reverse: {reverse_mode}")
    logger.info(f"Padding side: {padding_side}")
    logger.info(f"Div neg batch: {div_neg_batch}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side

    if model_type == "basic" or model_type == "baseline_basic" or model_type == "basic_reverse" or model_type == "basic_plus_reverse":
        model = BasicModel.build(args)
    elif model_type == "map" or model_type == "baseline_map" or model_type == "map_plus":
        model = BasicModel.build(args)
    # elif model_type == "attn" or model_type == "baseline_attn":
    #     model = AttentionModel.build(args)
    # elif model_type == "attn_map" or model_type == "baseline_attn_map":
    #     model = AttentionMapModel.build(args)
    elif model_type == "basic_add" or model_type == "baseline_basic_add":
        model = BasicModel.build(args)
    elif model_type == "map_add" or model_type == "baseline_map_add":
        model = BasicModel.build(args)
    # elif model_type == "attn_add" or model_type == "baseline_attn_add":
    #     model = AttentionAddModel.build(args)
    # elif model_type == "attn_map_add" or model_type == "baseline_attn_map_add":
    #     model = AttentionMapAddModel.build(args)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    model.update_tokenizer(tokenizer)
    
    # Convert model to bf16 and move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.is_available():
        model = model.to(torch.bfloat16)
        logger.info(f"Model converted to bfloat16 and moved to {device}")
    else:
        logger.warning("CUDA not available, model will run on CPU without bf16 conversion")

    encode_kwargs = {
        "batch_size": 128, 
        "num_workers": 16,
    }
    output_dir = "./mair_results"
    task_name = args.task
    sub_dir = f"{model_type}_{model_name}_{pooling_type}_share_encoder_{share_encoder}_epoch_{epoch}_contrast_mode_{contrast_mode}_reverse_{reverse_mode}_padding_{padding_side}_div_neg_batch_{div_neg_batch}"
    os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)
    outputs = eval_embedding(model, tasks=[task_name], instruct=True) 
    with open(os.path.join(output_dir, sub_dir, f"{task_name}.json"), 'w') as f:
        json.dump(outputs, f)
if __name__ == "__main__":
    main()


# basic_qwen2.5-1.5B_last_share_encoder_True
# basic_qwen2.5-1.5B_last_share_encoder_False
# attn_qwen2.5-1.5B_last_share_encoder_True
# attn_qwen2.5-1.5B_last_share_encoder_False
# basic_ModernBERT_cls_share_encoder_True
# basic_ModernBERT_cls_share_encoder_False
# attn_ModernBERT_cls_share_encoder_True
# attn_ModernBERT_cls_share_encoder_False
# basic_qwen2.5-1.5B-Instruct_last_share_encoder_True
# basic_qwen2.5-1.5B-Instruct_last_share_encoder_False
# attn_qwen2.5-1.5B-Instruct_last_share_encoder_True
# attn_qwen2.5-1.5B-Instruct_last_share_encoder_False