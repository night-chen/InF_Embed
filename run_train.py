import os
import wandb
import logging
from datetime import datetime
import argparse
from functools import partial
from typing import Dict
import torch
import random
import numpy as np
from torch import nn
from logger_config import logger, LoggerCallback
from transformers.utils.logging import enable_explicit_format
from transformers import HfArgumentParser, set_seed, AutoTokenizer
from transformers.trainer_callback import PrinterCallback
from accelerate import Accelerator

from models.basic_model import BasicModel, BasicAddModel
from models.map_model import MapModel, MapAddModel
from models.attention_model import AttentionModel, AttentionAddModel
from models.attention_map_model import AttentionMapModel, AttentionMapAddModel
from models.map_sum_model import MapSumModel
from collators.basic_collator import BasicCollator
from collators.attention_collator import AttentionCollator
from dataloaders.basic_dataloader import MSMarcoDataLoader
from dataloaders.attention_dataloader import AttentionDataLoader

from trainers.basic_trainer import BasicTrainer
from trainers.map_trainer import MapTrainer
from trainers.attention_trainer import AttentionTrainer

from metrics import accuracy, batch_mrr
from config import Arguments, AcceleratorConfig


os.environ["WANDB_PROJECT"] = "IF_Embed"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_DISABLE_CODE"] = "true"


def _common_setup(args):
    # logger.setLevel(logger.INFO)
    enable_explicit_format()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

def _compute_metrics(args, eval_pred, compute_result) -> Dict[str, float]:
    preds = eval_pred.predictions
    scores = torch.tensor(preds[-1]).float()
    labels = torch.arange(0, scores.shape[0], dtype=torch.long) * args.train_n_passages
    labels = labels % scores.shape[1]
    topk_metrics = accuracy(output=scores, target=labels, topk=(1, 3))
    mrr = batch_mrr(output=scores, target=labels)
    return {'mrr': mrr, 'acc1': topk_metrics[0], 'acc3': topk_metrics[1]}

def update_args(args):
    parser = HfArgumentParser((Arguments,))
    hfargs: Arguments = parser.parse_args_into_dataclasses()[0]
    for key, value in args.__dict__.items():
        if key not in hfargs.__dict__:
            # print(key, value)
            setattr(hfargs, key, value)
        else:
            # print(key, value)
            setattr(hfargs, key, value)
    args = hfargs
    args.remove_unused_columns = False
    args.use_accelerator = True
    args.bf16 = True
    args.pooling = "last"
    args.per_device_train_batch_size = 4
    args.share_encoder = True
    args.model_type = "basic"
    args.padding_side = "left"
    args.report_to = 'wandb'
    args.do_eval = False
    args.per_device_test_batch_size = 128
    args.num_workers = 16
    args.data_reverse = False
    args.train_file = "aarontrinh02/ms_marco_synthetic_data"
    args.div_neg_batch = 2
    args.prompt_method = "none"
    args.num_train_epochs = 2
    args.contrast_mode = "qk"
    args.eval_output_dir = ""
    return args
    
def main():

    
    parser = argparse.ArgumentParser(description='Baseline of training cross-encoder model.')
    parser.add_argument('--name', type=str, help='Name of the experiment.', default='IF_Embed')
    parser.add_argument('--log_dir', type=str, help='Directory to save logs.', default='./logs')
    parser.add_argument('--output_dir', type=str,  help='Output directory for model checkpoints and predictions.', default='checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--data_seed', type=int, default=42, help='Random seed for data loading.')
    parser.add_argument('--use_accelerator', action='store_true', help='Use accelerate for training.')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Cache directory for datasets and models.')
    parser.add_argument('--report_to', type=str, default='wandb', help='Reporting tool to use (wandb or tensorboard).')

    # data configurations
    parser.add_argument('--train_file', type=str, help='Path or huggingface name of the training data.', default="aarontrinh02/ms_marco_synthetic_data_unfiltered")
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
    parser.add_argument('--save_strategy', type=str, default='steps', help='Save strategy.')
    parser.add_argument('--save_steps', type=int, default=2400, help='Save steps.')
    parser.add_argument('--load_best_model_at_end', action='store_true', help='Load the best model at the end of training.')
    parser.add_argument('--full_determinism', action='store_true', help='Use full determinism.')
    parser.add_argument('--deepspeed_plugin', type=str, default=None, help='Path to deepspeed plugin config file.')
    parser.add_argument('--eval_use_gather_object', action='store_true', help='Use gradient checkpointing.')
    parser.add_argument('--save_only_model', action='store_true', help='Use gradient checkpointing.')
    parser.add_argument('--skip_memory_metrics', type=bool, default=True, help='Skip memory metrics.')
    parser.add_argument('--logging_steps', type=int, default=1, help='Logging steps.')
    args = parser.parse_args()
    args.accelerator_config = AcceleratorConfig(
        split_batches=False,
        dispatch_batches=None,
        even_batches=True,
        use_seedable_sampler=True,
        non_blocking=False,
        gradient_accumulation_kwargs=None,
        use_configured_state=False
    )
    args = update_args(args)

    assert args.model_type in ["basic", "map", "attn", "attn_map", "basic_unfilter",
                               "basic_add", "map_add", "attn_add", "attn_map_add", "map_sum"]
    assert args.pooling in ["cls", "last", "avg"]
    if args.model_type in ["basic", "attn", "basic_add", "attn_add", "basic_unfilter"]:
        assert args.contrast_mode in ["qk", "kq", "no_trick", "same_tower"]
    elif args.model_type in ["map", "attn_map", "map_add", "attn_map_add", "map_sum"]:
        assert args.contrast_mode in ["qk_with_neg", "kq_with_neg", "no_trick_with_neg", "same_tower_with_neg", "only_neg"]
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    
    model_name = args.model_name_or_path.split("/")[-1]
    args.output_dir = f"./checkpoints/{args.model_type}_{model_name}_{args.pooling}_share_encoder_{args.share_encoder}_epoch_{args.num_train_epochs}_contrast_mode_{args.contrast_mode}_reverse_{args.data_reverse}_padding_{args.padding_side}"
    args.eval_output_dir = f"./all_results/{args.model_type}_{model_name}_{args.pooling}_share_encoder_{args.share_encoder}_epoch_{args.num_train_epochs}_contrast_mode_{args.contrast_mode}_reverse_{args.data_reverse}_padding_{args.padding_side}"
    if args.model_type in ["map", "attn_map", "map_add", "attn_map_add", "map_sum"]:
        args.output_dir += f"_div_neg_batch_{args.div_neg_batch}"
        args.eval_output_dir += f"_div_neg_batch_{args.div_neg_batch}"
    elif args.train_file == "aarontrinh02/ms_marco_synthetic_data":
        args.eval_steps = 1200 * args.num_train_epochs
        args.save_steps = 1199 * args.num_train_epochs
    elif args.train_file == "aarontrinh02/ms_marco_synthetic_data_unfiltered":
        args.eval_steps = 3124 * args.num_train_epochs
        args.save_steps = 3123 * args.num_train_epochs
    else:
        raise ValueError(f"Invalid train file: {args.train_file}")
    _common_setup(args)
    logger.info('Args={}'.format(str(args)))
    
    
    accelerator = Accelerator(log_with='wandb', project_dir=f"{args.log_dir}/{args.name}") if args.use_accelerator else None
    date = datetime.now().strftime("%m%d%H%M")
    run_id = f"{args.model_type}-{date}"
    accelerator.init_trackers(
        project_name="IF_Embed", 
        config=args.__dict__,
        init_kwargs={"wandb": {"id": run_id}}
    )
    


    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = args.padding_side
    
    if args.model_type == "basic":
        model = BasicModel.build(args)
        data_collator = BasicCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
        dataloader = MSMarcoDataLoader(args=args, tokenizer=tokenizer)
        trainer_class = BasicTrainer
    elif args.model_type == "map":
        model = MapModel.build(args)
        data_collator = AttentionCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
        dataloader = AttentionDataLoader(args=args, tokenizer=tokenizer)
        trainer_class = MapTrainer
    elif args.model_type == "attn":
        model = AttentionModel.build(args)
        data_collator = AttentionCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
        dataloader = AttentionDataLoader(args=args, tokenizer=tokenizer)
        trainer_class = AttentionTrainer
    elif args.model_type == "attn_map":
        model = AttentionMapModel.build(args)
        data_collator = AttentionCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
        dataloader = AttentionDataLoader(args=args, tokenizer=tokenizer)
        trainer_class = AttentionTrainer
    elif args.model_type == "basic_add":
        model = BasicAddModel.build(args)
        data_collator = BasicCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
        dataloader = MSMarcoDataLoader(args=args, tokenizer=tokenizer)
        trainer_class = BasicTrainer
    elif args.model_type == "map_add":
        model = MapAddModel.build(args)
        data_collator = AttentionCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
        dataloader = AttentionDataLoader(args=args, tokenizer=tokenizer)
        trainer_class = MapTrainer
    elif args.model_type == "attn_add":
        model = AttentionAddModel.build(args)
        data_collator = AttentionCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
        dataloader = AttentionDataLoader(args=args, tokenizer=tokenizer)
        trainer_class = AttentionTrainer
    elif args.model_type == "attn_map_add":
        model = AttentionMapAddModel.build(args)
        data_collator = AttentionCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
        dataloader = AttentionDataLoader(args=args, tokenizer=tokenizer)
        trainer_class = AttentionTrainer
    elif args.model_type == "map_sum":
        model = MapSumModel.build(args)
        data_collator = AttentionCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
        dataloader = AttentionDataLoader(args=args, tokenizer=tokenizer)
        trainer_class = MapTrainer
    elif args.model_type == "basic_unfilter":
        model = BasicModel.build(args)
        data_collator = BasicCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
        dataloader = MSMarcoDataLoader(args=args, tokenizer=tokenizer)
        trainer_class = BasicTrainer
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    
    
    
    model.update_tokenizer(tokenizer)
    logger.info(model)
    logger.info('Vocab size: {}'.format(len(tokenizer)))
    
    train_dataset = dataloader.train_dataset
    eval_dataset = dataloader.eval_dataset if args.do_eval else None
    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=partial(_compute_metrics, args),
        tokenizer=tokenizer
    )
    
    if accelerator != None:
        trainer = accelerator.prepare(trainer)
        model.lm_q, dataloader, trainer = accelerator.prepare(model.lm_q, dataloader, trainer)
        if not args.share_encoder:
            model.lm_p = accelerator.prepare(model.lm_p)
    
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LoggerCallback)
    dataloader.trainer = trainer
    model.trainer = trainer
    
    if args.do_train:
        train_result = trainer.train()
        print(train_result)
        if args.do_lora:
            trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    
    if not args.do_lora:
        if accelerator != None:
            accelerator.save_model(model.lm_q, args.output_dir, max_shard_size="2GB", safe_serialization=True)
        else:
            save_model = unwrap_model(model.lm_q)
            state_dict = save_model.state_dict()
            save_model.save_pretrained(args.output_dir, state_dict=state_dict, safe_serialization=False)
    return
    accelerator.end_training()

def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model

if __name__ == "__main__":
    main()