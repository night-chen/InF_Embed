import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from functools import partial
from torch import Tensor
from transformers import (
    AutoModel,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput
from peft import LoraConfig, get_peft_model, PeftConfig

from logger_config import logger
from utils import (
    dist_gather_tensor, 
    select_grouped_indices, 
    full_contrastive_scores_and_labels, 
    full_contrastive_scores_and_labels_add,
    angle_loss, 
    print_trainable_parameters
)
from .pooling import pool
import torch.distributed as dist

from mteb.encoder_interface import PromptType
from dataloaders.basic_dataloader import IndexedDataset
from tqdm import tqdm


def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model

def formulate_with_prompt(sentences, prompt_method):
    prompts = []
    if not isinstance(sentences, list):
        sentences = [sentences]
    if prompt_method is None or prompt_method.lower() == "none":
        return sentences
    for sentence in sentences:
        # sentence = ' '.join(sentence) if sentence != [] else '.'
        if len(sentence) > 0 and sentence[-1] not in '.?!"\'': 
                sentence += '.'
        sentence = sentence.replace('"', '\'')
        prompts.append(prompt_method.replace('*sent 0*', sentence).replace('_', ' ').strip())
    return prompts


@dataclass
class BasicOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BasicModel(nn.Module):
    def __init__(self, args,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel):
        super().__init__()
        self.uni_encoder = args.share_encoder
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.args = args
        self.linear_pooler = nn.Linear(self.lm_q.config.hidden_size, args.out_dimension) if args.add_pooler else nn.Identity()
        self.config = self.lm_q.config

        if "e5" in args.model_name_or_path:
            if hasattr(self.lm_q, 'pooler') and self.lm_q.pooler is not None:
                logger.info("Disabling gradients for lm_q.pooler as its output is not used by BasicModel's _encode method.")
                for param in self.lm_q.pooler.parameters():
                    param.requires_grad = False
            
            # If lm_p is a different model instance (not shared encoder) and also has an unused pooler
            if not self.uni_encoder and hasattr(self.lm_p, 'pooler') and self.lm_p.pooler is not None:
                logger.info("Disabling gradients for lm_p.pooler as its output is not used by BasicModel's _encode method.")
                for param in self.lm_p.pooler.parameters():
                    param.requires_grad = False

        from trainers.basic_trainer import BasicTrainer
        self.trainer: Optional[BasicTrainer] = None
        self._formulate = partial(formulate_with_prompt, prompt_method=args.prompt_method)

    def update_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def sentence_embedding_flagemb(self, hidden_state, mask):
        if self.args.pooling == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.args.pooling == 'cls':
            return hidden_state[:, 0]
        elif self.args.pooling == "last":
            left_padding = (mask[:, -1].sum() == mask.shape[0])
            if left_padding:
                emb = hidden_state[:, -1]
            else:
                sequence_lengths = mask.sum(dim=1) - 1
                batch_size = hidden_state.shape[0]
                emb = hidden_state[torch.arange(batch_size, device=hidden_state.device), sequence_lengths]
            return emb

    def encode_flagemb(self, encoder: PreTrainedModel,  features):
        if features is None:
            return None
        psg_out = encoder(**features, return_dict=True)
        p_reps = self.sentence_embedding_flagemb(psg_out.last_hidden_state, features['attention_mask'])
        # if self.linear_pooler_use == True:
        p_reps = self.linear_pooler(p_reps)
        
        if self.args.l2_normalize:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward_flagemb(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_flagemb(self.lm_q, query)
        p_reps = self.encode_flagemb(self.lm_p, passage)

        if self.training:
            # if self.negatives_cross_device:
            q_reps = dist_gather_tensor(q_reps)
            p_reps = dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores / self.args.t
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))
            loss = self.cross_entropy(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return q_reps, p_reps, scores, loss

    def forward(self, 
                instruction_query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None):
        assert self.args.process_index >= 0
        scores, labels, q_reps, p_reps, all_scores, all_labels = self._compute_scores(instruction_query, passage)
        start = self.args.process_index * q_reps.shape[0]
        group_indices = select_grouped_indices(scores=scores,
                                               group_size=self.args.train_n_passages,
                                               start=start * self.args.train_n_passages)

        if not self.args.do_kd_biencoder:
            # training biencoder from scratch
            if self.args.use_scaled_loss:
                loss = self.cross_entropy(all_scores, all_labels)
                loss *= self.args.world_size if self.args.loss_scale <= 0 else self.args.loss_scale
            else:
                loss = self.cross_entropy(scores, labels)
        else:
            pass

        if self.args.do_angle_loss:
            loss += self.args.angle_loss_weight * self._compute_angle_loss(q_reps, p_reps)

        total_n_psg = self.args.world_size * q_reps.shape[0] #* self.args.train_n_passages

        return BasicOutput(loss=loss, q_reps=q_reps, p_reps=p_reps,
                               labels=labels.contiguous(),
                               scores=scores[:, :total_n_psg].contiguous())

    def _compute_scores(self, query: Dict[str, Tensor] = None,
                        passage: Dict[str, Tensor] = None) -> Tuple:
        q_reps = self._encode(self.lm_q, query)
        p_reps = self._encode(self.lm_p, passage)
        all_q_reps = dist_gather_tensor(q_reps)
        all_p_reps = dist_gather_tensor(p_reps)
        assert all_p_reps.shape[0] == self.args.world_size * q_reps.shape[0]

        all_scores, all_labels = full_contrastive_scores_and_labels(
            query=all_q_reps, key=all_p_reps,
            contrast_mode=self.args.contrast_mode,
            use_all_pairs=self.args.full_contrastive_loss)

        if self.args.l2_normalize:
            if self.args.t_warmup:
                scale = 1.0 / self.args.t * min(1.0, self.trainer.state.global_step / self.args.warmup_steps)
                scale = max(1.0, scale)
            else:
                scale = float(1.0 / (1e-4+self.args.t))
            all_scores = all_scores * float(scale)


        start = self.args.process_index * q_reps.shape[0]
        local_query_indices = torch.arange(start, start + q_reps.shape[0], dtype=torch.long).to(q_reps.device)
        # batch_size x (world_size x batch_size x train_n_passage)
        scores = all_scores.index_select(dim=0, index=local_query_indices)
        labels = all_labels.index_select(dim=0, index=local_query_indices)
        return scores, labels, q_reps, p_reps, all_scores, all_labels

    def _compute_angle_loss(self, q_reps: torch.Tensor, p_reps: torch.Tensor):
        assert q_reps.shape[0]*self.args.train_n_passages == p_reps.shape[0]
        # Get (query_embedding, doc_embedding) pairs for all positive docs and negative docs
        num_q = q_reps.shape[0]
        hidden_dim = q_reps.shape[1]
        expanded_q_reps = torch.unsqueeze(q_reps, 1).expand(num_q, self.args.train_n_passages, hidden_dim)
        expanded_q_reps = expanded_q_reps.reshape(-1, hidden_dim)
        pair_reps = torch.cat((expanded_q_reps, p_reps), 1).reshape(-1,hidden_dim)
        # Label 0 for negative doc, 1 for positive doc
        labels = torch.arange(0, num_q, dtype=torch.long, device=q_reps.device)
        labels = labels * self.args.train_n_passages
        labels_q_ids = labels*2
        labels_p_ids = labels_q_ids+1
        pair_labels = torch.zeros(pair_reps.shape[0]).to(q_reps.device)
        #logger.debug(f"pair_labels.shape: {pair_labels.shape} labels_q_ids: {labels_q_ids}, labels_p_ids: {labels_p_ids}")

        pair_labels[labels_p_ids]=1
        pair_labels[labels_q_ids]=1
        pair_labels = torch.unsqueeze(pair_labels, -1)
        return angle_loss(pair_labels, pair_reps)


    def _encode(self, encoder: PreTrainedModel, input_dict: dict) -> Optional[torch.Tensor]:
        if not input_dict:
            return None
        outputs = encoder(**{k: v for k, v in input_dict.items() if k not in ['kd_labels']}, return_dict=True)
        hidden_state = outputs.last_hidden_state
        embeds = pool(last_hidden_states=hidden_state, attention_mask=input_dict['attention_mask'], pool_type=self.args.pooling)
        embeds = self.linear_pooler(embeds)

        if self.args.l2_normalize:
            embeds = F.normalize(embeds, dim=-1)
        return embeds.contiguous()
    
    def encode(
        self,
        sentences: list[str],
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.
            
            Args:
                sentences: The sentences to encode.
                task_name: The name of the task.
                prompt_type: The prompt type to use.
                **kwargs: Additional arguments to pass to the encoder.
                
            Returns:
                The encoded sentences.
        """     
        # accelerator = kwargs.get("accelerator", None)
        sentences = self._formulate(sentences)
        batch_size = kwargs.get("batch_size", 32)
        all_embeddings = []
        if prompt_type == PromptType.query:
            self.tokenizer_config = {
                "padding": 'max_length', 
                "truncation": True,
                "max_length": self.args.q_max_len,
                "return_tensors": "pt",
                "pad_to_multiple_of": 8,
                "padding_side": self.args.padding_side
            }
        else:
            self.tokenizer_config = {
                "padding": 'max_length',
                "truncation": True,
                "max_length": self.args.p_max_len,
                "return_tensors": "pt",
                "pad_to_multiple_of": 8,
                "padding_side": self.args.padding_side
            }
        # if accelerator is None:
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index+batch_size]
            model_inputs = self.tokenizer(
                sentences_batch,
                **self.tokenizer_config,
            )
            with torch.no_grad():
                if prompt_type == PromptType.query: 
                    model_inputs = {k: v.to(self.lm_q.device) for k, v in model_inputs.items()}
                    embeddings = self._encode(self.lm_q, model_inputs)
                else:
                    model_inputs = {k: v.to(self.lm_p.device) for k, v in model_inputs.items()}
                    embeddings = self._encode(self.lm_p, model_inputs)
            all_embeddings.append(embeddings.detach().cpu())
        final_embeddings = torch.cat(all_embeddings, dim=0)
        return final_embeddings.float().numpy()


    @classmethod
    def build(cls, args, **hf_kwargs):
        # load local
        if os.path.isdir(args.model_name_or_path):
            if not args.share_encoder:
                _qry_model_path = os.path.join(args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    logger.info('loading query and passage model from one model')
                    _qry_model_path = args.model_name_or_path
                    _psg_model_path = args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                try:
                    lm_q = AutoModel.from_pretrained(_qry_model_path, cache_dir = args.cache_dir, attn_implementation="flash_attention_2", trust_remote_code=True, **hf_kwargs)
                except Exception as e:
                    logger.info(f'loading query model weight from {_qry_model_path} failed, trying without flash attention')
                    lm_q = AutoModel.from_pretrained(_qry_model_path, cache_dir = args.cache_dir, trust_remote_code=True, **hf_kwargs)
                #  attn_implementation="flash_attention_2",
                logger.info(f'loading passage model weight from {_psg_model_path}')
                try:
                    lm_p = AutoModel.from_pretrained(_psg_model_path, cache_dir = args.cache_dir, attn_implementation="flash_attention_2", trust_remote_code=True, **hf_kwargs)
                except Exception as e:
                    logger.info(f'loading passage model weight from {_psg_model_path} failed, trying without flash attention')
                    lm_p = AutoModel.from_pretrained(_psg_model_path, cache_dir = args.cache_dir, trust_remote_code=True, **hf_kwargs)
                if args.extract_first_n_layers > 0:
                    raise Exception(" Not implemented extract layers for bi-encoder")
            else:
                encoder_name = os.path.join(args.model_name_or_path, 'encoder')
                if args.extract_first_n_layers > 0:
                    logger.info(f"load first {args.extract_first_n_layers} transformers layers from {args.model_name_or_path}")
                    try:
                        lm_q = AutoModel.from_pretrained(encoder_name, attn_implementation="flash_attention_2",  num_hidden_layers=args.extract_first_n_layers, cache_dir = args.cache_dir, **hf_kwargs)
                    except Exception as e:
                        logger.info(f'loading shared model weight from {args.model_name_or_path} failed, trying without flash attention')
                        lm_q = AutoModel.from_pretrained(encoder_name, attn_implementation="flash_attention_2",  num_hidden_layers=args.extract_first_n_layers, cache_dir = args.cache_dir, **hf_kwargs)
                    # attn_implementation="flash_attention_2",
                else:
                    try:
                        logger.info(f'loading shared model weight from {args.model_name_or_path}')
                        lm_q = AutoModel.from_pretrained(encoder_name, attn_implementation="flash_attention_2", cache_dir = args.cache_dir, **hf_kwargs)
                    except Exception as e:
                        logger.info(f'loading shared model weight from {args.model_name_or_path} failed, trying without flash attention')
                        lm_q = AutoModel.from_pretrained(encoder_name, cache_dir = args.cache_dir, **hf_kwargs)
                    #  attn_implementation="flash_attention_2",

        # load pre-trained from hub
        else:
            # take entire model if args.extract_first_n_layers = 0 
            if args.extract_first_n_layers > 0:
                logger.info(f"load first {args.extract_first_n_layers} transformers layers from the entire model")
                lm_q = AutoModel.from_pretrained(args.model_name_or_path, attn_implementation="flash_attention_2", num_hidden_layers=args.extract_first_n_layers, cache_dir = args.cache_dir, **hf_kwargs)
                # attn_implementation="flash_attention_2",  
            else:
                try:
                    if "falcon" in args.model_name_or_path:
                        logger.info("Not Flash Attention 2!")
                        lm_q = AutoModel.from_pretrained(args.model_name_or_path, cache_dir = args.cache_dir, torch_dtype=torch.bfloat16, **hf_kwargs)
                    else:
                        logger.info("Flash Attention 2!")
                        lm_q = AutoModel.from_pretrained(args.model_name_or_path, attn_implementation="flash_attention_2", cache_dir = args.cache_dir, torch_dtype=torch.bfloat16, **hf_kwargs)
                except ValueError:
                    logger.info("Error!")
                    lm_q = AutoModel.from_pretrained(args.model_name_or_path, cache_dir = args.cache_dir, **hf_kwargs)

            if not args.share_encoder:
                lm_p = copy.deepcopy(lm_q) 
        
        if args.share_encoder:
            lm_p = lm_q

        model = cls(args=args, lm_q=lm_q, lm_p=lm_p)
        return model

    def save(self, output_dir: str):
        if self.args.do_lora:
            # Save LoRA-only parameters
            output_dir = os.path.join(output_dir, "lora")
            logger.info(f"save Lora-only parameters")
        if not self.args.share_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'passage_model'), exist_ok=True)
            logger.info(f"save bi-encoders query model into {output_dir}/query_model")
            logger.info(f"save bi-encoders passage model into {output_dir}/passage_model")
            self.query_model_save_path = os.path.join(output_dir, 'query_model') 
            self.passage_model_save_path = os.path.join(output_dir, 'passage_model')
            save_lm_p = unwrap_model(self.lm_p)
            lm_p_state_dict = save_lm_p.state_dict()
            save_lm_p.save_pretrained(self.passage_model_save_path)
            save_lm_q = unwrap_model(self.lm_q)
            lm_q_state_dict = save_lm_q.state_dict()
            save_lm_q.save_pretrained(self.query_model_save_path)
        else:
            save_lm_q = unwrap_model(self.lm_q)
            lm_q_state_dict = save_lm_q.state_dict()
            os.makedirs(os.path.join(output_dir, 'encoder'), exist_ok=True)
            self.query_model_save_path = os.path.join(output_dir, 'encoder')
            save_lm_q.save_pretrained(self.query_model_save_path)
    

class BasicAddModel(nn.Module):
    def __init__(self, args,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel):
        super().__init__()
        self.uni_encoder = args.share_encoder
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.args = args
        self.linear_pooler = nn.Linear(self.lm_q.config.hidden_size, args.out_dimension) if args.add_pooler else nn.Identity()
        self.config = self.lm_q.config

        from trainers.basic_trainer import BasicTrainer
        self.trainer: Optional[BasicTrainer] = None
        self._formulate = partial(formulate_with_prompt, prompt_method=args.prompt_method)

    def update_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def sentence_embedding_flagemb(self, hidden_state, mask):
        if self.args.pooling == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.args.pooling == 'cls':
            return hidden_state[:, 0]
        elif self.args.pooling == "last":
            left_padding = (mask[:, -1].sum() == mask.shape[0])
            if left_padding:
                emb = hidden_state[:, -1]
            else:
                sequence_lengths = mask.sum(dim=1) - 1
                batch_size = hidden_state.shape[0]
                emb = hidden_state[torch.arange(batch_size, device=hidden_state.device), sequence_lengths]
            return emb

    def encode_flagemb(self, encoder: PreTrainedModel,  features):
        if features is None:
            return None
        psg_out = encoder(**features, return_dict=True)
        p_reps = self.sentence_embedding_flagemb(psg_out.last_hidden_state, features['attention_mask'])
        p_reps = self.linear_pooler(p_reps)
        
        if self.args.l2_normalize:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward_flagemb(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_flagemb(self.lm_q, query)
        p_reps = self.encode_flagemb(self.lm_p, passage)

        if self.training:
            # if self.negatives_cross_device:
            q_reps = dist_gather_tensor(q_reps)
            p_reps = dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores / self.args.t
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))
            loss = self.cross_entropy(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return q_reps, p_reps, scores, loss

    def forward(self, 
                instruction_query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None):
        assert self.args.process_index >= 0
        scores, labels, q_reps, p_reps, all_scores, all_labels = self._compute_scores(instruction_query, passage)
        start = self.args.process_index * q_reps.shape[0]
        group_indices = select_grouped_indices(scores=scores,
                                               group_size=self.args.train_n_passages,
                                               start=start * self.args.train_n_passages)

        if not self.args.do_kd_biencoder:
            # training biencoder from scratch
            if self.args.use_scaled_loss:
                # print(self.args.contrast_mode)
                if self.args.contrast_mode == "qk" or self.args.contrast_mode == "kq":
                    loss = self.cross_entropy(all_scores, all_labels)
                elif self.args.contrast_mode == "no_trick":
                    # Split scores in the middle along the second dimension
                    mid_point = all_scores.shape[1] // 2
                    scores_part1 = all_scores[:, :mid_point]
                    scores_part2 = all_scores[:, mid_point:]
                    # Calculate cross entropy for each part and sum them
                    loss = self.cross_entropy(scores_part1, all_labels) + self.cross_entropy(scores_part2, all_labels)
                elif self.args.contrast_mode == "same_tower":
                    # Split scores into four parts along the second dimension
                    quarter_point = all_scores.shape[1] // 4
                    scores_part1 = all_scores[:, :quarter_point]
                    scores_part2 = all_scores[:, quarter_point:2*quarter_point]
                    scores_part3 = all_scores[:, 2*quarter_point:3*quarter_point]
                    scores_part4 = all_scores[:, 3*quarter_point:]
                    # Calculate cross entropy for each part and sum them
                    loss = (self.cross_entropy(scores_part1, all_labels) + 
                           self.cross_entropy(scores_part2, all_labels) + 
                           self.cross_entropy(scores_part3, all_labels) + 
                           self.cross_entropy(scores_part4, all_labels))
                else:
                    raise ValueError(f"Unknown contrast_mode: {self.args.contrast_mode}")
                loss *= self.args.world_size if self.args.loss_scale <= 0 else self.args.loss_scale
            else:
                # print(self.args.contrast_mode)
                if self.args.contrast_mode == "qk" or self.args.contrast_mode == "kq":
                    loss = self.cross_entropy(all_scores, all_labels)
                elif self.args.contrast_mode == "no_trick":
                    # Split scores in the middle along the second dimension
                    mid_point = all_scores.shape[1] // 2
                    scores_part1 = all_scores[:, :mid_point]
                    scores_part2 = all_scores[:, mid_point:]
                    # Calculate cross entropy for each part and sum them
                    loss = self.cross_entropy(scores_part1, all_labels) + self.cross_entropy(scores_part2, all_labels)
                elif self.args.contrast_mode == "same_tower":
                    # Split scores into four parts along the second dimension
                    quarter_point = all_scores.shape[1] // 4
                    scores_part1 = all_scores[:, :quarter_point]
                    scores_part2 = all_scores[:, quarter_point:2*quarter_point]
                    scores_part3 = all_scores[:, 2*quarter_point:3*quarter_point]
                    scores_part4 = all_scores[:, 3*quarter_point:]
                    # Calculate cross entropy for each part and sum them
                    loss = (self.cross_entropy(scores_part1, all_labels) + 
                           self.cross_entropy(scores_part2, all_labels) + 
                           self.cross_entropy(scores_part3, all_labels) + 
                           self.cross_entropy(scores_part4, all_labels))
                else:
                    raise ValueError(f"Unknown contrast_mode: {self.args.contrast_mode}")

        else:
            pass

        if self.args.do_angle_loss:
            loss += self.args.angle_loss_weight * self._compute_angle_loss(q_reps, p_reps)

        total_n_psg = self.args.world_size * q_reps.shape[0]

        return BasicOutput(loss=loss, q_reps=q_reps, p_reps=p_reps,
                               labels=labels.contiguous(),
                               scores=scores[:, :total_n_psg].contiguous())

    def _compute_scores(self, query: Dict[str, Tensor] = None,
                        passage: Dict[str, Tensor] = None) -> Tuple:
        q_reps = self._encode(self.lm_q, query)
        p_reps = self._encode(self.lm_p, passage)
        all_q_reps = dist_gather_tensor(q_reps)
        all_p_reps = dist_gather_tensor(p_reps)
        assert all_p_reps.shape[0] == self.args.world_size * q_reps.shape[0]

        all_scores, all_labels = full_contrastive_scores_and_labels_add(
            query=all_q_reps, key=all_p_reps,
            contrast_mode=self.args.contrast_mode,
            use_all_pairs=self.args.full_contrastive_loss)

        if self.args.l2_normalize:
            if self.args.t_warmup:
                scale = 1.0 / self.args.t * min(1.0, self.trainer.state.global_step / self.args.warmup_steps)
                scale = max(1.0, scale)
            else:
                scale = float(1.0 / (1e-4+self.args.t))
            all_scores = all_scores * float(scale)

        start = self.args.process_index * q_reps.shape[0]
        local_query_indices = torch.arange(start, start + q_reps.shape[0], dtype=torch.long).to(q_reps.device)
        # batch_size x (world_size x batch_size x train_n_passage)
        scores = all_scores.index_select(dim=0, index=local_query_indices)
        labels = all_labels.index_select(dim=0, index=local_query_indices)

        return scores, labels, q_reps, p_reps, all_scores, all_labels

    def _compute_angle_loss(self, q_reps: torch.Tensor, p_reps: torch.Tensor):
        assert q_reps.shape[0]*self.args.train_n_passages == p_reps.shape[0]
        # Get (query_embedding, doc_embedding) pairs for all positive docs and negative docs
        num_q = q_reps.shape[0]
        hidden_dim = q_reps.shape[1]
        expanded_q_reps = torch.unsqueeze(q_reps, 1).expand(num_q, self.args.train_n_passages, hidden_dim)
        expanded_q_reps = expanded_q_reps.reshape(-1, hidden_dim)
        pair_reps = torch.cat((expanded_q_reps, p_reps), 1).reshape(-1,hidden_dim)
        # Label 0 for negative doc, 1 for positive doc
        labels = torch.arange(0, num_q, dtype=torch.long, device=q_reps.device)
        labels = labels * self.args.train_n_passages
        labels_q_ids = labels*2
        labels_p_ids = labels_q_ids+1
        pair_labels = torch.zeros(pair_reps.shape[0]).to(q_reps.device)


        pair_labels[labels_p_ids]=1
        pair_labels[labels_q_ids]=1
        pair_labels = torch.unsqueeze(pair_labels, -1)
        return angle_loss(pair_labels, pair_reps)


    def _encode(self, encoder: PreTrainedModel, input_dict: dict) -> Optional[torch.Tensor]:
        if not input_dict:
            return None
        outputs = encoder(**{k: v for k, v in input_dict.items() if k not in ['kd_labels']}, return_dict=True)
        hidden_state = outputs.last_hidden_state
        embeds = pool(last_hidden_states=hidden_state, attention_mask=input_dict['attention_mask'], pool_type=self.args.pooling)
        embeds = self.linear_pooler(embeds)

        if self.args.l2_normalize:
            embeds = F.normalize(embeds, dim=-1)
        return embeds.contiguous()
    
    def encode(
        self,
        sentences: list[str],
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.
            
            Args:
                sentences: The sentences to encode.
                task_name: The name of the task.
                prompt_type: The prompt type to use.
                **kwargs: Additional arguments to pass to the encoder.
                
            Returns:
                The encoded sentences.
        """     
        # accelerator = kwargs.get("accelerator", None)
        sentences = self._formulate(sentences)
        batch_size = kwargs.get("batch_size", 32)
        all_embeddings = []
        if prompt_type == PromptType.query:
            self.tokenizer_config = {
                "padding": 'max_length', 
                "truncation": True,
                "max_length": self.args.q_max_len,
                "return_tensors": "pt",
                "pad_to_multiple_of": 8,
                "padding_side": self.args.padding_side
            }
        else:
            self.tokenizer_config = {
                "padding": 'max_length',
                "truncation": True,
                "max_length": self.args.p_max_len,
                "return_tensors": "pt",
                "pad_to_multiple_of": 8,
                "padding_side": self.args.padding_side
            }
        # if accelerator is None:
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index+batch_size]
            model_inputs = self.tokenizer(
                sentences_batch,
                **self.tokenizer_config,
            )
            with torch.no_grad():
                if prompt_type == PromptType.query: 
                    model_inputs = {k: v.to(self.lm_q.device) for k, v in model_inputs.items()}
                    embeddings = self._encode(self.lm_q, model_inputs)
                else:
                    model_inputs = {k: v.to(self.lm_p.device) for k, v in model_inputs.items()}
                    embeddings = self._encode(self.lm_p, model_inputs)
            all_embeddings.append(embeddings.detach().cpu())
        final_embeddings = torch.cat(all_embeddings, dim=0)
        # print(final_embeddings)
        return final_embeddings.float().numpy()



    @classmethod
    def build(cls, args, **hf_kwargs):
        # load local
        if os.path.isdir(args.model_name_or_path):
            if not args.share_encoder:
                _qry_model_path = os.path.join(args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    logger.info('loading query and passage model from one model')
                    _qry_model_path = args.model_name_or_path
                    _psg_model_path = args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                try:
                    lm_q = AutoModel.from_pretrained(_qry_model_path, cache_dir = args.cache_dir, attn_implementation="flash_attention_2", trust_remote_code=True, **hf_kwargs)
                except Exception as e:
                    logger.info(f'loading query model weight from {_qry_model_path} failed, trying without flash attention')
                    lm_q = AutoModel.from_pretrained(_qry_model_path, cache_dir = args.cache_dir, trust_remote_code=True, **hf_kwargs)
                #  attn_implementation="flash_attention_2",
                logger.info(f'loading passage model weight from {_psg_model_path}')
                try:
                    lm_p = AutoModel.from_pretrained(_psg_model_path, cache_dir = args.cache_dir, attn_implementation="flash_attention_2", trust_remote_code=True, **hf_kwargs)
                except Exception as e:
                    logger.info(f'loading passage model weight from {_psg_model_path} failed, trying without flash attention')
                    lm_p = AutoModel.from_pretrained(_psg_model_path, cache_dir = args.cache_dir, trust_remote_code=True, **hf_kwargs)
                if args.extract_first_n_layers > 0:
                    raise Exception(" Not implemented extract layers for bi-encoder")
            else:
                encoder_name = os.path.join(args.model_name_or_path, 'encoder')
                if args.extract_first_n_layers > 0:
                    logger.info(f"load first {args.extract_first_n_layers} transformers layers from {args.model_name_or_path}")
                    try:
                        lm_q = AutoModel.from_pretrained(encoder_name, attn_implementation="flash_attention_2",  num_hidden_layers=args.extract_first_n_layers, cache_dir = args.cache_dir, **hf_kwargs)
                    except Exception as e:
                        logger.info(f'loading shared model weight from {args.model_name_or_path} failed, trying without flash attention')
                        lm_q = AutoModel.from_pretrained(encoder_name, attn_implementation="flash_attention_2",  num_hidden_layers=args.extract_first_n_layers, cache_dir = args.cache_dir, **hf_kwargs)
                    # attn_implementation="flash_attention_2",
                else:
                    try:
                        logger.info(f'loading shared model weight from {args.model_name_or_path}')
                        lm_q = AutoModel.from_pretrained(encoder_name, attn_implementation="flash_attention_2", cache_dir = args.cache_dir, **hf_kwargs)
                    except Exception as e:
                        logger.info(f'loading shared model weight from {args.model_name_or_path} failed, trying without flash attention')
                        lm_q = AutoModel.from_pretrained(encoder_name, cache_dir = args.cache_dir, **hf_kwargs)
                    #  attn_implementation="flash_attention_2",

        # load pre-trained from hub
        else:
            # take entire model if args.extract_first_n_layers = 0 
            if args.extract_first_n_layers > 0:
                logger.info(f"load first {args.extract_first_n_layers} transformers layers from the entire model")
                lm_q = AutoModel.from_pretrained(args.model_name_or_path, attn_implementation="flash_attention_2", num_hidden_layers=args.extract_first_n_layers, cache_dir = args.cache_dir, **hf_kwargs)
                # attn_implementation="flash_attention_2",  
            else:
                try:
                    if "falcon" in args.model_name_or_path:
                        logger.info("Not Flash Attention 2!")
                        lm_q = AutoModel.from_pretrained(args.model_name_or_path, cache_dir = args.cache_dir, torch_dtype=torch.bfloat16, **hf_kwargs)
                    else:
                        logger.info("Flash Attention 2!")
                        lm_q = AutoModel.from_pretrained(args.model_name_or_path, attn_implementation="flash_attention_2", cache_dir = args.cache_dir, torch_dtype=torch.bfloat16, **hf_kwargs)
                except ValueError:
                    logger.info("Error!")
                    lm_q = AutoModel.from_pretrained(args.model_name_or_path, cache_dir = args.cache_dir, **hf_kwargs)

            if not args.share_encoder:
                lm_p = copy.deepcopy(lm_q) 
        
        if args.share_encoder:
            lm_p = lm_q

        model = cls(args=args, lm_q=lm_q, lm_p=lm_p)
        return model

    def save(self, output_dir: str):
        if self.args.do_lora:
            # Save LoRA-only parameters
            output_dir = os.path.join(output_dir, "lora")
            logger.info(f"save Lora-only parameters")
        if not self.args.share_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'passage_model'), exist_ok=True)
            logger.info(f"save bi-encoders query model into {output_dir}/query_model")
            logger.info(f"save bi-encoders passage model into {output_dir}/passage_model")
            self.query_model_save_path = os.path.join(output_dir, 'query_model') 
            self.passage_model_save_path = os.path.join(output_dir, 'passage_model')
            save_lm_p = unwrap_model(self.lm_p)
            lm_p_state_dict = save_lm_p.state_dict()
            save_lm_p.save_pretrained(self.passage_model_save_path)
            save_lm_q = unwrap_model(self.lm_q)
            lm_q_state_dict = save_lm_q.state_dict()
            save_lm_q.save_pretrained(self.query_model_save_path)

        else:
            save_lm_q = unwrap_model(self.lm_q)
            lm_q_state_dict = save_lm_q.state_dict()
            os.makedirs(os.path.join(output_dir, 'encoder'), exist_ok=True)
            self.query_model_save_path = os.path.join(output_dir, 'encoder')
            save_lm_q.save_pretrained(self.query_model_save_path)
