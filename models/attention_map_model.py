import os
import copy
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

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
    full_contrastive_scores_and_labels_with_neg, 
    full_contrastive_scores_and_labels_with_neg_add,
    angle_loss, 
    print_trainable_parameters
)
from .pooling import pool
import torch.distributed as dist

from mteb.encoder_interface import PromptType
from dataloaders.basic_dataloader import IndexedDataset
from tqdm import tqdm
from .basic_model import BasicOutput


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
        if len(sentence) > 0 and sentence[-1] not in '.?!"\'': 
                sentence += '.'
        sentence = sentence.replace('"', '\'')
        prompts.append(prompt_method.replace('*sent 0*', sentence).replace('_', ' ').strip())
    return prompts

class AttentionMapModel(nn.Module):
    def __init__(self, args,
                 lm_i: PreTrainedModel,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel):
        super().__init__()
        self.uni_encoder = args.share_encoder
        self.lm_i = lm_i
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
            s = torch.sum(hidden_state * mask.unsquseeze(-1).float(), dim=1)
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
                instruction: Dict[str, Tensor] = None,
                query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None):
        assert self.args.process_index >= 0
        scores, labels, q_reps, p_reps, all_scores, all_labels = self._compute_scores(instruction, query, passage)
        start = self.args.process_index * q_reps.shape[0]

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

        total_n_psg = self.args.world_size * q_reps.shape[0]

        return BasicOutput(loss=loss, q_reps=q_reps, p_reps=p_reps,
                               labels=labels.contiguous(),
                               scores=scores[:, :total_n_psg].contiguous())

    def _compute_scores(self, 
                        instruction: Dict[str, Tensor] = None,
                        query: Dict[str, Tensor] = None,
                        passage: Dict[str, Tensor] = None) -> Tuple:
        
        i_reps_all = self._encode_all(self.lm_i, instruction) # (bs, seq_len_i, dim)
        q_reps_all = self._encode_all(self.lm_q, query)       # (bs, seq_len_q, dim)
        p_reps = self._encode(self.lm_p, passage)           # (bs * n_passages, dim) -> Check shape based on dataloader

        if i_reps_all is None or q_reps_all is None:
             raise ValueError("Instruction and query inputs cannot be None")

        batch_size = q_reps_all.shape[0]
        device = q_reps_all.device
        hidden_dim = q_reps_all.shape[-1]

        # --- 1. Compute Positive IQ Reps ---
        # Calculate attention between matching i_reps and q_reps
        # i_reps_all @ q_reps_all.transpose: (bs, seq_len_i, dim) @ (bs, dim, seq_len_q) -> (bs, seq_len_i, seq_len_q)
        iq_attention_pos = torch.matmul(i_reps_all, q_reps_all.transpose(-1, -2)) / (hidden_dim ** 0.5)
        i_mask_pos = instruction["attention_mask"][:, :, None].to(device) # (bs, seq_len_i, 1)
        q_mask_pos = query["attention_mask"][:, None, :].to(device)       # (bs, 1, seq_len_q)
        iq_mask_pos = (i_mask_pos * q_mask_pos).float()                   # (bs, seq_len_i, seq_len_q)
        iq_attention_pos = iq_attention_pos.masked_fill(iq_mask_pos == 0, -1e8) # Apply mask
        iq_attention_prob_pos = nn.functional.softmax(iq_attention_pos, dim=-1) # (bs, seq_len_i, seq_len_q)
        
        # Attend: iq_attention_prob_pos @ q_reps_all
        # (bs, seq_len_i, seq_len_q) @ (bs, seq_len_q, dim) -> (bs, seq_len_i, dim)
        iq_emb_pos = torch.matmul(iq_attention_prob_pos, q_reps_all) 
        
        # Pool the attended embeddings using instruction mask
        pos_iq_reps = pool(last_hidden_states=iq_emb_pos, attention_mask=instruction['attention_mask'], pool_type='avg') # (bs, dim)
        pos_iq_reps = self.linear_pooler(pos_iq_reps).contiguous() # (bs, dim)


        # --- 2. Compute Negative IQ Reps ---
        # Number of instructions per query (1 positive + k negatives)
        neg_count = batch_size // self.args.div_neg_batch 
        if neg_count == 0: # Handle cases where batch_size is smaller than div_neg_batch
            neg_count = 1
        
        all_neg_iq_reps_list = []
        all_indices = list(range(batch_size))

        # Generate indices for negative sampling for the entire batch
        neg_indices_map = {}
        for i in range(batch_size):
            possible_negs = [j for j in all_indices if j != i]
            # Ensure we sample exactly neg_count-1 negatives
            num_negs_to_sample = min(neg_count - 1, len(possible_negs))
            sampled_negs = random.sample(possible_negs, num_negs_to_sample)
            # Indices for this query: positive first, then negatives
            current_indices = [i] + sampled_negs
            # Pad with the positive index if we didn't get enough unique negatives
            while len(current_indices) < neg_count:
                 current_indices.append(i) 
            neg_indices_map[i] = current_indices[:neg_count] # Ensure exactly neg_count

        # Loop through each query to compute its reps against neg_count instructions
        for i in range(batch_size):
            q_rep_i = q_reps_all[i].unsqueeze(0) # (1, seq_len_q, dim)
            q_mask_i = query["attention_mask"][i].unsqueeze(0) # (1, seq_len_q)

            # Get the neg_count instructions (positive + negatives) for this query
            current_indices = neg_indices_map[i]
            # Ensure indices are on the correct device if they are tensors
            if isinstance(current_indices, torch.Tensor):
                 current_indices = current_indices.to(device)
            
            i_reps_neg = i_reps_all[current_indices] # (neg_count, seq_len_i, dim)
            i_mask_neg = instruction["attention_mask"][current_indices] # (neg_count, seq_len_i)

            # Repeat query representation for batch computation with neg_count instructions
            q_rep_i_expanded = q_rep_i.expand(neg_count, -1, -1) # (neg_count, seq_len_q, dim)
            q_mask_i_expanded = q_mask_i.expand(neg_count, -1) # (neg_count, seq_len_q)

            # Calculate attention: i_reps_neg @ q_rep_i_expanded.T
            # Shapes: (neg_count, seq_len_i, dim) @ (neg_count, dim, seq_len_q) -> (neg_count, seq_len_i, seq_len_q)
            iq_attention_neg = torch.matmul(i_reps_neg, q_rep_i_expanded.transpose(-1, -2)) / (hidden_dim ** 0.5)

            # Apply mask using broadcasted instruction and query masks
            i_mask_neg_b = i_mask_neg[:, :, None].to(device) # (neg_count, seq_len_i, 1)
            q_mask_i_expanded_b = q_mask_i_expanded[:, None, :].to(device) # (neg_count, 1, seq_len_q)
            iq_mask_neg = (i_mask_neg_b * q_mask_i_expanded_b).float() # (neg_count, seq_len_i, seq_len_q)
            iq_attention_neg = iq_attention_neg.masked_fill(iq_mask_neg == 0, -1e8) # Apply mask

            # Softmax over the query sequence length dimension
            iq_attention_prob_neg = nn.functional.softmax(iq_attention_neg, dim=-1) # (neg_count, seq_len_i, seq_len_q)

            # Attend: iq_attention_prob_neg @ q_rep_i_expanded
            # Shapes: (neg_count, seq_len_i, seq_len_q) @ (neg_count, seq_len_q, dim) -> (neg_count, seq_len_i, dim)
            iq_emb_neg = torch.matmul(iq_attention_prob_neg, q_rep_i_expanded)

            # Pool the attended embeddings using the instruction mask
            # Pool expects (batch_size, seq_len, hidden_dim), here batch_size is neg_count
            iq_reps_neg = pool(last_hidden_states=iq_emb_neg, attention_mask=i_mask_neg, pool_type='avg') # (neg_count, dim)

            # Apply linear pooler
            iq_reps_neg = self.linear_pooler(iq_reps_neg) # (neg_count, dim)

            all_neg_iq_reps_list.append(iq_reps_neg)

        # Stack results for all queries into a single tensor
        # Shape: (bs, neg_count, dim)
        neg_iq_reps = torch.stack(all_neg_iq_reps_list) 

        # --- 3. Normalization (applied to pooled representations) ---
        if self.args.l2_normalize:
            pos_iq_reps = F.normalize(pos_iq_reps, p=2, dim=-1)
            # Normalize each of the (bs * neg_count) vectors independently
            neg_iq_reps = F.normalize(neg_iq_reps, p=2, dim=-1)

        # --- 4. Gather tensors across devices ---
        # Gather positive query representations
        all_pos_iq_reps = dist_gather_tensor(pos_iq_reps) # (world_size * bs, dim)
        # Gather passage representations
        all_p_reps = dist_gather_tensor(p_reps)         # (world_size * bs * n_passages, dim) - Verify shape based on input p_reps
        # Gather negative query representations
        all_neg_iq_reps = dist_gather_tensor(neg_iq_reps) # (world_size * bs, neg_count, dim) - Shape should be correct if gather concatenates on dim 0

        # --- 5. Compute Scores using function that handles negative queries ---
        # Ensure `full_contrastive_scores_and_labels_with_neg_add` exists and is imported
        # It should accept `query`, `key`, and `neg_query` arguments
        all_scores, all_labels= full_contrastive_scores_and_labels_with_neg(
            query=all_pos_iq_reps,          # (total_bs, dim)
            key=all_p_reps,                 # (total_bs * n_passages, dim)
            neg_query=all_neg_iq_reps,      # (total_bs, neg_count, dim)
            use_all_pairs=self.args.full_contrastive_loss,
            contrast_mode=self.args.contrast_mode,
            div_neg_batch=self.args.div_neg_batch # Pass if needed by the loss function
        )
        # The shape of all_scores depends on the implementation of the loss function, 
        # typically (total_bs, total_keys + total_bs * (neg_count-1)) if negatives are added to score matrix columns


        # --- 6. Apply Temperature Scaling ---
        if self.args.l2_normalize: # Apply scaling only if normalized
            if self.args.t_warmup and self.trainer is not None and hasattr(self.trainer, 'state'):
                # Check global_step availability
                current_step = self.trainer.state.global_step if hasattr(self.trainer.state, 'global_step') else 0
                warmup_steps = self.args.warmup_steps if self.args.warmup_steps > 0 else 1 # Avoid division by zero
                scale_factor = min(1.0, current_step / warmup_steps)
                effective_temp = self.args.t # Assuming t is the target temperature
                scale = 1.0 / (effective_temp * scale_factor + 1e-6) # Example: Inverse linear scaling from high temp to target temp
                # Or use the original logic if preferred:
                # scale = 1.0 / self.args.t * min(1.0, current_step / warmup_steps)
                # scale = max(1.0, scale) # This still seems counter-intuitive for warmup
            else:
                scale = float(1.0 / (1e-4 + self.args.t))
            all_scores = all_scores * scale

        # --- 7. Select local scores and labels ---
        # Calculate the start index for the current process
        start = self.args.process_index * batch_size # Use local batch size
        # Create indices for the local batch
        local_query_indices = torch.arange(start, start + batch_size, dtype=torch.long, device=device)
        
        # Select scores corresponding to local queries
        # Shape depends on all_scores output, likely (local_bs, total_num_targets)
        scores = all_scores.index_select(dim=0, index=local_query_indices)
        # Select labels corresponding to local positive pairs
        labels = all_labels.index_select(dim=0, index=local_query_indices) # Shape: (local_bs,)
        # neg_labels = all_neg_labels.index_select(dim=0, index=local_query_indices) # Shape: (local_bs,)

        # --- 8. Return ---
        # Return pooled positive query reps, original passage reps, local scores/labels, and all scores/labels
        # Use pos_iq_reps (shape: bs, dim) instead of original token-level q_reps
        return scores, labels, pos_iq_reps, p_reps, all_scores, all_labels

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
    
    def _encode_all(self, encoder: PreTrainedModel, input_dict: dict) -> Optional[torch.Tensor]:
        if not input_dict:
            return None
        outputs = encoder(**{k: v for k, v in input_dict.items() if k not in ['kd_labels']}, return_dict=True)
        hidden_state = outputs.last_hidden_state
        attention_mask = input_dict['attention_mask']
        embeds = hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return embeds.contiguous()
    
    def encode(
        self,
        sentences: list[str],
        task_name: str,
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

        file_path = '/path/to/queries.json'
        query_list = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                query_list.append(data['text'])
        query_list = list(set(query_list))
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
            if prompt_type == PromptType.query:
                instruction_batch, query_batch = [], []
                for sent in sentences_batch:
                    flag = False
                    for query in query_list:
                        if query in sent:
                            instruction = sent.replace(query, '')[1:]
                            flag = True
                            break
                    if not flag:
                        raise ValueError('No instruction found for query:', sent)
                    instruction_batch.append(instruction)
                    query_batch.append(query)

                if self.args.padding_side == "left":
                    self.tokenizer.padding_side = "left"
                else:
                    self.tokenizer.padding_side = "right"
                instruction_inputs = self.tokenizer(
                    instruction_batch,
                    **self.tokenizer_config,
                )
                query_inputs = self.tokenizer(
                    query_batch,
                    **self.tokenizer_config,
                )
                with torch.no_grad():
                    instruction_inputs = {k: v.to(self.lm_q.device) for k, v in instruction_inputs.items()}
                    query_inputs = {k: v.to(self.lm_q.device) for k, v in query_inputs.items()}
                    instruction_embeddings = self._encode_all(self.lm_i, instruction_inputs)
                    query_embeddings = self._encode_all(self.lm_q, query_inputs)

                    i_embed = instruction_embeddings
                    q_embed = query_embeddings

                    iq_att = torch.matmul(i_embed, q_embed.transpose(-1, -2)) / (i_embed.size(-1) ** 0.5)
                    i_att_mask = instruction_inputs["attention_mask"][:,:,None].to(iq_att.device)
                    q_att_mask = query_inputs["attention_mask"][:,None,:].to(iq_att.device)
                    iq_att_mask = (i_att_mask * q_att_mask).float()
                    iq_att = iq_att.masked_fill(iq_att_mask == 0, -1e-8)
                    iq_att = nn.functional.softmax(iq_att, dim=-1)
                    embeddings = torch.matmul(iq_att, q_embed)
                    embeddings = pool(last_hidden_states=embeddings, attention_mask=instruction_inputs["attention_mask"], pool_type='avg')
                    embeddings = self.linear_pooler(embeddings)
                    if self.args.l2_normalize:
                        embeddings = F.normalize(embeddings, dim=-1)
            else:
                model_inputs = self.tokenizer(
                    sentences_batch,
                    **self.tokenizer_config,
                )
                with torch.no_grad():
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
                lm_q = AutoModel.from_pretrained(_qry_model_path, cache_dir = args.cache_dir, attn_implementation="flash_attention_2", trust_remote_code=True, **hf_kwargs)
                #  attn_implementation="flash_attention_2",
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModel.from_pretrained(_psg_model_path, cache_dir = args.cache_dir, attn_implementation="flash_attention_2", trust_remote_code=True, **hf_kwargs)
                if args.extract_first_n_layers > 0:
                    raise Exception(" Not implemented extract layers for bi-encoder")
            else:
                if args.extract_first_n_layers > 0:
                    logger.info(f"load first {args.extract_first_n_layers} transformers layers from {args.model_name_or_path}")
                    lm_q = AutoModel.from_pretrained(args.model_name_or_path, attn_implementation="flash_attention_2",  num_hidden_layers=args.extract_first_n_layers, cache_dir = args.cache_dir, **hf_kwargs)
                    # attn_implementation="flash_attention_2",
                else:
                    try:
                        logger.info(f'loading shared model weight from {args.model_name_or_path}')
                        lm_q = AutoModel.from_pretrained(args.model_name_or_path, attn_implementation="flash_attention_2", cache_dir = args.cache_dir, **hf_kwargs)
                    except ValueError:
                        logger.info("Error!")
                        lm_q = AutoModel.from_pretrained(args.model_name_or_path, cache_dir = args.cache_dir, **hf_kwargs)
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
                lm_i = copy.deepcopy(lm_q)
        
        if args.share_encoder:
            lm_p = lm_q
            lm_i = lm_q

        model = cls(args=args, lm_i=lm_i, lm_q=lm_q, lm_p=lm_p)
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
            self.instruction_model_save_path = os.path.join(output_dir, 'instruction_model')
            self.query_model_save_path = os.path.join(output_dir, 'query_model') 
            self.passage_model_save_path = os.path.join(output_dir, 'passage_model')
            save_lm_p = unwrap_model(self.lm_p)
            lm_p_state_dict = save_lm_p.state_dict()
            save_lm_p.save_pretrained(self.passage_model_save_path)

            save_lm_q = unwrap_model(self.lm_q)
            lm_q_state_dict = save_lm_q.state_dict()
            save_lm_q.save_pretrained(self.query_model_save_path)

            save_lm_i = unwrap_model(self.lm_i)
            lm_i_state_dict = save_lm_i.state_dict()
            save_lm_i.save_pretrained(self.instruction_model_save_path)

        else:

            save_lm_q = unwrap_model(self.lm_q)
            lm_q_state_dict = save_lm_q.state_dict()
            os.makedirs(os.path.join(output_dir, 'encoder'), exist_ok=True)
            self.query_model_save_path = os.path.join(output_dir, 'encoder')
            save_lm_q.save_pretrained(self.query_model_save_path)


class AttentionMapAddModel(nn.Module):
    def __init__(self, args,
                 lm_i: PreTrainedModel,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel):
        super().__init__()
        self.uni_encoder = args.share_encoder
        self.lm_i = lm_i
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
            s = torch.sum(hidden_state * mask.unsquseeze(-1).float(), dim=1)
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
                instruction: Dict[str, Tensor] = None,
                query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None):
        assert self.args.process_index >= 0
        scores, labels, neg_labels, q_reps, p_reps, all_scores, all_labels, all_neg_labels = self._compute_scores(instruction, query, passage)
        start = self.args.process_index * q_reps.shape[0]


        if not self.args.do_kd_biencoder:
            # training biencoder from scratch
            if self.args.use_scaled_loss:
                if self.args.contrast_mode == "qk_with_neg" or self.args.contrast_mode == "kq_with_neg":
                    # Split scores into two parts: regular scores and negative scores
                    regular_scores = all_scores[:, :self.args.world_size * len(query['input_ids'])]
                    neg_scores = all_scores[:, self.args.world_size * len(query['input_ids']):]
                    
                    # Assert to confirm dimensions
                    assert neg_scores.shape[1] == len(query['input_ids']) // self.args.div_neg_batch, f"Expected neg_scores dimension {len(query['input_ids']) // self.args.div_neg_batch}, got {neg_scores.shape[1]}"
                    
                    # Calculate loss for each part and sum them
                    loss_regular = self.cross_entropy(regular_scores, all_labels)
                    loss_neg = self.cross_entropy(neg_scores, all_neg_labels)
                    loss = loss_regular + loss_neg
                elif self.args.contrast_mode == "no_trick_with_neg":
                    # Split scores into three parts
                    part1 = all_scores[:, :self.args.world_size * len(query['input_ids'])]
                    part2 = all_scores[:, self.args.world_size * len(query['input_ids']):2 * self.args.world_size * len(query['input_ids'])]
                    part3 = all_scores[:, 2 * self.args.world_size * len(query['input_ids']):]
                    
                    # Assert to confirm dimensions
                    assert part3.shape[1] == len(query['input_ids']) // self.args.div_neg_batch, f"Expected part3 dimension {len(query['input_ids']) // self.args.div_neg_batch}, got {part3.shape[1]}"
                    
                    # Calculate loss for each part and sum them
                    loss1 = self.cross_entropy(part1, all_labels)
                    loss2 = self.cross_entropy(part2, all_labels)
                    loss3 = self.cross_entropy(part3, all_neg_labels)
                    loss = loss1 + loss2 + loss3
                elif self.args.contrast_mode == "same_tower_with_neg":
                    # Split scores into five parts
                    part1 = all_scores[:, :self.args.world_size * len(query['input_ids'])]
                    part2 = all_scores[:, self.args.world_size * len(query['input_ids']):2 * self.args.world_size * len(query['input_ids'])]
                    part3 = all_scores[:, 2 * self.args.world_size * len(query['input_ids']):3 * self.args.world_size * len(query['input_ids'])]
                    part4 = all_scores[:, 3 * self.args.world_size * len(query['input_ids']):4 * self.args.world_size * len(query['input_ids'])]
                    part5 = all_scores[:, 4 * self.args.world_size * len(query['input_ids']):]
                    
                    # Assert to confirm dimensions
                    assert part5.shape[1] == len(query['input_ids']) // self.args.div_neg_batch, f"Expected part5 dimension {len(query['input_ids']) // self.args.div_neg_batch}, got {part5.shape[1]}"
                    
                    # Calculate loss for each part and sum them
                    loss1 = self.cross_entropy(part1, all_labels)
                    loss2 = self.cross_entropy(part2, all_labels)
                    loss3 = self.cross_entropy(part3, all_labels)
                    loss4 = self.cross_entropy(part4, all_labels)
                    loss5 = self.cross_entropy(part5, all_neg_labels)
                    loss = loss1 + loss2 + loss3 + loss4 + loss5

                elif self.args.contrast_mode == "only_neg":
                    # Directly calculate cross entropy for only_neg
                    labels = neg_labels
                    loss = self.cross_entropy(all_scores, all_neg_labels)
                else:
                    raise ValueError(f"Unknown contrast_mode: {self.args.contrast_mode}")
                
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

    def _compute_scores(self, 
                        instruction: Dict[str, Tensor] = None,
                        query: Dict[str, Tensor] = None,
                        passage: Dict[str, Tensor] = None) -> Tuple:
        
        i_reps_all = self._encode_all(self.lm_i, instruction) # (bs, seq_len_i, dim)
        q_reps_all = self._encode_all(self.lm_q, query)       # (bs, seq_len_q, dim)
        p_reps = self._encode(self.lm_p, passage)           # (bs * n_passages, dim) -> Check shape based on dataloader

        if i_reps_all is None or q_reps_all is None:
             raise ValueError("Instruction and query inputs cannot be None")

        batch_size = q_reps_all.shape[0]
        device = q_reps_all.device
        hidden_dim = q_reps_all.shape[-1]

        # --- 1. Compute Positive IQ Reps ---
        # Calculate attention between matching i_reps and q_reps
        # i_reps_all @ q_reps_all.transpose: (bs, seq_len_i, dim) @ (bs, dim, seq_len_q) -> (bs, seq_len_i, seq_len_q)
        iq_attention_pos = torch.matmul(i_reps_all, q_reps_all.transpose(-1, -2)) / (hidden_dim ** 0.5)
        i_mask_pos = instruction["attention_mask"][:, :, None].to(device) # (bs, seq_len_i, 1)
        q_mask_pos = query["attention_mask"][:, None, :].to(device)       # (bs, 1, seq_len_q)
        iq_mask_pos = (i_mask_pos * q_mask_pos).float()                   # (bs, seq_len_i, seq_len_q)
        iq_attention_pos = iq_attention_pos.masked_fill(iq_mask_pos == 0, -1e8) # Apply mask
        iq_attention_prob_pos = nn.functional.softmax(iq_attention_pos, dim=-1) # (bs, seq_len_i, seq_len_q)
        
        # Attend: iq_attention_prob_pos @ q_reps_all
        # (bs, seq_len_i, seq_len_q) @ (bs, seq_len_q, dim) -> (bs, seq_len_i, dim)
        iq_emb_pos = torch.matmul(iq_attention_prob_pos, q_reps_all) 
        
        # Pool the attended embeddings using instruction mask
        pos_iq_reps = pool(last_hidden_states=iq_emb_pos, attention_mask=instruction['attention_mask'], pool_type='avg') # (bs, dim)
        pos_iq_reps = self.linear_pooler(pos_iq_reps).contiguous() # (bs, dim)


        # --- 2. Compute Negative IQ Reps ---
        # Number of instructions per query (1 positive + k negatives)
        neg_count = batch_size // self.args.div_neg_batch 
        if neg_count == 0: # Handle cases where batch_size is smaller than div_neg_batch
            neg_count = 1
        
        all_neg_iq_reps_list = []
        all_indices = list(range(batch_size))

        # Generate indices for negative sampling for the entire batch
        neg_indices_map = {}
        for i in range(batch_size):
            possible_negs = [j for j in all_indices if j != i]
            # Ensure we sample exactly neg_count-1 negatives
            num_negs_to_sample = min(neg_count - 1, len(possible_negs))
            sampled_negs = random.sample(possible_negs, num_negs_to_sample)
            # Indices for this query: positive first, then negatives
            current_indices = [i] + sampled_negs
            # Pad with the positive index if we didn't get enough unique negatives
            while len(current_indices) < neg_count:
                 current_indices.append(i) 
            neg_indices_map[i] = current_indices[:neg_count] # Ensure exactly neg_count

        # Loop through each query to compute its reps against neg_count instructions
        for i in range(batch_size):
            q_rep_i = q_reps_all[i].unsqueeze(0) # (1, seq_len_q, dim)
            q_mask_i = query["attention_mask"][i].unsqueeze(0) # (1, seq_len_q)

            # Get the neg_count instructions (positive + negatives) for this query
            current_indices = neg_indices_map[i]
            # Ensure indices are on the correct device if they are tensors
            if isinstance(current_indices, torch.Tensor):
                 current_indices = current_indices.to(device)
            
            i_reps_neg = i_reps_all[current_indices] # (neg_count, seq_len_i, dim)
            i_mask_neg = instruction["attention_mask"][current_indices] # (neg_count, seq_len_i)

            # Repeat query representation for batch computation with neg_count instructions
            q_rep_i_expanded = q_rep_i.expand(neg_count, -1, -1) # (neg_count, seq_len_q, dim)
            q_mask_i_expanded = q_mask_i.expand(neg_count, -1) # (neg_count, seq_len_q)

            # Calculate attention: i_reps_neg @ q_rep_i_expanded.T
            # Shapes: (neg_count, seq_len_i, dim) @ (neg_count, dim, seq_len_q) -> (neg_count, seq_len_i, seq_len_q)
            iq_attention_neg = torch.matmul(i_reps_neg, q_rep_i_expanded.transpose(-1, -2)) / (hidden_dim ** 0.5)

            # Apply mask using broadcasted instruction and query masks
            i_mask_neg_b = i_mask_neg[:, :, None].to(device) # (neg_count, seq_len_i, 1)
            q_mask_i_expanded_b = q_mask_i_expanded[:, None, :].to(device) # (neg_count, 1, seq_len_q)
            iq_mask_neg = (i_mask_neg_b * q_mask_i_expanded_b).float() # (neg_count, seq_len_i, seq_len_q)
            iq_attention_neg = iq_attention_neg.masked_fill(iq_mask_neg == 0, -1e8) # Apply mask

            # Softmax over the query sequence length dimension
            iq_attention_prob_neg = nn.functional.softmax(iq_attention_neg, dim=-1) # (neg_count, seq_len_i, seq_len_q)

            # Attend: iq_attention_prob_neg @ q_rep_i_expanded
            # Shapes: (neg_count, seq_len_i, seq_len_q) @ (neg_count, seq_len_q, dim) -> (neg_count, seq_len_i, dim)
            iq_emb_neg = torch.matmul(iq_attention_prob_neg, q_rep_i_expanded)

            # Pool the attended embeddings using the instruction mask
            # Pool expects (batch_size, seq_len, hidden_dim), here batch_size is neg_count
            iq_reps_neg = pool(last_hidden_states=iq_emb_neg, attention_mask=i_mask_neg, pool_type='avg') # (neg_count, dim)

            # Apply linear pooler
            iq_reps_neg = self.linear_pooler(iq_reps_neg) # (neg_count, dim)

            all_neg_iq_reps_list.append(iq_reps_neg)

        # Stack results for all queries into a single tensor
        # Shape: (bs, neg_count, dim)
        neg_iq_reps = torch.stack(all_neg_iq_reps_list) 

        # --- 3. Normalization (applied to pooled representations) ---
        if self.args.l2_normalize:
            pos_iq_reps = F.normalize(pos_iq_reps, p=2, dim=-1)
            # Normalize each of the (bs * neg_count) vectors independently
            neg_iq_reps = F.normalize(neg_iq_reps, p=2, dim=-1)

        # --- 4. Gather tensors across devices ---
        # Gather positive query representations
        all_pos_iq_reps = dist_gather_tensor(pos_iq_reps) # (world_size * bs, dim)
        # Gather passage representations
        all_p_reps = dist_gather_tensor(p_reps)         # (world_size * bs * n_passages, dim) - Verify shape based on input p_reps
        # Gather negative query representations
        all_neg_iq_reps = dist_gather_tensor(neg_iq_reps) # (world_size * bs, neg_count, dim) - Shape should be correct if gather concatenates on dim 0

        # --- 5. Compute Scores using function that handles negative queries ---
        # Ensure `full_contrastive_scores_and_labels_with_neg_add` exists and is imported
        # It should accept `query`, `key`, and `neg_query` arguments
        all_scores, all_labels, all_neg_labels = full_contrastive_scores_and_labels_with_neg_add(
            query=all_pos_iq_reps,          # (total_bs, dim)
            key=all_p_reps,                 # (total_bs * n_passages, dim)
            neg_query=all_neg_iq_reps,      # (total_bs, neg_count, dim)
            use_all_pairs=self.args.full_contrastive_loss,
            contrast_mode=self.args.contrast_mode,
            div_neg_batch=self.args.div_neg_batch # Pass if needed by the loss function
        )
        # The shape of all_scores depends on the implementation of the loss function, 
        # typically (total_bs, total_keys + total_bs * (neg_count-1)) if negatives are added to score matrix columns


        # --- 6. Apply Temperature Scaling ---
        if self.args.l2_normalize: # Apply scaling only if normalized
            if self.args.t_warmup and self.trainer is not None and hasattr(self.trainer, 'state'):
                # Check global_step availability
                current_step = self.trainer.state.global_step if hasattr(self.trainer.state, 'global_step') else 0
                warmup_steps = self.args.warmup_steps if self.args.warmup_steps > 0 else 1 # Avoid division by zero
                scale_factor = min(1.0, current_step / warmup_steps)
                effective_temp = self.args.t # Assuming t is the target temperature
                scale = 1.0 / (effective_temp * scale_factor + 1e-6) # Example: Inverse linear scaling from high temp to target temp
                # Or use the original logic if preferred:
                # scale = 1.0 / self.args.t * min(1.0, current_step / warmup_steps)
                # scale = max(1.0, scale) # This still seems counter-intuitive for warmup
            else:
                scale = float(1.0 / (1e-4 + self.args.t))
            all_scores = all_scores * scale

        # --- 7. Select local scores and labels ---
        # Calculate the start index for the current process
        start = self.args.process_index * batch_size # Use local batch size
        # Create indices for the local batch
        local_query_indices = torch.arange(start, start + batch_size, dtype=torch.long, device=device)
        
        # Select scores corresponding to local queries
        # Shape depends on all_scores output, likely (local_bs, total_num_targets)
        scores = all_scores.index_select(dim=0, index=local_query_indices)
        # Select labels corresponding to local positive pairs
        labels = all_labels.index_select(dim=0, index=local_query_indices) # Shape: (local_bs,)
        neg_labels = all_neg_labels.index_select(dim=0, index=local_query_indices) # Shape: (local_bs,)

        # --- 8. Return ---
        # Return pooled positive query reps, original passage reps, local scores/labels, and all scores/labels
        # Use pos_iq_reps (shape: bs, dim) instead of original token-level q_reps
        return scores, labels, neg_labels, pos_iq_reps, p_reps, all_scores, all_labels, all_neg_labels

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
    
    def _encode_all(self, encoder: PreTrainedModel, input_dict: dict) -> Optional[torch.Tensor]:
        if not input_dict:
            return None
        outputs = encoder(**{k: v for k, v in input_dict.items() if k not in ['kd_labels']}, return_dict=True)
        hidden_state = outputs.last_hidden_state
        attention_mask = input_dict['attention_mask']
        embeds = hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return embeds.contiguous()
    
    def encode(
        self,
        sentences: list[str],
        task_name: str,
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
        file_path = '/path/to/queries.json'
        query_list = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                query_list.append(data['text'])
        query_list = list(set(query_list))
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
            if prompt_type == PromptType.query:
                instruction_batch, query_batch = [], []
                for sent in sentences_batch:
                    flag = False

                    for query in query_list:
                        if query in sent:
                            instruction = sent.replace(query, '')[1:]
                            flag = True
                            break

                    if not flag:
                        raise ValueError('No instruction found for query:', sent)
                    instruction_batch.append(instruction)
                    query_batch.append(query)

                if self.args.padding_side == "left":
                    self.tokenizer.padding_side = "left"
                else:
                    self.tokenizer.padding_side = "right"
                instruction_inputs = self.tokenizer(
                    instruction_batch,
                    **self.tokenizer_config,
                )
                query_inputs = self.tokenizer(
                    query_batch,
                    **self.tokenizer_config,
                )
                with torch.no_grad():
                    instruction_inputs = {k: v.to(self.lm_q.device) for k, v in instruction_inputs.items()}
                    query_inputs = {k: v.to(self.lm_q.device) for k, v in query_inputs.items()}
                    instruction_embeddings = self._encode_all(self.lm_i, instruction_inputs)
                    query_embeddings = self._encode_all(self.lm_q, query_inputs)
                    i_embed = instruction_embeddings
                    q_embed = query_embeddings
                    iq_att = torch.matmul(i_embed, q_embed.transpose(-1, -2)) / (i_embed.size(-1) ** 0.5)
                    i_att_mask = instruction_inputs["attention_mask"][:,:,None].to(iq_att.device)
                    q_att_mask = query_inputs["attention_mask"][:,None,:].to(iq_att.device)
                    iq_att_mask = (i_att_mask * q_att_mask).float()
                    iq_att = iq_att.masked_fill(iq_att_mask == 0, -1e-8)
                    iq_att = nn.functional.softmax(iq_att, dim=-1)
                    embeddings = torch.matmul(iq_att, q_embed)
                    embeddings = pool(last_hidden_states=embeddings, attention_mask=instruction_inputs["attention_mask"], pool_type='avg')
                    embeddings = self.linear_pooler(embeddings)
                    if self.args.l2_normalize:
                        embeddings = F.normalize(embeddings, dim=-1)
            else:
                model_inputs = self.tokenizer(
                    sentences_batch,
                    **self.tokenizer_config,
                )
                with torch.no_grad():
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
                lm_q = AutoModel.from_pretrained(_qry_model_path, cache_dir = args.cache_dir, attn_implementation="flash_attention_2", trust_remote_code=True, **hf_kwargs)
                #  attn_implementation="flash_attention_2",
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModel.from_pretrained(_psg_model_path, cache_dir = args.cache_dir, attn_implementation="flash_attention_2", trust_remote_code=True, **hf_kwargs)
                if args.extract_first_n_layers > 0:
                    raise Exception(" Not implemented extract layers for bi-encoder")
            else:
                if args.extract_first_n_layers > 0:
                    logger.info(f"load first {args.extract_first_n_layers} transformers layers from {args.model_name_or_path}")
                    lm_q = AutoModel.from_pretrained(args.model_name_or_path, attn_implementation="flash_attention_2",  num_hidden_layers=args.extract_first_n_layers, cache_dir = args.cache_dir, **hf_kwargs)
                    # attn_implementation="flash_attention_2",
                else:
                    try:
                        logger.info(f'loading shared model weight from {args.model_name_or_path}')
                        lm_q = AutoModel.from_pretrained(args.model_name_or_path, attn_implementation="flash_attention_2", cache_dir = args.cache_dir, **hf_kwargs)
                    except ValueError:
                        logger.info("Error!")
                        lm_q = AutoModel.from_pretrained(args.model_name_or_path, cache_dir = args.cache_dir, **hf_kwargs)
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
                lm_i = copy.deepcopy(lm_q)
        
        if args.share_encoder:
            lm_p = lm_q
            lm_i = lm_q
        # for param in lm_p.parameters():
        #     param.requires_grad = False
        # for param in lm_q.parameters():
        #     param.requires_grad = False
        model = cls(args=args, lm_i=lm_i, lm_q=lm_q, lm_p=lm_p)
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
            self.instruction_model_save_path = os.path.join(output_dir, 'instruction_model')
            self.query_model_save_path = os.path.join(output_dir, 'query_model') 
            self.passage_model_save_path = os.path.join(output_dir, 'passage_model')
            save_lm_p = unwrap_model(self.lm_p)
            lm_p_state_dict = save_lm_p.state_dict()
            save_lm_p.save_pretrained(self.passage_model_save_path)

            save_lm_q = unwrap_model(self.lm_q)
            lm_q_state_dict = save_lm_q.state_dict()
            save_lm_q.save_pretrained(self.query_model_save_path)

            save_lm_i = unwrap_model(self.lm_i)
            lm_i_state_dict = save_lm_i.state_dict()
            save_lm_i.save_pretrained(self.instruction_model_save_path)

        else:

            save_lm_q = unwrap_model(self.lm_q)
            lm_q_state_dict = save_lm_q.state_dict()
            os.makedirs(os.path.join(output_dir, 'encoder'), exist_ok=True)
            self.query_model_save_path = os.path.join(output_dir, 'encoder')
            save_lm_q.save_pretrained(self.query_model_save_path)


