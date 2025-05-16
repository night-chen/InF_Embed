import os.path
import time
import torch
import json
import numpy as np
import pytrec_eval
import tiktoken
from argparse import Namespace
from tqdm import tqdm,trange
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.basic_model import BasicModel, BasicAddModel
from models.map_model import MapModel, MapAddModel
from models.attention_model import AttentionModel, AttentionAddModel
from models.attention_map_model import AttentionMapModel, AttentionMapAddModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from mteb.encoder_interface import PromptType

def cut_text(text,tokenizer,threshold):
    text_ids = tokenizer(text)['input_ids']
    if len(text_ids) > threshold:
        text = tokenizer.decode(text_ids[:threshold])
    return text

def cut_text_openai(text,tokenizer,threshold=6000):
    token_ids = tokenizer.encode(text)
    if len(token_ids) > threshold:
        text = tokenizer.decode(token_ids[:threshold])
    return text


TASK_MAP = {
    'biology': 'Biology',
    'earth_science': 'Earth Science',
    'economics': 'Economics',
    'psychology': 'Psychology',
    'robotics': 'Robotics',
    'stackoverflow': 'Stack Overflow',
    'sustainable_living': 'Sustainable Living',
}

def add_instruct_concatenate(texts,task,instruction):
    return [instruction.format(task=task)+t for t in texts]

def add_instruct_concatenate_reverse(texts,task,instruction):
    return [instruction.format(Query=t,task=task) for t in texts]

def add_instruct_list(texts,task,instruction):
    return [[instruction.format(task=task),t] for t in texts]

def last_token_pool(last_hidden_states,attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_scores(query_ids,doc_ids,scores,excluded_ids):
    assert len(scores)==len(query_ids),f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0])==len(doc_ids),f"{len(scores[0])}, {len(doc_ids)}"
    emb_scores = {}
    for query_id,doc_scores in zip(query_ids,scores):
        cur_scores = {}
        assert len(excluded_ids[query_id])==0 or (isinstance(excluded_ids[query_id][0], str) and isinstance(excluded_ids[query_id], list))
        for did,s in zip(doc_ids,doc_scores):
            cur_scores[str(did)] = s
        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A":
                cur_scores.pop(did)
        cur_scores = sorted(cur_scores.items(),key=lambda x:x[1],reverse=True)[:1000]
        emb_scores[str(query_id)] = {}
        for pair in cur_scores:
            emb_scores[str(query_id)][pair[0]] = pair[1]
    return emb_scores

@torch.no_grad()
def retrieval_if(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    model_list = ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct",
                  "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct",
                  "answerdotai/ModernBERT-base", "intfloat/e5-base-v2", "intfloat/e5-large-v2"]
    checkpoint_path = kwargs.get('checkpoint', 'Qwen/Qwen2.5-1.5B')
    assert checkpoint_path is not None, "Checkpoint path must be provided for retrieval_if"
    
    if checkpoint_path not in model_list:
        # Extract the model info part from the checkpoint path
        checkpoint_parts = checkpoint_path.split('/')
        for part in checkpoint_parts:
            if 'checkpoint-' in part:
                continue
            if any(model in part for model in ["Qwen2.5", "Llama-3.2", "ModernBERT", "e5"]):
                model_info = part
                break
        
        # Extract model_type (everything before the model name)
        if "_Qwen2.5" in model_info:
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

        if "padding_right" in model_info:
            padding_side = "right"
        else:
            padding_side = "left"

        if "div_neg_batch_" in model_info:
            div_neg_batch = int(model_info.split("_div_neg_batch_")[1])
        else:
            div_neg_batch = None
    else:
        model_type = kwargs.get('model_type', 'basic')
        model_name = checkpoint_path.split('/')[-1]
        pooling_type = kwargs.get('pooling_type', 'last')
        share_encoder = kwargs.get('share_encoder', False)
        epoch = kwargs.get('epoch', 1)
        contrast_mode = kwargs.get('contrast_mode', 'same_tower')
        reverse_mode = kwargs.get('reverse_mode', False)
        padding_side = kwargs.get('padding_side', 'right')
        div_neg_batch = kwargs.get('div_neg_batch', None)


    args = Namespace(
        model_type=model_type,
        remove_unused_columns=False,
        pooling=pooling_type,
        share_encoder=share_encoder,
        extract_first_n_layers=0,
        prompt_method="none", # Assuming no special prompt formatting within the model itself
        num_train_epochs=epoch,
        contrast_mode=contrast_mode,
        cache_dir='./cache',
        p_max_len=512,
        q_max_len=256,
        model_name_or_path=checkpoint_path,
        add_pooler=False,
        l2_normalize=False,
        padding_side=padding_side,
        div_neg_batch=div_neg_batch
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    if args.model_type == "basic" or args.model_type == "baseline_basic":
        model = BasicModel.build(args)
    elif args.model_type == "map" or args.model_type == "baseline_map":
        model = BasicModel.build(args)
    # elif args.model_type == "attn" or args.model_type == "baseline_attn":
    #     model = AttentionModel.build(args)
    # elif args.model_type == "attn_map" or args.model_type == "baseline_attn_map":
    #     model = AttentionMapModel.build(args)
    elif args.model_type == "basic_add" or args.model_type == "baseline_basic_add":
        model = BasicModel.build(args)
    elif args.model_type == "map_add" or args.model_type == "baseline_map_add":
        model = BasicModel.build(args)
    # elif args.model_type == "attn_add" or args.model_type == "baseline_attn_add":
    #     model = AttentionAddModel.build(args)
    # elif args.model_type == "attn_map_add" or args.model_type == "baseline_attn_map_add":
    #     model = AttentionMapAddModel.build(args)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    model.update_tokenizer(tokenizer) # Pass the tokenizer to the model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(torch.bfloat16)
    model = model.to(device)
    model = model.eval()

    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    print(len(queries))
    batch_size = kwargs.get('encode_batch_size', 32) # Adjust default batch size as needed

    # --- Document Encoding ---
    doc_emb = None
    # Construct a more descriptive cache name
    model_info_for_cache = f"{model_type}_{model_name}_{pooling_type}_share_encoder_{share_encoder}_epoch_{epoch}_contrast_mode_{contrast_mode}_reverse_{reverse_mode}_padding_{padding_side}_div_neg_batch_{div_neg_batch}"
    # cache_name = f"{model_info_for_cache}_long_{long_context}.npy"
    # cache_path = os.path.join(cache_dir, 'doc_emb', f'{model_info_for_cache}', cache_name)

    if doc_emb is None:
        print("Encoding documents...")
        all_doc_embeddings_list = []
        # Use model.encode which handles batching and progress bar internally
        doc_emb = model.encode(
            sentences=documents,
            task_name=task, # Pass task name if needed by model's prompt formulation
            prompt_type=PromptType.passage,
            batch_size=batch_size
        )

    doc_emb = torch.tensor(doc_emb, dtype=torch.float32).to(device)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    print("doc_emb shape:", doc_emb.shape)
    # Normalization should be handled within model.encode if args.l2_normalize=True

    # --- Query Encoding ---
    print("Encoding queries...")
    # Use model.encode for queries as well
    query_emb = model.encode(
        sentences=queries,
        task_name=task, # Pass task name if needed by model's prompt formulation
        prompt_type=PromptType.query,
        batch_size=batch_size
    )
    query_emb = torch.tensor(query_emb, dtype=torch.float32).to(device)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    print("query_emb shape:", query_emb.shape)

    # --- Score Calculation ---
    # Compute cosine similarity (dot product for normalized embeddings)
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.cpu().tolist() # Move scores to CPU and convert to list

    # --- Format and Return ---
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)

RETRIEVAL_FUNCS = {
    'if': retrieval_if
}

def calculate_retrieval_metrics(results, qrels, k_values=[1, 5, 10, 25, 50, 100]):
    # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L66
    # follow evaluation from BEIR, which is just using the trec eval
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"MRR": 0}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    # https://github.com/cvangysel/pytrec_eval/blob/master/examples/simple_cut.py
    # qrels = {qid: {'pid': [0/1] (relevance label)}}
    # results = {qid: {'pid': float (retriever score)}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels,
                                               {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
        mrr["MRR"] += scores[query_id]["recip_rank"]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
    mrr["MRR"] = round(mrr["MRR"] / len(scores), 5)

    output = {**ndcg, **_map, **recall, **precision, **mrr}
    print(output)
    return output