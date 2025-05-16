import torch
from torch.utils.data import Dataset
from typing import List, Dict
from dataclasses import dataclass
from transformers import DataCollatorWithPadding

def collate_fn(batch: List[Dict], tokenizer, tokenizer_config: Dict = {}, tokenizer_query_only: bool = False):
    """
    Collate function to create a batch with dynamic padding.
    Args:
        batch: List of dictionaries containing the data.
        tokenizer: Tokenizer to use for padding.
        tokenizer_config: Configuration for the tokenizer.
        tokenizer_query_only: If True, only tokenize the query part.
    Returns:
        A dict with keys:
            - 'query': dict of padded query tensors (input_ids, attention_mask, etc.) of shape [B, max_query_len] or [B*(N+1), max_query_len]
            - 'passages': dict of padded passage tensors (input_ids, attention_mask, etc.) of shape [b*(N+1), max_passage_len]
    """
    if tokenize_query_only:
        texts = [item["text"] for item in batch]
        indices = [item["indices"] for item in batch]
        padded_queries = toeknizer(
            texts,
            **tokenizer_config
        )
        indices = torch.tensor(indices, dtype=torch.long)
        padded_queries["indices"] = indices
        return padded_queries
    
    if not tokenize_query_only and batch[0].get("passages", None):
        passage_encodings = []
        for item in batch:
            passage_encodings.extend(item['passages'])
        
        # multiple query with multiple passages
        if is_instance(batch[0]["query"], list):
            query_encodings = []
            for item in batch:
                query_encodings.extend(item["query"])
            assert len(query_encodings) == len(passage_encodings)
        else:
            query_encodings = [item["query"] for item in batch]
        padded_queres = toeknizer(
            query_encodings,
            **tokenizer_config
        )
        padded_passages = tokenizer(
            passage_encodings,
            **tokenizer_config
        )
        batch_output = {
            "query": padded_queries,
            "passages": padded_passages
        }
    return batch_output

