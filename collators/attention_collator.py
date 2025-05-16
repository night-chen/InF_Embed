import torch
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import BatchEncoding, DataCollatorWithPadding

def _attention_unpack_doc_values(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    doc_examples = []
    for f in features:
        keys = list(f.keys())
        print('-->', keys)
        lists_per_key = len(f[keys[0]])
        print('-->', lists_per_key)
        for idx in range(lists_per_key):
            doc_examples.append({k: f[k][idx] for k in keys})
    return doc_examples

@dataclass
class AttentionCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        i_prefix, q_prefix, p_prefix = 'i_', 'q_', 'p_'
        instruction_examples = [{k[len(i_prefix):]: v for k, v in f.items() if k.startswith(i_prefix)} for f in features]
        query_examples = [{k[len(q_prefix):]: v for k, v in f.items() if k.startswith(q_prefix)} for f in features]
        passage_examples = [{k[len(p_prefix):]: v for k, v in f.items() if k.startswith(p_prefix)} for f in features]
        assert len(passage_examples) % len(query_examples) == 0, \
            '{} doc, {} queries, and {} instructions'.format(len(passage_examples), len(query_examples), len(instruction_examples))
        # already truncated during tokenization
        i_collated = self.tokenizer.pad(
            instruction_examples,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        q_collated = self.tokenizer.pad(
            query_examples,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        p_collated = self.tokenizer.pad(
            passage_examples,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )

        # merge into a single BatchEncoding by adding prefix
        for k in list(q_collated.keys()):
            q_collated[q_prefix + k] = q_collated[k]
            del q_collated[k]
        for k in list(i_collated.keys()):
            q_collated[i_prefix + k] = i_collated[k]
        for k in p_collated:
            q_collated[p_prefix + k] = p_collated[k]
        

        merged_batch_dict = q_collated
        # dummy placeholder for field "labels", won't use it to compute loss
        labels = torch.zeros(len(query_examples), dtype=torch.long)
        merged_batch_dict['labels'] = labels

        if 'kd_labels' in features[0]:
            kd_labels = torch.stack([torch.tensor(f['kd_labels']) for f in features], dim=0).float()
            merged_batch_dict['kd_labels'] = kd_labels
        return merged_batch_dict