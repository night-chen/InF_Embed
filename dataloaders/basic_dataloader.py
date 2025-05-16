import os.path
import random

from typing import Tuple, Dict, List, Optional
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedTokenizerFast, Trainer

from logger_config import logger

def merge_batch_dict(instruction_query_batch_dict, passage_batch_dict):
    logger.debug(instruction_query_batch_dict.keys(), len(instruction_query_batch_dict['input_ids']))
    merged_dict = {'iq_{}'.format(k): v for k, v in instruction_query_batch_dict.items()}
    for k, v in passage_batch_dict.items():
        merged_dict['p_{}'.format(k)] = v
    return merged_dict


class BasicDataLoader:
    def __init__(self, args, tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.train_dataset, self.eval_dataset = self._get_transformed_datasets()
        self.train_dataset.set_transform(self._transform_func)
        self.tokenizer = tokenizer

        # use its state to decide which positives/negatives to sample
        self.trainer: Optional[Trainer] = None

    def _transform_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        current_epoch = int(self.trainer.state.epoch or 0)
        logger.debug(len(examples['query_positive_fewshot']), examples['query_positive_fewshot'][0])
        instruction_query = []
        passage = []
        for i in range(len(examples['query_positive_fewshot'])):
            # concatenate instruction and query
            instruction_query.append(f"{examples['query_positive_fewshot'][i]} {examples['instruction_positive_fewshot'][i]}".strip())
            passage.append(examples['document'][i])
            instruction_query.append(f"{examples['query_positive_fewshot'][i]} {examples['instruction_negative_fewshot'][i]}".strip())
            passage.append(examples['hard_negative_document_1'][i])
            instruction_query.append(f"{examples['query_negative_fewshot'][i]} {examples['instruction_positive_fewshot'][i]}".strip())
            passage.append(examples['hard_negative_document_2'][i])
        instruction_query_batch_dict = self.tokenizer(instruction_query, 
                                                      max_length=self.args.q_max_len,
                                                      padding=PaddingStrategy.DO_NOT_PAD,
                                                      truncation=True)
        passage_batch_dict = self.tokenizer(passage,
                                            max_length=self.args.p_max_len,
                                            padding=PaddingStrategy.DO_NOT_PAD,
                                            truncation=True)
        merged_dict = merge_batch_dict(instruction_query_batch_dict=instruction_query_batch_dict, passage_batch_dict=passage_batch_dict)
        return merged_dict

    def _get_transformed_datasets(self) -> Tuple:
        data_files = {}
        raw_dataset = load_dataset(self.args.train_file)['train']

        # train_dataset, eval_dataset = None, None
        # split the dataset into train and eval
        if self.args.do_train:
            train_dataset = raw_dataset
            eval_dataset = None
        else:
            train_dataset = None
            eval_dataset = raw_dataset

        if self.args.do_train:
            if self.args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.args.max_train_samples))
            train_dataset.set_transform(self._transform_func)

        if self.args.do_eval:
            eval_dataset.set_transform(self._transform_func)

        return train_dataset, eval_dataset

class BaseDataset(Dataset):
    """
    Base class for contrastive learning datasets.
    It expects a `self.samples` list, each element describing:
      - tokenized query
      - tokenized positive doc
      - tokenized negative docs (list of length `num_hard_negatives`)
    """
    def __init__(self, data=[], num_hard_negatives: int = 1):
        super().__init__()
        self.num_hard_negatives = num_hard_negatives
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def select(self, indices):
        # Create a new instance without calling __init__
        new_dataset = self.__class__.__new__(self.__class__)
        # Shallow copy the current __dict__
        new_dataset.__dict__ = self.__dict__.copy()
        # Replace data with only the selected items
        new_dataset.data = [self.data[i] for i in indices]
        return new_dataset


class IndexedDataset(BaseDataset):
    def __init__(self, data=[]):
        self.data = data
        # Store the index of each sample
        self.indices = list(range(len(data)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            "text": self.data[idx],
            "indices": self.indices[idx]  # or just return (self.data[idx], idx)
        }


class MSMarcoDataLoader:
    def __init__(self, args, tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.train_dataset, self.eval_dataset = self._get_transformed_datasets()
        self.train_dataset.set_transform(self._transform_func)
        self.tokenizer = tokenizer

        # use its state to decide which positives/negatives to sample
        self.trainer: Optional[Trainer] = None

    def _transform_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        current_epoch = int(self.trainer.state.epoch or 0)
        logger.debug(len(examples['query_positive']), examples['query_positive'][0])
        instruction_query = []
        passage = []
        
        # Determine document key based on train file
        if self.args.train_file == "aarontrinh02/instruction_following_synthetic":
            doc_key = "document_positive"
        elif self.args.train_file == "aarontrinh02/ms_marco_synthetic_data" or self.args.train_file == "aarontrinh02/ms_marco_synthetic_data_unfiltered":
            doc_key = "document"
        else:
            raise ValueError(f"Invalid train file: {self.args.train_file}")
        neg_doc_key1 = "hard_negative_document_1"
        neg_doc_key2 = "hard_negative_document_2"

        
        for i in range(len(examples['query_positive'])):
            # Format based on reverse flag
            if self.args.data_reverse:
                # Reverse format: Instruction first, then query
                instruction_query.append(f"Instruct: {examples['instruction_positive'][i]}\nQuery: {examples['query_positive'][i]} ".strip())
                passage.append(examples[doc_key][i])
                instruction_query.append(f"Instruct: {examples['instruction_negative'][i]}\nQuery: {examples['query_positive'][i]} ".strip())
                passage.append(examples[neg_doc_key1][i])
                instruction_query.append(f"Instruct: {examples['instruction_positive'][i]}\nQuery: {examples['query_negative'][i]} ".strip())
                passage.append(examples[neg_doc_key2][i])
            else:
                # Standard format: Query first, then instruction
                instruction_query.append(f"{examples['query_positive'][i]} {examples['instruction_positive'][i]}".strip())
                passage.append(examples[doc_key][i])
                instruction_query.append(f"{examples['query_positive'][i]} {examples['instruction_negative'][i]}".strip())
                passage.append(examples[neg_doc_key1][i])
                instruction_query.append(f"{examples['query_negative'][i]} {examples['instruction_positive'][i]}".strip())
                passage.append(examples[neg_doc_key2][i])
                
        instruction_query_batch_dict = self.tokenizer(instruction_query, 
                                                      max_length=self.args.q_max_len,
                                                      padding=PaddingStrategy.DO_NOT_PAD,
                                                      truncation=True)
        passage_batch_dict = self.tokenizer(passage,
                                            max_length=self.args.p_max_len,
                                            padding=PaddingStrategy.DO_NOT_PAD,
                                            truncation=True)
        merged_dict = merge_batch_dict(instruction_query_batch_dict=instruction_query_batch_dict, passage_batch_dict=passage_batch_dict)
        return merged_dict

    def _get_transformed_datasets(self) -> Tuple:
        data_files = {}
        combined_dataset = None
        
        # Load and combine all category datasets
        if self.args.train_file == "aarontrinh02/instruction_following_synthetic":
            for category in ['metamath']:
                category_dataset = load_dataset(self.args.train_file)[category]
                if combined_dataset is None:
                    combined_dataset = category_dataset
                else:
                    # Manually concatenate datasets
                    combined_dataset = concatenate_datasets([combined_dataset, category_dataset])
        elif self.args.train_file == "aarontrinh02/ms_marco_synthetic_data" or self.args.train_file == "aarontrinh02/ms_marco_synthetic_data_unfiltered":
            combined_dataset = load_dataset(self.args.train_file)['train']
        else:
            raise ValueError(f"Invalid train file: {self.args.train_file}")
        
        raw_dataset = combined_dataset
        
        if self.args.do_train:
            train_dataset = raw_dataset
            eval_dataset = None
        else:
            train_dataset = None
            eval_dataset = raw_dataset

        if self.args.do_train:
            if self.args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.args.max_train_samples))
            train_dataset.set_transform(self._transform_func)

        if self.args.do_eval:
            eval_dataset.set_transform(self._transform_func)

        return train_dataset, eval_dataset