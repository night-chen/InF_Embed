import os.path
import random

from typing import Tuple, Dict, List, Optional
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedTokenizerFast, Trainer

from logger_config import logger

def attention_merge_batch_dict(instruction_batch_dict, query_batch_dict, passage_batch_dict):
    logger.debug(instruction_batch_dict.keys(), len(instruction_batch_dict['input_ids']))
    merged_dict = {'i_{}'.format(k): v for k, v in instruction_batch_dict.items()}
    for k, v in query_batch_dict.items():
        merged_dict['q_{}'.format(k)] = v
    for k, v in passage_batch_dict.items():
        merged_dict['p_{}'.format(k)] = v
    return merged_dict


class AttentionDataLoader:
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
        instruction = []
        query = []
        passage = []
        if self.args.train_file == "aarontrinh02/instruction_following_synthetic":
            doc_key = "document_positive"
        elif self.args.train_file == "aarontrinh02/ms_marco_synthetic_data":
            doc_key = "document"
        else:
            raise ValueError(f"Invalid train file: {self.args.train_file}")
        for i in range(len(examples['query_positive'])):
            # concatenate instruction and query
            query.append(f"{examples['query_positive'][i]}".strip())
            instruction.append(f"{examples['instruction_positive'][i]}".strip())
            passage.append(examples[doc_key][i])
            query.append(f"{examples['query_positive'][i]}".strip())
            instruction.append(f"{examples['instruction_negative'][i]}".strip())
            passage.append(examples['hard_negative_document_1'][i])
            query.append(f"{examples['query_negative'][i]}".strip())
            instruction.append(f"{examples['instruction_positive'][i]}".strip())
            passage.append(examples['hard_negative_document_2'][i])
        instruction_batch_dict = self.tokenizer(instruction,
                                                max_length=self.args.q_max_len,
                                                padding=PaddingStrategy.DO_NOT_PAD,
                                                truncation=True)
        query_batch_dict = self.tokenizer(query, 
                                            max_length=self.args.q_max_len,
                                            padding=PaddingStrategy.DO_NOT_PAD,
                                            truncation=True)
        passage_batch_dict = self.tokenizer(passage,
                                            max_length=self.args.p_max_len,
                                            padding=PaddingStrategy.DO_NOT_PAD,
                                            truncation=True)
        merged_dict = attention_merge_batch_dict(instruction_batch_dict=instruction_batch_dict, query_batch_dict=query_batch_dict, passage_batch_dict=passage_batch_dict)
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
                    combined_dataset = concatenate_datasets([combined_dataset, category_dataset])
        elif self.args.train_file == "aarontrinh02/ms_marco_synthetic_data":
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
