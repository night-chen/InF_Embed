import os
import random
import logging

from typing import Tuple, Dict, List, Optional
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedTokenizerFast, Trainer

from config import Arguments
from logger_config import logger
from .loader_utils import group_doc_ids, _slice_with_mod, merge_batch_dict
    
class RetrievalDataLoader:

    def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.negative_size = args.train_n_passages - 1
        assert self.negative_size >= 0
        self.tokenizer = tokenizer
        # Megatron format data from json file
        if args.use_megatron_format_train_data:
            if len(args.data_dir_list) == 1:
                logger.debug("use megatron format train data, no eval_data is created")
                
                train_dataset = load_dataset('json', data_files=args.data_dir_list[0], split='train', cache_dir=args.cache_dir)
                
                self.train_dataset = self._get_transformed_megatron_dataset(train_dataset)
            elif len(args.data_dir_list) > 1:
                logger.debug("use multiple megatron format train data, no eval_data is created")
                train_datasets = []
                for data_dir in args.data_dir_list:
                    # chunk_size to avoid pyarrow read large json file error: https://github.com/huggingface/datasets/issues/6108
                    train_datasets.append(load_dataset('json', data_files=data_dir, cache_dir=args.cache_dir, split='train', chunksize=10<<20))
                combined_train_dataset = concatenate_datasets(train_datasets)
                self.train_dataset = self._get_transformed_megatron_dataset(combined_train_dataset)


        # MS MARCO dataset from intfloat/simlm-msmarco
        else:
            logger.info("Use MS MARCO data")
            assert len(args.data_dir_list) == 1
            corpus_path = os.path.join(args.data_dir_list[0], 'passages.jsonl.gz')
            self.corpus: Dataset = load_dataset('json', cache_dir=args.cache_dir, data_files=corpus_path)['train']
            self.train_dataset, self.eval_dataset = self._get_transformed_datasets()

        # use its state to decide which positives/negatives to sample
        self.trainer: Optional[Trainer] = None
    
    def _transform_query(self, args, question, prompt=""):
        question = question.strip()
        if args.query_prefix in ['none', '', " "]:
            return question
        elif args.query_prefix in ['plain', 'query:']:
            return "query:" + " " + question
        elif args.query_prefix in ['instruct']:
            return "Instruct: Given a question, retrieve documents that can help answer the question\nQuery:" + " " + question
        elif args.query_prefix in ['pt']:
            return "Represent this query for searching relevant passages\nquery:" + " " + question
        elif args.query_prefix in ['task']:
            # print(prompt)
            if prompt == "":
                # print("NO PROMPT")
                return "Query:" + " " + question
            else:
                return f"{prompt}\nQuery:" + " " + question
        elif args.query_prefix in ['chat']:
            chat = [{'role': 'user', 'content': "Given a question, retrieve documents that can help answer the question\nQuery:" + " " + question}]
            tokenized_chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            return tokenized_chat
        else:
            raise NotImplementedError
        

    def _transform_passage(self, args, passage, prompt=""):
        passage = passage.strip()
        if args.passage_prefix in ['none', '', " "]:
            return passage
        elif args.passage_prefix in ['plain', 'passage:']:
            return "passage:" + " " + passage
        elif args.passage_prefix in ['pt']:
            return "Represent this passage\npassage:" + " " + passage
        elif args.passage_prefix in ['instruct']:
            return "Instruct: Represent Passage for text retrieval task\nPassage:" + " " + passage
        elif args.passage_prefix in ['task']:
            if prompt == "":
                raise NotImplementedError
            return f"Instruct: {prompt}\nPassage:" + " " + passage
        elif args.passage_prefix in ['chat']:
            chat = [{'role': 'user', 'content': "Instruct: Represent Passage for text retrieval task\nPassage:" + " " + passage}]
            tokenized_chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            return tokenized_chat
        else:
            raise NotImplementedError


    def _transform_func_megatron_format(self, examples: Dict[str, List]) -> Dict[str, List]:
        current_epoch = int(self.trainer.state.epoch or 0)
        if "question" in examples:
            questions = examples['question'] 
        elif "query" in examples:
            questions = examples['query'] 
        else:
            raise Exception(f"Question and Query not in Example keys")
        # print("Question:\t", examples['question'][:2], )
        # print("-------------------")
        # print("posdoc", examples['pos_doc'][:2])
        # print("-------------------")
        # print("negdoc", examples['neg_doc'][:2])
        # assert 0
        if self.args.query_prefix != "":
            if "prompt" in examples: # add task descriptions dynamically
                prompts =  examples["prompt"]
                # print(prompts, questions)
                # print(examples["prompt"],examples['question'],examples['pos_doc'],examples['neg_doc'] )
                questions = [self._transform_query(self.args, question, prompt) for prompt, question in zip(prompts, questions)]
                # print(questions)
                # assert 0
                # print("Q:", questions[0], '\n---------')
            else:
                questions = [self._transform_query(self.args, question) for question in questions]
                # print("Q:", questions[0], '\n---------')
                # questions = [self.args.query_prefix+" "+question + self.tokenizer.eos_token for question in questions]
        if self.negative_size > len(examples['neg_doc'][0]):
            raise Exception(f"--negative_size {self.negative_size} is bigger than 'neg_docs: {len(examples['neg_doc'])}" )
        
        batch_positives: List[str] = examples['pos_doc']
        batch_negatives: List[str] = examples['neg_doc']
        cur_pos_neg_doc_batch = []
        assert len(batch_positives) == len(batch_negatives)
        for i_example in range(len(batch_positives)):

            # Get one pos doc
            positives = batch_positives[i_example]
            if isinstance(positives, List):
                pos_ids = [i for i in range(len(positives))]
                # Get one pos doc
                cur_pos_id = _slice_with_mod(
                    elements=pos_ids, offset=current_epoch + self.args.seed, cnt=1)
                cur_pos_neg_doc_batch.append(positives[cur_pos_id[0]])
            # One positive str
            else:
                cur_pos_neg_doc_batch.append(positives) 

            # Append negative_size neg docs after one pos doc
            if self.negative_size > 0:
                negatives = batch_negatives[i_example]
                neg_ids = [i for i in range(len(negatives))]
                # Get self.negative_size pos
                cur_neg_ids = _slice_with_mod(
                        elements=neg_ids, offset=current_epoch + self.args.seed, cnt=self.negative_size)
                cur_pos_neg_doc_batch+=[negatives[n_id] for n_id in cur_neg_ids]
        # if self.args.passage_prefix != "":
        # cur_pos_neg_doc_batch = [self.args.passage_prefix+" "+passage+self.tokenizer.eos_token for passage in cur_pos_neg_doc_batch] 
        cur_pos_neg_doc_batch = [self._transform_passage(self.args, passage) for passage in cur_pos_neg_doc_batch] 
        # print("P:", cur_pos_neg_doc_batch[0], '\n***********')
        
        # query_batch_dict = self.tokenizer(questions,
        #                                   max_length=self.args.q_max_len - 1,
        #                                   padding=PaddingStrategy.DO_NOT_PAD,
        #                                   truncation=True)
        
        query_batch_dict = self.tokenizer(questions, max_length=self.args.q_max_len - 1, return_attention_mask=False, padding=PaddingStrategy.DO_NOT_PAD, truncation=True)
        # # append eos_token_id to every input_ids
        query_batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in query_batch_dict['input_ids']]
        query_batch_dict = self.tokenizer.pad(query_batch_dict, padding=False, return_attention_mask=True)

        # print(query_batch_dict )
        # query_batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in query_batch_dict['input_ids']]
        # doc_batch_dict = self.tokenizer(cur_pos_neg_doc_batch,
        #                                 max_length=self.args.p_max_len - 1,
        #                                 padding=PaddingStrategy.DO_NOT_PAD,
        #                                 truncation=True)
        doc_batch_dict = self.tokenizer(cur_pos_neg_doc_batch, max_length=self.args.p_max_len - 1, return_attention_mask=False, padding=PaddingStrategy.DO_NOT_PAD, truncation=True)
        # # append eos_token_id to every input_ids
        doc_batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in doc_batch_dict['input_ids']]
        doc_batch_dict = self.tokenizer.pad(doc_batch_dict, padding=False, return_attention_mask=True)
        # print("------------")
        # print(questions[:2])        
        # print(query_batch_dict['input_ids'][:2], query_batch_dict['attention_mask'][:2], '\n*************')
        # print("------------")
        # print(cur_pos_neg_doc_batch[:2])
        # print(doc_batch_dict['input_ids'][:2], doc_batch_dict['attention_mask'][:2], '\n------------')
        # exit()
        # doc_batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in doc_batch_dict['input_ids']]
        # print("original", query_batch_dict['input_ids'][:2])
        # print("original", doc_batch_dict['input_ids'][:2])
        # doc_batch_dict = self.tokenizer.tokenize(cur_pos_neg_doc_batch[0])
        merged_dict = merge_batch_dict(query_batch_dict=query_batch_dict, doc_batch_dict=doc_batch_dict, train_n_passages=self.args.train_n_passages)
        return merged_dict


    def _transform_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        current_epoch = int(self.trainer.state.epoch or 0)
        logger.debug(len(examples['query']), examples['query'][0], examples['query'][1])
        # 32 what gas is produced by hcl and zinc
        input_doc_ids: List[int] = group_doc_ids(
            examples=examples,
            negative_size=self.negative_size,
            offset=current_epoch + self.args.seed,
            use_first_positive=self.args.use_first_positive
        )
        assert len(input_doc_ids) == len(examples['query']) * self.args.train_n_passages
        logging.debug("1", len(input_doc_ids), input_doc_ids[0])
        # 256 6862506
        # 32 queries * (1 pos + 7 neg) = 256

        input_docs: List[str] = [self.corpus[doc_id]['contents'] for doc_id in input_doc_ids]
        input_titles: List[str] = [self.corpus[doc_id]['title'] for doc_id in input_doc_ids]
        logging.debug("2", len(input_docs), input_docs[0])
        # 256 Prostatic Stent. The Spanner® Prostatic Stent is a temporary stent
        query_batch_dict = self.tokenizer(examples['query'],
                                          max_length=self.args.q_max_len,
                                          padding=PaddingStrategy.DO_NOT_PAD,
                                          truncation=True)
        doc_batch_dict = self.tokenizer(input_titles,
                                        text_pair=input_docs,
                                        max_length=self.args.p_max_len,
                                        padding=PaddingStrategy.DO_NOT_PAD,
                                        truncation=True)

        merged_dict = merge_batch_dict(query_batch_dict=query_batch_dict, doc_batch_dict=doc_batch_dict, train_n_passages=self.args.train_n_passages)

        step_size = self.args.train_n_passages
        if self.args.do_kd_biencoder:
            qid_to_doc_id_to_score = {}

            def _update_qid_pid_score(q_id: str, ex: Dict):
                assert len(ex['doc_id']) == len(ex['score'])
                if q_id not in qid_to_doc_id_to_score:
                    qid_to_doc_id_to_score[q_id] = {}
                for doc_id, score in zip(ex['doc_id'], ex['score']):
                    qid_to_doc_id_to_score[q_id][int(doc_id)] = score

            for idx, query_id in enumerate(examples['query_id']):
                _update_qid_pid_score(query_id, examples['positives'][idx])
                _update_qid_pid_score(query_id, examples['negatives'][idx])

            merged_dict['kd_labels'] = []
            for idx in range(0, len(input_doc_ids), step_size):
                qid = examples['query_id'][idx // step_size]
                cur_kd_labels = [qid_to_doc_id_to_score[qid][doc_id] for doc_id in input_doc_ids[idx:idx + step_size]]
                merged_dict['kd_labels'].append(cur_kd_labels)
            assert len(merged_dict['kd_labels']) == len(examples['query_id']), \
                '{} != {}'.format(len(merged_dict['kd_labels']), len(examples['query_id']))

        # Custom formatting function must return a dict
        return merged_dict

    def _get_transformed_megatron_dataset(self, train_dataset) -> Dataset:
        if self.args.do_train:
            if self.args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.args.max_train_samples))
            # Log a few random samples from the training set:
            
            for index in random.sample(range(len(train_dataset)), 1):
                logger.debug(f"Sample {index} of the training set: {train_dataset[index]}.")
            train_dataset.set_transform(self._transform_func_megatron_format)
        if self.args.do_eval:
            raise ValueError("--do_eval requires a validation dataset, split not implemented")
        return train_dataset

    def _get_transformed_datasets(self) -> Tuple:
        data_files = {}
        if self.args.train_file is not None:
            data_files["train"] = self.args.train_file.split(',')
        if self.args.validation_file is not None:
            data_files["validation"] = self.args.validation_file
        raw_datasets: DatasetDict = load_dataset('json', data_files=data_files)

        train_dataset, eval_dataset = None, None

        if self.args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if self.args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.args.max_train_samples))
            # Log a few random samples from the training set:
            for index in random.sample(range(len(train_dataset)), 3):
                logger.debug(f"Sample {index} of the training set: {train_dataset[index]}.")
            #train_dataset
            #Dataset({
            #    features: ['query_id', 'query', 'positives', 'negatives'],
            #    num_rows: 485905
            train_dataset.set_transform(self._transform_func)

        if self.args.do_eval:
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            eval_dataset.set_transform(self._transform_func)

        return train_dataset, eval_dataset