import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments
from typing import List

from logger_config import logger
import copy

@dataclass
class Arguments(TrainingArguments):
    model_name_or_path: str = field(
        default='Qwen/Qwen2.5-1.5B',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    # data_dir_list: List[str] = field(
    #     default=['aarontrinh02/robust04-synthetic-filtered'], metadata={"help": "Path to train directory"}
    # )
    train_file: str = field(
        default='aarontrinh02/robust04-synthetic-filtered',
        metadata={"help": "Path or huggingface name of the training data"}
    )
    use_megatron_format_train_data: bool = field(
        default=False,
        metadata={"help": "use megatron format json for training data"}
    ) 
    do_eval: bool = field(
        default=False,
        metadata={"help": "if evaluate during training"}
    ) 
    local_rank: int = field(
        default=0
    )
    task_type: str = field(
        default='ir', metadata={"help": "task type: ir / qa"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics on (a jsonlines file)."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The directory for saving model base ckpts."
        },
    )

    train_n_passages: int = field(
        default=4,
        metadata={"help": "number of passages for each example (including both positive and negative passages)"}
    )
    share_encoder: bool = field(
        default=True,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )
    use_first_positive: bool = field(
        default=False,
        metadata={"help": "Always use the first positive passage"}
    )
    use_scaled_loss: bool = field(
        default=True,
        metadata={"help": "Use scaled loss or not"}
    )
    loss_scale: float = field(
        default=-1.,
        metadata={"help": "loss scale, -1 will use world_size"}
    )
    add_pooler: bool = field(
        default=False,
        metadata={"help": "Add a linear pooling layer"})
    out_dimension: int = field(
        default=768,
        metadata={"help": "output dimension for pooler"}
    )
    t: float = field(default=0.05, metadata={"help": "temperature of biencoder training"})
    l2_normalize: bool = field(default=True, metadata={"help": "L2 normalize embeddings or not"})
    t_warmup: bool = field(default=False, metadata={"help": "warmup temperature"})
    full_contrastive_loss: bool = field(default=True, metadata={"help": "use full contrastive loss or not"})
    pooling: str = field(
        default='mean', metadata={"help": "'mean', 'max', 'cls', 'weightedmean', 'lasttoken'"}
    )

    # following arguments are used for encoding documents
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})
    encode_in_path: str = field(default=None, metadata={"help": "Path to data to encode"})
    encode_save_dir: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_shard_size: int = field(default=int(2 * 10**6))
    encode_batch_size: int = field(default=32)

    # used for index search
    do_search: bool = field(default=False, metadata={"help": "run the index search loop"})
    search_split: str = field(default='dev', metadata={"help": "which split to search"})
    search_batch_size: int = field(default=128, metadata={"help": "query batch size for index search"})
    search_topk: int = field(default=200, metadata={"help": "return topk search results"})
    search_out_dir: str = field(default='', metadata={"help": "output directory for writing search results"})

    # used for prefix
    query_prefix: str = field(default='', metadata={"help": "prefix for queries"})
    passage_prefix: str = field(default='', metadata={"help": "prefix for passages"})

    # used for reranking
    do_rerank: bool = field(default=False, metadata={"help": "run the reranking loop"})
    rerank_max_length: int = field(default=256, metadata={"help": "max length for rerank inputs"})
    rerank_in_path: str = field(default='', metadata={"help": "Path to predictions for rerank"})
    rerank_out_path: str = field(default='', metadata={"help": "Path to write rerank results"})
    rerank_split: str = field(default='dev', metadata={"help": "which split to rerank"})
    rerank_batch_size: int = field(default=128, metadata={"help": "rerank batch size"})
    rerank_depth: int = field(default=1000, metadata={"help": "rerank depth, useful for debugging purpose"})
    rerank_forward_factor: int = field(
        default=1,
        metadata={"help": "forward n passages, then select top n/factor passages for backward"}
    )
    rerank_use_rdrop: bool = field(default=False, metadata={"help": "use R-Drop regularization for re-ranker"})

    # used for angle loss
    do_angle_loss: bool = field(default=False, metadata={"help": "use AngLE-Optimized Text Embedding loss"})
    angle_loss_weight: float = field(default=1.0, metadata={"help": "weight for angle loss"})

    # used for lora
    do_lora: bool = field(default=False, metadata={"help": "use lora"})
    lora_r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    lora_alpha: float = field(default=8.0, metadata={"help": "The alpha parameter for Lora scaling"})
    pretrained_lora_path: str = field(default=None, metadata={"help": "Pretrained lora parameters"})
    use_accelerator: bool = field(default=False, metadata={"help": "use accelerator"})

    # used for extract part layers from a model
    extract_first_n_layers: int = field(default=0, metadata={"help": "num of hidden transformers blocks to be extracted, \
        i.e. build a model with the first n layers of it. set 0 to take the entire model"})

    # used for knowledge distillation
    do_kd_gen_score: bool = field(default=False, metadata={"help": "run the score generation for distillation"})
    kd_gen_score_split: str = field(default='dev', metadata={
        "help": "Which split to use for generation of teacher score"
    })
    kd_gen_score_batch_size: int = field(default=128, metadata={"help": "batch size for teacher score generation"})
    kd_gen_score_n_neg: int = field(default=30, metadata={"help": "number of negatives to compute teacher scores"})

    do_kd_biencoder: bool = field(default=False, metadata={"help": "knowledge distillation to biencoder"})
    kd_mask_hn: bool = field(default=True, metadata={"help": "mask out hard negatives for distillation"})
    kd_cont_loss_weight: float = field(default=1.0, metadata={"help": "weight for contrastive loss"})

    rlm_generator_model_name: Optional[str] = field(
        default='google/electra-base-generator',
        metadata={"help": "generator for replace LM pre-training"}
    )
    rlm_freeze_generator: Optional[bool] = field(
        default=True,
        metadata={'help': 'freeze generator params or not'}
    )
    rlm_generator_mlm_weight: Optional[float] = field(
        default=0.2,
        metadata={'help': 'weight for generator MLM loss'}
    )
    all_use_mask_token: Optional[bool] = field(
        default=False,
        metadata={'help': 'Do not use 80:10:10 mask, use [MASK] for all places'}
    )
    rlm_num_eval_samples: Optional[int] = field(
        default=4096,
        metadata={"help": "number of evaluation samples pre-training"}
    )
    rlm_max_length: Optional[int] = field(
        default=144,
        metadata={"help": "max length for MatchLM pre-training"}
    )
    rlm_decoder_layers: Optional[int] = field(
        default=2,
        metadata={"help": "number of transformer layers for MatchLM decoder part"}
    )
    rlm_encoder_mask_prob: Optional[float] = field(
        default=0.3,
        metadata={'help': 'mask rate for encoder'}
    )
    rlm_decoder_mask_prob: Optional[float] = field(
        default=0.5,
        metadata={'help': 'mask rate for decoder'}
    )

    q_max_len: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query."
        },
    )
    p_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    dry_run: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set dry_run to True for debugging purpose'}
    )

    def __post_init__(self):
        # TODO: replace split by automatic arg mapping
        logger.debug(f"self.data_dir_list: {self.train_file}")
        # if len(self.data_dir_list) == 1: 
        #     self.data_dir_list = self.data_dir_list[0].split()
        # print(self.data_dir_list)
        # for data_dir in self.data_dir_list: 
        #     assert os.path.exists(data_dir)
        assert torch.cuda.is_available(), 'Only support running on GPUs'
        assert self.task_type in ['ir', 'qa']
        self.train_file = 'aarontrinh02/robust04-synthetic-filtered'

        if self.dry_run:
            self.logging_steps = 1
            self.max_train_samples = self.max_train_samples or 128
            self.num_train_epochs = 1
            self.per_device_train_batch_size = min(2, self.per_device_train_batch_size)
            self.train_n_passages = min(4, self.train_n_passages)
            self.rerank_forward_factor = 1
            self.gradient_accumulation_steps = 1
            self.rlm_num_eval_samples = min(256, self.rlm_num_eval_samples)
            self.max_steps = 30
            self.save_steps = self.eval_steps = 30
            logger.warning('Dry run: set logging_steps=1')

        if self.do_encode:
            assert self.encode_save_dir
            os.makedirs(self.encode_save_dir, exist_ok=True)
            assert os.path.exists(self.encode_in_path)

        if self.do_search:
            assert os.path.exists(self.encode_save_dir)
            assert self.search_out_dir
            os.makedirs(self.search_out_dir, exist_ok=True)

        if self.do_rerank:
            assert os.path.exists(self.rerank_in_path)
            logger.info('Rerank result will be written to {}'.format(self.rerank_out_path))
            assert self.train_n_passages > 1, 'Having positive passages only does not make sense for training re-ranker'
            assert self.train_n_passages % self.rerank_forward_factor == 0

        if self.do_kd_gen_score:
            for data_dir in self.data_dir_list:
                assert os.path.exists('{}/{}.jsonl'.format(self.data_dir, self.kd_gen_score_split))

        if self.do_kd_biencoder:
            if self.use_scaled_loss:
                assert not self.kd_mask_hn, 'Use scaled loss only works with not masking out hard negatives'

        if torch.cuda.device_count() <= 1:
            self.logging_steps = min(10, self.logging_steps)

        super(Arguments, self).__post_init__()

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.label_names = ['labels']


@dataclass
class AcceleratorConfig:
    """
    A subset of arguments relating to the underlying [`accelerate.Accelerator`]
    implementation utilized in the `Trainer` that can be customized.
    Mostly relating to data.

    Parameters:
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If
            `True` the actual batch size used will be the same on any kind of distributed processes, but it must be a
            round multiple of the `num_processes` you are using. If `False`, actual batch size used will be the one set
            in your script multiplied by the number of processes.
        dispatch_batches (`bool`, *optional*):
            If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process
            and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose
            underlying dataset is an `IterableDataset`, `False` otherwise.
        even_batches (`bool`, *optional*, defaults to `True`):
            If set to `True`, in cases where the total batch size across all processes does not exactly divide the
            dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among
            all workers.
        use_seedable_sampler (`bool`, *optional*, defaults to `True`):
            Whether or not use a fully seedable random sampler ([`accelerate.data_loader.SeedableRandomSampler`]). Ensures
            training results are fully reproducible using a different sampling technique. While seed-to-seed results
            may differ, on average the differences are negligible when using multiple different seeds to compare. Should
            also be ran with [`~utils.set_seed`] for the best results.
        gradient_accumulation_kwargs (`dict`, *optional*):
            Additional kwargs to configure gradient accumulation, see [`accelerate.utils.GradientAccumulationPlugin`].
            Any of the following (optional) keys are acceptable:
              num_steps (`int`): Will take precedence over [`~.TrainingArguments.gradient_accumulation_steps`] if
                the latter is set to 1, otherwise an exception will be raised.
              adjust_scheduler (`bool`): Whether to adjust the scheduler steps to account for [`~.TrainingArguments.gradient_accumulation_steps`].
                The [`accelerate.utils.GradientAccumulationPlugin`] default is `True`.
              sync_each_batch (`bool`): Whether to synchronize the gradients at each data batch.
                The [`accelerate.utils.GradientAccumulationPlugin`] default is `False`.
        non_blocking (`bool`, *optional*, defaults to `False`):
            Whether to use non-blocking CUDA calls to help minimize synchronization during
            distributed training with prepared `DataLoader` inputs being moved to device.
            Best if used with `pin_memory=True` in the `TrainingArguments`.
        use_configured_state (`bool*, *optional*, defaults to `False`):
            Whether or not to use a pre-configured `AcceleratorState` or `PartialState` defined
            before calling `TrainingArguments`. If `True`, an `Accelerator` or `PartialState`
            must be initialized. May lead to issues using sweeps or hyperparameter tuning.

    """

    # Data related arguments
    split_batches: bool = field(
        default=False,
        metadata={
            "help": "Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If"
            " `True` the actual batch size used will be the same on any kind of distributed processes, but it must be a"
            " round multiple of the `num_processes` you are using. If `False`, actual batch size used will be the one set"
            " in your script multiplied by the number of processes."
        },
    )
    dispatch_batches: Optional[bool] = field(
        default=None,
        metadata={
            "help": "If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process"
            " and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose"
            " underlying dataset is an `IterableDataslet`, `False` otherwise."
        },
    )
    even_batches: bool = field(
        default=True,
        metadata={
            "help": "If set to `True`, in cases where the total batch size across all processes does not exactly divide the"
            " dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among"
            " all workers."
        },
    )
    use_seedable_sampler: bool = field(
        default=True,
        metadata={
            "help": "Whether or not use a fully seedable random sampler ([`accelerate.data_loader.SeedableRandomSampler`])."
            "Ensures training results are fully reproducible using a different sampling technique. "
            "While seed-to-seed results may differ, on average the differences are negligible when using"
            "multiple different seeds to compare. Should also be ran with [`~utils.set_seed`] for the best results."
        },
    )

    non_blocking: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use non-blocking CUDA calls to help minimize synchronization during "
            "distributed training with prepared `DataLoader` inputs being moved to device. "
            "Best if used with `pin_memory=True` in the `TrainingArguments`. Requires accelerate "
            "v0.30.0."
        },
    )

    gradient_accumulation_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Additional kwargs to configure gradient accumulation, see [`accelerate.utils.GradientAccumulationPlugin`]. "
            "Any of the following (optional) keys are acceptable: "
            "  num_steps (`int`): Will take precedence over [`~.TrainingArguments.gradient_accumulation_steps`] if "
            "    the latter is set to 1, otherwise an exception will be raised. "
            "  adjust_scheduler (`bool`): Whether to adjust the scheduler steps to account for [`~.TrainingArguments.gradient_accumulation_steps`]. "
            "    The [`accelerate.utils.GradientAccumulationPlugin`] default is `True`. "
            "  sync_each_batch (`bool`): Whether to synchronize the gradients at each data batch. "
            "    The [`accelerate.utils.GradientAccumulationPlugin`] default is `False`."
        },
    )
    use_configured_state: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use a pre-configured `AcceleratorState` or `PartialState` defined before calling `TrainingArguments`."
            "If `True`, an `Accelerator` or `PartialState` must be initialized. May lead to issues using sweeps or hyperparameter tuning."
        },
    )

    @classmethod
    def from_json_file(cls, json_file):
        # Check if exists
        open_file = io.open if os.path.exists(json_file) else open
        with open_file(json_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        # Check for keys and load sensible defaults
        extra_keys = sorted(key for key in config_dict.keys() if key not in cls.__dataclass_fields__.keys())
        if len(extra_keys) > 0:
            raise ValueError(
                f"The config file at {json_file} had unknown keys ({extra_keys}), please try upgrading your `transformers`"
                " version or fix (and potentially remove these keys) from your config file."
            )
        return cls(**config_dict)

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def pop(self, key, default=None):
        return self.__dict__.pop(key, default)