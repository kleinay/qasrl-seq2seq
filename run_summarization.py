#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
from argparse import Namespace
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Tuple
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict

import wandb
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric
import pandas as pd

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import EvalPrediction, get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
from preprocessing import Preprocessor
from pipeline import get_markers_for_model
from qasrl_gs.scripts.common import QuestionAnswer
from utils import setup_wandb, replaceKeys, str2num, without_prefix, without_suffix
import run_evaluation

check_min_version("4.10.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_type: str = field(
        metadata={"help": "Type of the model (t5 or bart)"}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    debug_mode: bool = field(
        default=False
    )
    freeze_parameters: bool = field(
        default=False
    )
    wandb_run_name: str = field(
        default=None
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={"help": "Decoding with top_k tokens."},
    )
    top_p: Optional[float] = field(
        default=None,
        metadata={"help": "Decoding using the top_p algorithm (0 < p <= 1)."},
    )
    num_beam_groups: Optional[int] = field(
        default=None,
        metadata={"help": "Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams."},
    )
    constrain_generation: bool = field(
        default=False,
        metadata={"help": "Whether to apply constrained decoding during prediction."},
    )
    


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    do_eval_on: Optional[str] = field(
        default="validation", metadata={"help": "Wether to perform evaluation on 'validation' set or on 'test' set"}
    )
    limit_train_data: Optional[float] = field(
        default=1.0, metadata={"help": "Percentage of training samples to use during training."}
    )
    limit_eval_data: Optional[float] = field(
        default=1.0, metadata={"help": "Percentage of evaluation samples (both for `evaluate` and `predict`) to use during evaluation."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    # summary_column: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    # )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=200,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    preprocess_input_func: str = field(
        default='input_predicate_marker'
    )
    preprocess_output_func: str = field(
        default='all_by_answer_ordering'
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    append_verb_form: bool = field(
        default=True, metadata={"help": "Whether to append the verb_form feature (if exists) to the end of the input sequence."}
    ) 
    predicate_marker_type: str = field(
        # Literal["generic", "pred_type"]
        default="generic", 
        metadata={"help": "The method by which predicate marker is marking the predicate. Only in use when `preprocess_input_func`=input_predicate_marker"}
    )
    use_bilateral_predicate_marker: bool = field(
        default=False, metadata={"help": "Whether to demarcate the predicate from both sides or just before it. Only in use when `preprocess_input_func`=input_predicate_marker"}
    )
    learn_predicate_type: Optional[str] = field(
        # Literal["pre", "post"]
        default=None, metadata={"help": "Whether and how incorporate `predicate_type` info into the output sequence, as an MTL objective to enhance joint learning."}
    ) 
    
    # Arguments for inference (predict)
    predicate_type: Optional[str] = field(
        default=None, metadata={"help": "Either 'verbal' or 'nominal'; this will determine the `predicate_type` during inference, if the input file doesn't have 'predicate_type' column."}
    ) 

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            # raise ValueError("Need either a dataset name or a training/validation file.")
            pass
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def _clean_mem():
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def _freeze_parameters(model):
    # TODO: Do not freeze special tokens

    for param in model.encoder.parameters():
        param.requires_grad = False

    for param in model.decoder.parameters():
        param.requires_grad = False

    for param in model.lm_head.parameters():
        param.requires_grad = True


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        # fix `do_eval` mistake
        training_args.do_eval = "--do_eval" in sys.argv


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    run = wandb.run or setup_wandb("wandb" in training_args.report_to, model_args.wandb_run_name)
    wandb.config.update(data_args.__dict__, allow_val_change=True)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # logger.info(f"Training/evaluation parameters {training_args}")
    # logger.info(f"Model parameters {model_args}")

    if model_args.model_type not in ["t5", "bart"]:
        raise ValueError(f"Invalid model_type received ; model_type {model_args.model_type}")

    is_t5_model = model_args.model_type == "t5" or model_args.model_type in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]
    if data_args.source_prefix is None and is_t5_model:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    def get_predicate_type_label_from_dataset_name(dataset_name) -> str:
        if "qa_srl" in dataset_name:
            predicate_type = "verbal" 
        elif "qanom" in dataset_name: 
            predicate_type = "nominal"
        elif "discourse" in dataset_name:
            predicate_type = "discourse"
        else:
            raise ValueError(f"dataset {dataset_name.strip()} is not supported shen setting predicate_type feature")
        return predicate_type
    
    def load_dataset_with_predicate_type(dataset_name, split) -> Dataset:
        dataset = load_dataset(dataset_name.strip(), data_args.dataset_config_name, cache_dir=model_args.cache_dir)[split]
        predicate_type = get_predicate_type_label_from_dataset_name(dataset_name)
        dataset = dataset.add_column("predicate_type", [predicate_type]*len(dataset))
        return dataset

    def load_dataset_dict_with_predicate_type(dataset_name) -> DatasetDict:
        predicate_type = get_predicate_type_label_from_dataset_name(dataset_name)
        dataset_dict = load_dataset(dataset_name.strip(), data_args.dataset_config_name, cache_dir=model_args.cache_dir)
        dataset_dict = {split: dataset.add_column("predicate_type", [predicate_type]*len(dataset))
                        for split, dataset in dataset_dict.items()}
        return dataset_dict    
    
    if data_args.dataset_name is not None:
        # Optionally, dataset_name is a list, given in the form [dataset_1, dataset_2, ...]
        if data_args.dataset_name.startswith("["):  
            source_datasets_names = data_args.dataset_name[1:-1].split(",")
            source_dataset_dicts = [load_dataset_dict_with_predicate_type(dataset_name)
                               for dataset_name in source_datasets_names]
            raw_datasets = datasets.DatasetDict({
                split: datasets.source_datasets([dataset[split] for dataset in source_dataset_dicts], info=source_dataset_dicts[0].info)
                for split in source_dataset_dicts[0]
            })
        # Optionally, dataset_name is a dict, specifying different datasets for different splits,
        # given in the form {train: dataset_1, dataset_2; validation: ...; test: ...}.
        #   We can also specify a factor in which we "duplicate" each dataset to have custom train-set proportions; e.g.
        # {train: 5 * dataset_1, dataset_2; validation: ...; test: ...}   will repeat dataset_1 5-times in the resulting training-set.  
        elif data_args.dataset_name.startswith("{"):
            split_datasets_str = data_args.dataset_name[1:-1].split("; ")
            datasets_str_per_split = dict([d.split(":") for d in split_datasets_str])
            datasets_specifications_per_split = {spl: datasets.split(",") 
                                        for spl, datasets in datasets_str_per_split.items()}
            def get_facotr_and_name_from_dataset_specification(specification: str) -> Tuple[int, str]:
                # extract duplication factors per dataset_specification
                if "*" not in specification:
                    return 1, specification
                else:
                    factor, name = specification.split("*")
                    return str2num(factor), name
            dataset_factor_and_name_per_split: Dict[str, List[Tuple[int, str]]]
            dataset_factor_and_name_per_split = {spl: [get_facotr_and_name_from_dataset_specification(dataset_specification)
                                                       for dataset_specification in dataset_specification_list]
                                                 for spl, dataset_specification_list in datasets_specifications_per_split.items()}
            # implement duplication factor by concatenating loaded Dataset objects from the same type, and interleaving the different types
            datasets_loaded_per_split = {}
            for spl, dataset_specs in dataset_factor_and_name_per_split.items():
                loaded_datasets = []
                for factor, name in dataset_specs:
                    dataset: Dataset = load_dataset_with_predicate_type(name, spl)
                    if isinstance(factor, int):
                        dupl_dataset = datasets.concatenate_datasets([dataset] * factor, info=dataset.info)
                    else: # isinstance(facotr, float):
                        dupl_dataset = dataset.select(range(int(len(dataset) * factor))) 
                    loaded_datasets.append(dupl_dataset)
                datasets_loaded_per_split[spl] = loaded_datasets
            # interleave the different-typed datasets 
            raw_datasets = datasets.DatasetDict({
                split: datasets.concatenate_datasets(source_datasets, info=source_datasets[0].info)
                for split, source_datasets in datasets_loaded_per_split.items()
            })
            
        # Downloading and loading a single dataset from the hub.
        else:
            raw_datasets = load_dataset_dict_with_predicate_type(data_args.dataset_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Special tokens used in modelling the sequences
    special_tokens_constants = get_markers_for_model(is_t5_model)
    all_special_tokens = None
    if not is_t5_model:
        all_special_tokens = list(vars(special_tokens_constants).values())
    unnormalized_tokens = ["``"]


    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        additional_special_tokens=all_special_tokens
    )
    tokenizer.add_tokens(unnormalized_tokens)
    # add to special_tokens_constants other special tokens from tokenizer
    special_tokens_constants.eos_token = tokenizer.eos_token
    special_tokens_constants.bos_token = tokenizer.bos_token
    special_tokens_constants.pad_token = tokenizer.pad_token

    optional_params_to_pass_to_model_config = ("top_k", "top_p", "num_beam_groups")
    # optionally: specify here decoding method params, e.g. "top_k", "top_p", 
    # comment: "num_beams" from DataTrainingArguments is already handled 
    kwargs_for_model_config = {key: value
                        for key, value in model_args.__dict__.items() 
                        if key in optional_params_to_pass_to_model_config 
                           and value is not None} 
    config.update(kwargs_for_model_config)       
    wandb.config.update({"model_name_or_path": model_args.model_name_or_path}, allow_val_change=True) 
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model_args.freeze_parameters:
        _freeze_parameters(model)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets[data_args.do_eval_on].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    if data_args.text_column is None:
        # text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        text_column = 'sentence'
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    preprocessor = Preprocessor(data_args, special_tokens_constants)

    # Preprocssing for train and eval - preparing both input sequence and expected output sequence ("labels")
    def preprocess_function(examples):
        inputs = examples[text_column]
        batch_size = len(inputs)
        # in 2015 dataset the column is predicate, and in 2020 it is verb
        predicate_key = 'predicate' if 'predicate' in examples else 'verb'
        # in 2015 dataset it is predicate_idx, and in 2020 it is verb_idx
        predicate_index_key = 'predicate_idx' if 'predicate_idx' in examples else 'verb_idx'
        # in 2015 dataset the question is an array, and in 2020 it is a string
        questions = [" ".join(x) if isinstance(x, list) else x for x in examples['question']]
        # in 2015 dataset the answers is an array, and in 2020 it is a string separated by ~!~
        answers = [x.split("~!~") if isinstance(x, str) else x for x in examples['answers']]
        if 'answer_ranges' in examples:
            # in qasrl-2018 and qanom, the answer_range is an array, 
            answer_ranges = [x.split("~!~") if isinstance(x, str) else x for x in examples['answer_ranges']]
        elif 'answer_range' in examples:
            # in qasrl-2020 it is a string separated by ~!~
            answer_ranges = [x.split("~!~") for x in examples['answer_range']]
        else:
            answer_ranges = [None] * batch_size
        if 'verb_form' in examples:
            verb_forms = examples["verb_form"]
        else:
            verb_forms = [None] * batch_size
        if 'is_verbal' in examples:
            is_verbals = examples["is_verbal"]
        else:
            is_verbals = [True] * batch_size    
        if 'predicate_type' in examples:    # only at joint training  
            pred_types = examples["predicate_type"]
        else:
            pred_types = [None] * batch_size    
            
        # in 2015 dataset there is no ids so just initialize empty, and in 2020 it is qasrl_id
        sent_id_keys = list({"sent_id", "qasrl_id"}.intersection(examples))
        qasrl_ids = examples[sent_id_keys[0]] if sent_id_keys else ["" for x in examples[predicate_index_key]]

        predicates = examples[predicate_key]
        predicate_indices = examples[predicate_index_key]
                
        df = pd.DataFrame([{"sentence": input, "predicate": predicate, 
                            "question": question, "answer": answer, "answer_range": answer_range, 
                            "predicate_idx": predicate_idx, "qasrl_id": qasrl_id, 
                            "verb_form": verb_form, "is_verbal": is_verbal, "predicate_type": pred_type} 
                           for input, predicate, question, answer, answer_range, predicate_idx, qasrl_id, verb_form, is_verbal, pred_type in 
                           zip(inputs, predicates, questions, answers, answer_ranges, predicate_indices, qasrl_ids, verb_forms, is_verbals, pred_types)
                           if is_verbal # TODO change to is_verbal     # Don't train on non-predicates
                           ])     

        grouped_df = df.groupby(['sentence', 'predicate_idx'])
        preprocessed_inputs = grouped_df.apply(preprocessor.preprocess_input).tolist()
        targets = grouped_df.apply(preprocessor.preprocess_output).tolist()
        
        # Tokenizing the input sequences
        model_inputs = tokenizer(preprocessed_inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        # We can take index 0 because of the group by all values have the same predicate
        predicate_level_df = grouped_df.apply(lambda x: x.iloc[0])
        model_inputs["predicate"] = predicate_level_df['predicate'].tolist()
        model_inputs["predicate_idx"] = predicate_level_df['predicate_idx'].tolist()
        model_inputs["sentence"] = predicate_level_df['sentence'].tolist()
        model_inputs["qasrl_id"] = predicate_level_df['qasrl_id'].tolist()
        return model_inputs
    
    def preprocess_for_inference(examples):
        batch_size = len(examples[text_column]) 
        tok_sents = [sent.split(" ") for sent in examples[text_column]]
        # Be permissive about column names (for ease of inference)
        predicate_index_key = ({'predicate_idx','verb_idx','target_idx'} & examples.keys()).pop()
        predicate_indices = examples[predicate_index_key]
        predicate_keys = ({'predicate','verb','noun'} & examples.keys())
        predicate_key = predicate_keys.pop() if predicate_keys else None                
        qasrl_id_keys = ({'qasrl_id','sent_id','sentence_id', 'SentenceId'} & examples.keys())
        qasrl_id_key = qasrl_id_keys.pop() if qasrl_id_keys else None
       
        examples_unified_labels = replaceKeys(examples, {text_column: "sentence", predicate_index_key: "predicate_idx", predicate_key: "predicate", qasrl_id_key: "qasrl_id"}, inplace=False) 
        df = pd.DataFrame(examples_unified_labels.data)
        
        for required_key in ("verb_form", "predicate_type"):
            if required_key not in df.columns:
                df[required_key] = None
        # Make sure returned dataset have "predicate" & "qasrl_id" columns, required for parsing output into qasrl format
        if 'predicate' not in df.columns:
            df['predicate'] = [toks[i] for toks, i in zip(tok_sents, predicate_indices)]
        if 'qasrl_id' not in df.columns:
            sent2id = {sent: f"sent_{i+1}" for i,sent in enumerate(set(df['sentence']))}
            df['qasrl_id'] = df['sentence'].apply[sent2id.get]
        
        # if dataset has an "is_verbal" column (e.g. when it is taken from qanom predicate_detector's output), only predict for positive predicates
        if "is_verbal" in df.columns:
            df = df[df["is_verbal"]]
                
        # preprocess input sequence (tokenized sentence, predicate marking, incorporating verb_form or predicate_type information, etc.)
        grouped_df = df.groupby(['sentence', 'predicate_idx'])  

        preprocessed_inputs = grouped_df.apply(preprocessor.preprocess_input).tolist()
        # Tokenizing the input sequences
        model_inputs = tokenizer(preprocessed_inputs, max_length=data_args.max_source_length, 
                                 padding=padding, truncation=True)
        
        predicate_level_df = grouped_df.apply(lambda x: x.iloc[0]) # instance-level --- #-rows is as #-predicates (=instances)

        model_inputs["predicate_idx"] = predicate_level_df['predicate_idx'].tolist()
        model_inputs["sentence"] = predicate_level_df['sentence'].tolist()
        model_inputs["predicate"] = predicate_level_df['predicate'].tolist()  
        model_inputs["qasrl_id"] = predicate_level_df['qasrl_id'].tolist()
        return model_inputs        

    # Prepare datasets (load, sample/limit, preprocess)
    _clean_mem()
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        if data_args.limit_train_data is not None:
            train_dataset = train_dataset.select(range(int(len(train_dataset)*data_args.limit_train_data)))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            orig_train_dataset = train_dataset
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Preprocessing train dataset",
            )
        # log a sample of training instances 
        data_example = train_dataset[0]
        logger.info(f"Data example: {data_example}") 

    if training_args.do_train or (training_args.do_eval and data_args.do_eval_on == "validation"):
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval and --do_train require a validation dataset")
        validation_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            validation_dataset = validation_dataset.select(range(data_args.max_eval_samples))
        if data_args.limit_eval_data is not None:
            validation_dataset = validation_dataset.select(range(int(len(validation_dataset)*data_args.limit_eval_data)))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            orig_validation_dataset = validation_dataset
            validation_dataset = validation_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Preprocessing validation dataset",
            )

    if training_args.do_predict or (training_args.do_eval and data_args.do_eval_on == "test"):
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict (or --do_eval_on='test') requires a test dataset")
        test_dataset = raw_datasets["test"]
        if data_args.max_eval_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_eval_samples))
        if data_args.limit_eval_data is not None:
            test_dataset = test_dataset.select(range(int(len(test_dataset)*data_args.limit_eval_data)))
        # If predict_dataset includes gold standard reference, use the regular preprocessing which also prepares `labels`;
        # Otherwise, do inference without labels 
        if "question" in test_dataset.column_names and "answers" in test_dataset.column_names:
            do_inference_without_labels = False
            preprocessing_func_for_test = preprocess_function
        else:
            do_inference_without_labels = True
            preprocessing_func_for_test = preprocess_for_inference
        with training_args.main_process_first(desc="test dataset map pre-processing"):
            orig_test_dataset = test_dataset
            test_dataset = test_dataset.map(
                preprocessing_func_for_test,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Preprocessing prediction dataset",
            )
    # Set `evaluation_dataset` as the dataset we perform stanalone evaluation on.
    #  it can be either the `validation_dataset` or `test_dataset`, depending on `data_args.do_eval_on`
    evaluation_dataset = validation_dataset if data_args.do_eval_on == "validation" else test_dataset
    orig_evaluation_dataset = orig_validation_dataset if data_args.do_eval_on == "validation" else orig_test_dataset


    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    
    from strings_to_objects_parser import StringsToObjectsParser
    strings_to_objects_parser = StringsToObjectsParser(special_tokens_constants, tokenizer)
    
    # Metric
    rouge_metric = load_metric("rouge")
    exact_match_metric = load_metric("metrics/exact_match.py")
    element_exact_match_metric = load_metric("metrics/element_exact_match.py")
    wh_answer_exact_match_metric = load_metric("metrics/wh_qasrl_match.py")

    def clean_output_seq(seq :str) -> str:
        seq = seq.rstrip(tokenizer.pad_token)
        seq = without_suffix(seq, tokenizer.eos_token.rstrip(tokenizer.pad_token))
        if special_tokens_constants.bos_token is not None:
            seq = without_prefix(seq, special_tokens_constants.bos_token)
        return seq.strip()
    
    def postprocess_sequences(sequences):
        return [clean_output_seq(pred) for pred in sequences]
    
    def prepare_for_rouge(seqs):
        return ["\n".join(nltk.sent_tokenize(seq)) for seq in seqs]


    def compute_metrics(eval_preds: EvalPrediction):
        logger.info("__Computing metrics__")
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=True)

        # Some simple post-processing
        decoded_preds = postprocess_sequences(decoded_preds)
        decoded_labels = postprocess_sequences(decoded_labels)

        result = rouge_metric.compute(predictions=prepare_for_rouge(decoded_preds), 
                                      references=prepare_for_rouge(decoded_labels), 
                                      use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        # Compute exact match (predicate-level)
        em_results = exact_match_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result["exact_match"] = em_results['accuracy']
        # Compute element exact match (QA level)
        eem_results = element_exact_match_metric.compute(predictions=decoded_preds, references=decoded_labels, separator=special_tokens_constants.separator_output_pairs)
        result["QA_exact_match_P"] = eem_results['precision']
        result["QA_exact_match_R"] = eem_results['recall']
        result["QA_exact_match_F1"] = eem_results['f1']
        wh_a_em_results = wh_answer_exact_match_metric.compute(predictions=decoded_preds, references=decoded_labels, 
                                                               qa_pairs_sep=special_tokens_constants.separator_output_pairs, qa_sep=special_tokens_constants.separator_output_question_answer)
        result["Wh_and_answer_EM_P"] = wh_a_em_results['precision']
        result["Wh_and_answer_EM_R"] = wh_a_em_results['recall']
        result["Wh_and_answer_EM_F1"] = wh_a_em_results['f1']
        
        # log to wandb some of the evaluation measures
        measures_to_send_wandb = ["rouge1", "rouge2", "gen_len", "exact_match", 
                                  "QA_exact_match_P", "QA_exact_match_R", "QA_exact_match_F1",
                                  "Wh_and_answer_EM_P", "Wh_and_answer_EM_R", "Wh_and_answer_EM_F1"]
        wandb_results = {k:v for k,v in result.items() if k in measures_to_send_wandb}
        wandb.log(wandb_results)
        
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    if model_args.debug_mode:
        if training_args.do_train:
            train_dataset = train_dataset.shuffle(seed=42).select(range(5))
            validation_dataset = validation_dataset.shuffle(seed=42).select(range(5))
        if training_args.do_eval:
            evaluation_dataset = evaluation_dataset.shuffle(seed=42).select(range(5))
        if training_args.do_predict:
            test_dataset = test_dataset.shuffle(seed=42).select(range(5))

    # Modify training_args before initializing Trainer, to allow evaluation during training
    # training_args.evaluation_strategy = transformers.trainer_utils.IntervalStrategy.STEPS # 'NO', 'EPOCH' or 'STEPS'
    # training_args.eval_steps = 500       

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset \
            if training_args.do_train or (training_args.do_eval and data_args.do_eval_on=="validation") \
            else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        _clean_mem()
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Prepare constrained generation
    if model_args.constrain_generation:
        from seq2seq_constrained_decoding.constrained_decoding.dfa import DFA
        from seq2seq_constrained_decoding.constrained_decoding.qasrl_constrained_decoding import get_qasrl_full_sequence_dfa
        from seq2seq_constrained_decoding.constrained_decoding.dfa_decoding import set_decoding_to_dfa_constrained
        
        # just naviely for testing 
        def dfa_factory(token_ids): 
            sentence = tokenizer.decode(token_ids, skip_special_tokens=True)
            sentence = preprocessor.reverse_input_preprocessing(sentence)
            return get_qasrl_full_sequence_dfa(sentence, tokenizer, special_tokens_constants)
        # enable special dfa-constrained beam search
        logger.info("Applying constrained decoding...")
        set_decoding_to_dfa_constrained(model, dfa_factory=dfa_factory, tokenizer=tokenizer)
        data_args.num_beams = data_args.num_beams or 2 # must enable beam search to utilize DFA-constrained decoding
    
    # file names for saving predictions - raw, validly parsed, & invalids   
    if "output_file" in training_args.__dict__:
        output_prediction_file = training_args.output_file
    else:
        output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.csv")
    invalid_output_prediction_file = os.path.join(training_args.output_dir, "invalid_generated_predictions.csv")
    raw_prediction_sequences_file = os.path.join(training_args.output_dir, "raw_generated_predictions.csv")

    """ 
    Inference (pre-evaluation) Procedure (dev/test) - parse predictions to qasrl format, save outputs to files, log,  
    """
    def decode_and_parse_predictions(dataset, predictions) -> pd.DataFrame:
        # assuming `predictions` is already textual, obtained by 
        #  tokenizer.batch_decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True) 
        # TODO copy from "do_predict" below
        predictions = [clean_output_seq(p) for p in predictions]
        input_seqs = [clean_output_seq(tokenizer.decode(dataset[i]['input_ids']))
                      for i in range(len(dataset))]
        
        #TODO future: remove non-QASRL segments of the output sequences 
        #  e.g. depends on data_args.learn_predicate_type
        
        # save raw output
        raw_out_df_dict = {"input": input_seqs, "predicted output": predictions}
        if 'labels' in dataset[0]:
            gold_out_seqs = [clean_output_seq(tokenizer.decode(dataset[i]['labels']))
                                for i in range(len(dataset))]
            raw_out_df_dict["gold output"] = gold_out_seqs
        raw_out_df = pd.DataFrame(raw_out_df_dict)
        raw_out_df.to_csv(raw_prediction_sequences_file, index=False)
        wandb.save(raw_prediction_sequences_file)
        # Log example instance 
        example_index = 0
        logger.info(f"\n\n**      Example (idx: {example_index}):      **")
        logger.info(raw_out_df.iloc[example_index])
        
        # count num of QAs per predicted sequence (= predicate) using QA_PAIR_SEP
        n_QAs_per_instance = np.array([1 + pred_seq.count(special_tokens_constants.separator_output_pairs) 
                                       for pred_seq in predictions])
        wandb.log({"Mean #-QAs": n_QAs_per_instance.mean()})
        wandb.run.summary["Mean #-QAs"] = n_QAs_per_instance.mean()
        
        # parse raw output; receive valid and invalid QAs 
        predicted_QAs: List[QuestionAnswer]
        invalid_qa_subseqs: List[str] 
        predicted_QAs, invalid_qa_subseqs = strings_to_objects_parser.to_qasrl_gs_csv_format(dataset, predictions)
        
        # Log invalid output sequences
        overall_n_qa_subseqs = len(predicted_QAs) + len(invalid_qa_subseqs)
        invalid_output_rate = len(invalid_qa_subseqs)/overall_n_qa_subseqs
        logger.info(f"Number of invalid (mal-structued) predicted output QAs: {len(invalid_qa_subseqs)} (%{100*invalid_output_rate:.1f})"
                    f"\n  Saving them into {invalid_output_prediction_file}")
        wandb.log({"invalid output QA rate overall": invalid_output_rate})
        wandb.run.summary["invalid output QA rate overall"] = invalid_output_rate
        invalid_pred_df = pd.DataFrame(invalid_qa_subseqs, columns=["Error-type", "output"])
        invalid_pred_df.to_csv(invalid_output_prediction_file, index=False)
        wandb.save(invalid_output_prediction_file)
        invalid_types_relative_frequency = invalid_pred_df["Error-type"].value_counts()/len(predictions)
        errors_log_dict = {f"invalid output QA rate - error type: {error_type}": relative_frequency
                            for error_type, relative_frequency in invalid_types_relative_frequency.items()}
        wandb.log(errors_log_dict)
        wandb.run.summary.update(errors_log_dict)
        wandb.log({"invalid output QA rate by type": invalid_types_relative_frequency.to_frame().transpose()})
    
        # Save parsed predictions in qasrl csv format                     
        df_parsed_predictions = pd.DataFrame([x.to_dict() for x in predicted_QAs])
        if len(df_parsed_predictions)>0 and "sentence" in dataset.column_names:
            qasrl_id2sent = {r["qasrl_id"]:r["sentence"] for r in dataset}
            df_parsed_predictions['sentence'] = df_parsed_predictions['qasrl_id'].apply(qasrl_id2sent.get) # dataset['qasrl_id']
        logger.info(f"Saving predictions into {output_prediction_file}...")
        if len(df_parsed_predictions) == 0:
            # write empty csv with correct header
            header = "qasrl_id,verb_idx,verb,question,answer,answer_range,verb_form,wh,aux,subj,obj,prep,obj2,is_negated,is_passive,sentence"
            with open(output_prediction_file, "w") as fout:
                fout.write(header)
        else:
            df_parsed_predictions.to_csv(output_prediction_file, index=False) 
        wandb.save(output_prediction_file)
        
        return df_parsed_predictions
    
    
    results = {}
    if training_args.do_eval:
        _clean_mem()
        logger.info("*** Evaluate ***")

        evaluation_predict_results = trainer.predict(
            evaluation_dataset,
            metric_key_prefix="eval",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = evaluation_predict_results.metrics
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(evaluation_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(evaluation_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        if trainer.is_world_process_zero():
            logger.info(f"length of predictions: {len(evaluation_predict_results.predictions)}")
            predictions = tokenizer.batch_decode(
                evaluation_predict_results.predictions, skip_special_tokens=False, clean_up_tokenization_spaces=True
            )
            
            df_parsed_predictions = decode_and_parse_predictions(evaluation_dataset, predictions)
            df_gold = pd.DataFrame(orig_evaluation_dataset)
            eval_measures = run_evaluation.run_qanom_evaluation(df_parsed_predictions, df_gold)
            unlabelled_arg, labelled_arg, unlabelled_role = eval_measures
            eval_measures_dict = {
                "Unlabled Arg f1": unlabelled_arg.f1(),
                "Unlabled Arg precision": unlabelled_arg.prec(),
                "Unlabled Arg recall": unlabelled_arg.recall(),
                "Labled Arg f1": labelled_arg.f1(),
                "Labled Arg precision": labelled_arg.prec(),
                "Labled Arg recall": labelled_arg.recall(),
                "Role f1": unlabelled_role.f1(),
                "Role precision": unlabelled_role.prec(),
                "Role recall": unlabelled_role.recall(),
            }
            run_evaluation.print_evaluations(unlabelled_arg, labelled_arg, unlabelled_role)
            wandb.log(eval_measures_dict)
            results.update(eval_measures_dict) 


    # Inference (Prediction on test)
    if training_args.do_predict:
        _clean_mem()
        logger.info("*** Predict ***")

        #TODO check what happens here when do_inference_without_labels is False, how does the decoder begins? does it use <pad> from the ground-turth `labels`?
        if do_inference_without_labels and model_args.model_type == 't5':
            # T5 decoders expect to be initialized with the pad_token
            test_dataset = test_dataset.add_column('labels', [[tokenizer.pad_token_id]] * test_dataset.num_rows)
        predict_results = trainer.predict(
            test_dataset,
            metric_key_prefix="predict",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = predict_results.metrics
        max_test_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(test_dataset)
        )
        metrics["predict_samples"] = min(max_test_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                logger.info(f"length of predict_results.predictions: {len(predict_results.predictions)}")
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=False, clean_up_tokenization_spaces=True
                )
                
                # this will also save the output csv to `output_prediction_file`
                df_parsed_predictions = decode_and_parse_predictions(test_dataset, predictions)
                  

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)
        
    # run.finish()

    return results, run


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
