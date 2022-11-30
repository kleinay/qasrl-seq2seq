# QASRL Seq2Seq
Parse QASRL using Seq2Seq technologies.

This repository contains code for our experiments and analysis in the paper "QASem Parsing - Text to Text modeling of QA-based Semantics" (Klein et. al., EMNLP 2022).  

## Important files and paths

* `notebook.py` - convenient entry point, with 3 relevant API functions:
  * `full_experiment` - train and evaluate a new model. See a few example calls at [small_experiments.py](small_experiments.py).
  * `load_and_evaluate` - evaluate an existing model.
  * `load_and_predict` - run inference on new texts using an existing model.
* `run_parsing_model.py` - main file for train, evaluate and predict logics. It is adapted from Huggingface `run_summarization.py` script but quite havily refactored. 
* `pipeline.py` - introduces a wrapper pipeline class for easily running text-to-text inference. Can load a model from Huggingface Hub or from a local directory. 
* [QASRL State machine example](https://github.com/eranhirs/qasrl_state_machine_example) - a docker project running the original QASRL State Machine, used in order to analyze the question format for evaluation. Eventually, for better efficiency we use our own machinery for that (see below the `seq2seq_constrained_decoding` submodule).
* `evaluation` - main file for the evaluation logics. Since we compare with previous QA-SRL and QANom works, we utilize their evaluation functionality from their own pakages, and just adapt the input format per package.

*Sub-modules:*

* `qasrl-gs` - used for evaluation of the results, according to Roit et al., 2020. We call this "qasrl evaluation methodology" throughout the repo.
* `QANom` - used for evaluation of the results, according to Klein et al., 2020. We call this "qanom evaluation methodology" throughout the repo.
* `seq2seq_constrained_decoding` - was used in some preliminary experiments to constrain model generation according to QASRL output sequence format. We end up not using constrained decoding as it introduced noise related to the mismtach between words and tokens. The Deterministic Finite Autmoata (`DFA`) defining QASRL question format is still being used in `dfa_fill_qasrl_slots.py` for analyzing the question format (i.e. decomposing the outout question string into the 7-slot template defining a valid QASRL question).

## Training New Models

There are two recommneded ways to run a training experiment, in which you can select a certain set of hyperparameters (including training dataset, pretrained model, and various modeling-related options), train (=fine-tune) a new model, and then evaluate it on the dev or test set.

### Option-1: `full_experiment` function

Inside `notebook.py` there is the `full_experiment` function.  
This function basically prepares the desired arguments for the `run_parsing_mode` script based on more simple parameters. It also wraps it with `wandb` utilities for logging the experiment. You can specify any seq2seq model from Huggingface Model Hub as `model_type`.

### Option-2: Through WandB sweeps

Commonly it is desired to run a series of training experiments with different hyperparamters, for tuning those hyperparamters to maxmize performance. In this project we use WandB and its [sweep utility](https://docs.wandb.ai/guides/sweeps) for performing hyperparamter search and other experiments.
A sweep (i.e. one hyperparameter search experiment) is defined by a YAML file - you can find some examples in the `sweeps/` directory.

To run a sweep, run:

```bash
wandb sweep sweep_config.yaml
```

And then initialize agents to execute actual calls to the command defined by *sweep_config.yaml* --- `wandb agent [sweep-id]`.  

In our context, you can take `sweeps/main_basic_grid.yaml` as a starting point. For training a single-task QASRL / QANom model, you should set `program` to be `run_parsing_model.py`, and provide the desired `--dataset_name`. Alternatively, for training a joint QASRL-QANom model on both datasets, comment out the `--dataset_name` arguments and set `program` to `joint_train_experiment.py` which will specify the dataset for you (while duplicating QANom train set by a factor of 14 to match the size of QASRL train set).


## Evaluating New Models

The `full_experiment` function is trigerring evaluation after training by default. It does so by sending the `--do_eval` argument to the main `run_parsing_model.py` script. You can set specify `do_eval_on` for evaluating over the "validation" set or "test" set. 

In order to run evaluation on a given trained model, call the `notebook.load_and_evaluate` function. 

If you wish to perform evaluation through WandB sweeps (e.g. for experiemting with decoding options, e.g. "num_beams"), take a look at `sweeps/inference.yaml` for an example. 

## Inference

The models from our paper are uploaded to [Huggingface](https://huggingface.co/kleinay). We wrapped the model with an easy-to-use API, utilizing Huggignface [Pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines), in an independant repository termed [QASem](https://github.com/kleinay/QASem); see documentation there. It is also available through `pip install qasem`.  

Locally, you can use the `notebook.load_and_predict` function, will can take a text file of instances (sentences marked with a predicate) or a Dataset as input. We recommend using the `pipeline.QASRL_Pipeline` wrapper class. 

## Cite

@TBD
