# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from typing import Literal, Any, Dict, List
from IPython import get_ipython
from traitlets.traitlets import default
import wandb

# Imports

from run_summarization import main
from run_evaluation import evaluate_qasrl, evaluate_qanom, print_evaluations
import os
import sys
import json
import datetime
import subprocess
from itertools import product
from utils import setup_wandb

# General variables
now = lambda: datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
system = get_ipython().system if get_ipython() else os.system

# tmp_dir = os.environ.get("TMPDIR", "/tmp")
tmp_dir = os.environ.get("HOME") + "/tmp"

### Data params

qasrl_2015_params = ['--dataset_name', 'qa_srl']
qasrl_2018_params = ['--dataset_name', 'biu-nlp/qa_srl2018']
qasrl_2020_params = [
    "--train_file", "qasrl_gs/data/gold/all.dev.combined.csv",
    "--validation_file", "qasrl_gs/data/gold/all.dev.combined.csv",
    "--test_file", "qasrl_gs/data/gold/all.test.combined.csv",
    "--text_column", "sentence", 
    "--summary_column", "answer"
]
qanom_params = ['--dataset_name', 'biu-nlp/qanom']
joint_qasrl_params = ['--dataset_name', '{train: biu-nlp/qanom, kleinay/qa_srl2018; validation: biu-nlp/qa_srl2020; test: biu-nlp/qa_srl2020}']  
joint_qanom_params = ['--dataset_name', '{train: biu-nlp/qanom, kleinay/qa_srl2018; validation: biu-nlp/qanom; test: biu-nlp/qanom}']  

qadiscourse_params = ['--dataset_name', 'biu-nlp/qa_discourse']

### Model params

def get_model_dir(model_type: str, train_task: str = None, directory_switch: str = None) -> str:
    model_dir = f"{tmp_dir}/{model_type}-tst-summarization"
    if train_task:
        model_dir += f"/{train_task}" 
    if directory_switch:
        model_dir += f"/{directory_switch}"
    return model_dir 

def set_model_params(model_type: str, model_dir: str = None, load_pretrained = True, source_prefix = 'summarize: ', **kwargs) -> List[str]:
    """
    source_prefix string is an actual generic prompt; for more controlled prompt behaiour, specify behaviour name enclosed by <>,
    For example, source_prefix="<predicate-type>" makes the prompt dependent on predicate type.
    These should be processed differently by the `preprocess_function__questions_answers` function in `run_summarization.py`.
    """
    sys.argv += ['--output_dir', model_dir]
    
    if model_type in ("t5", "t5-large"):
        model_name_to_load = "t5-small" if model_type=="t5" else model_type
        sys.argv += ['--model_name_or_path', model_name_to_load if load_pretrained else model_dir,
                     '--model_type', 't5',
                     '--source_prefix', source_prefix,
                     ]

        os.environ["T5_MODEL_DIR"] = model_dir      
    
    elif model_type in ("bart", "bart-large"):
        model_name_to_load = 'facebook/' + ('bart-base' if model_type=="bart" else model_type)
        sys.argv += ['--model_name_or_path', model_name_to_load if load_pretrained else model_dir,
                     '--model_type', 'bart',
                     ]
        os.environ["BART_MODEL_DIR"] = model_dir      
    else:
        raise ValueError(f"model_type doesn't exist ; model_type {model_type}")
    
# # Train, predict and evaluate ***********************

# %%
# ### (0) Run config

# # model_type = "bart"
# model_type = "t5"

# model_dir = t5_model_dir if model_type == "t5" else bart_model_dir

# # qasrl_train_dataset = "2015"
# # qasrl_train_dataset = "2018"
# qasrl_train_dataset = "qanom"

# # qasrl_test_dataset = "2015"
# # qasrl_test_dataset = "2020"
# qasrl_test_dataset = "qanom"

# train_epochs = 3

def get_default_kwargs() -> Dict[str, Any]:
    default_boolean_args = dict(overwrite_output_dir=True,
                                predict_with_generate=True,
                                debug_mode=False,)
    default_non_boolean_args = dict(per_device_train_batch_size=12,
                                    per_device_eval_batch_size=12,
                                    logging_steps=200)  
    defaults = dict(default_boolean_args, **default_non_boolean_args)
    return defaults
  

def linearize_kwargs_to_send_model(kwargs) -> List[str]:
    kwargs_to_send = []
    for key, value in kwargs.items():
        if type(value) == bool:
            if value is True:
                kwargs_to_send.extend([f'--{key}'])
        else:
            if value is not None:
                kwargs_to_send.extend([f'--{key}', f'{value}'])
    return kwargs_to_send

# %%
# ### (1) Train ********************************
def train(model_type, 
          qasrl_train_dataset, 
          train_epochs,
          model_dir,
          wandb_run_name=None, # a way to name experiment in wandb 
          limit_train_data=None,
          limit_eval_data=None,
          **kwargs):

    wandb_run_name = wandb_run_name or f"{now()}_{model_type}_train-on-{qasrl_train_dataset}_{train_epochs}-ep"
    sys.argv = [
        'run_summarization.py',
        '--do_train',
        '--do_eval',
        '--num_train_epochs', f'{train_epochs}',
        '--report_to', 'wandb',
        '--wandb_run_name', wandb_run_name,
        # '--debug_mode',
        # '--n_gpu', '[5,6]'
    ]
    
    if "limit_train_data" in kwargs and "overwrite_cache" not in kwargs:
        kwargs["overwrite_cache"] = True
        
        
    defaults = get_default_kwargs()
    kwargs = dict(defaults, **kwargs)   # integrate deafult kwargs values and override them by **kwargs
    
    kwargs_to_send = linearize_kwargs_to_send_model(kwargs)
    sys.argv += kwargs_to_send
    
    os.makedirs(model_dir, exist_ok=True)
    load_pretrained_model: bool = kwargs["overwrite_output_dir"]
    set_model_params(model_type, model_dir, load_pretrained=load_pretrained_model, **kwargs)
    
    if limit_train_data:
        sys.argv += ['--limit_train_data', f'{limit_train_data}']
    if limit_eval_data:
        sys.argv += ['--limit_eval_data', f'{limit_eval_data}']

    if qasrl_train_dataset == "2015":
        sys.argv.extend(qasrl_2015_params)
    elif qasrl_train_dataset == "2018":
        sys.argv.extend(qasrl_2018_params)
    elif qasrl_train_dataset == "qanom":
        sys.argv.extend(qanom_params)
    elif qasrl_train_dataset == "joint_qasrl":
        sys.argv.extend(joint_qasrl_params)
    elif qasrl_train_dataset == "joint_qanom":
        sys.argv.extend(joint_qanom_params)
    elif qasrl_train_dataset == "qadiscourse":
        sys.argv.extend(qadiscourse_params)
    else:
        raise ValueError(f"qasrl_train_dataset doesn't exist ; qasrl_train_dataset {qasrl_train_dataset}")

    _, run = main()

    # setup_wandb(True, run.name)
    wandb.save(f"{model_dir}/pytorch_model.bin")

    return run


# run = train(model_type, qasrl_train_dataset, train_epochs)
# %%
# ### (2) Predict ***************************

# !python run_summarization.py --model_name_or_path $TMPDIR/tst-summarization --do_predict --dataset_name qa_srl --output_dir $TMPDIR/tst-summarization --source_prefix "summarize: " --predict_with_generate
def predict(model_type, qasrl_test_dataset, model_dir, run=None, **kwargs):
    sys.argv = [
        'run_summarization.py',
        '--do_predict',
        '--predict_with_generate',
        '--eval_accumulation_steps', '10',  # Necessary to avoid OOM where all predictions are kept on one GPU    
        '--report_to', 'wandb',
        '--wandb_run_name', run.name if run else None
    ]
    
    if "limit_eval_data" in kwargs and "overwrite_cache" not in kwargs:
        kwargs["overwrite_cache"] = True
    if qasrl_test_dataset == "qadiscourse":
        kwargs["preprocess_input_func"] = "qadiscourse_input"
        kwargs["preprocess_output_func"] = "qadiscourse_output"
        
        
    defaults = get_default_kwargs()
    kwargs = dict(defaults, **kwargs)   # integrate deafult kwargs values and override them by **kwargs
    
    kwargs_to_send = linearize_kwargs_to_send_model(kwargs)
    sys.argv.extend(kwargs_to_send)
        
    # add model name for prediction
    set_model_params(model_type, model_dir, load_pretrained=False, **kwargs)  # always load the local model from training

    if qasrl_test_dataset == "2015":
        sys.argv.extend(qasrl_2015_params)
    elif qasrl_test_dataset == "2020":
        sys.argv.extend(qasrl_2020_params)
    elif qasrl_test_dataset == "qanom":
        sys.argv.extend(qanom_params)
    elif qasrl_test_dataset == "qadiscourse":
        sys.argv.extend(qadiscourse_params)
    else:
        raise ValueError(f"qasrl_test_dataset doesn't exist ; qasrl_test_dataset {qasrl_test_dataset}")

    results, run = main(generate_sentence_column_in_prediction= qasrl_test_dataset in ["qanom", "2020"])
    return run


# predict(model_type, qasrl_test_dataset, run)

# %% 
# ### (3) Run state machine using docker, for parsing the predicted questions into 7 slot format

def decode_into_qasrl(model_dir, test_dataset):
    "Call qasrl_state_machine_example within docker container to verify question formats match QASRL slots. "
    os.environ["MODEL_DIR"] = model_dir 
    if test_dataset != "qanom":
        shell_command = 'docker run -it -v "${MODEL_DIR}:/data" -v "$(pwd)/../qasrl_bart/qasrl_gs/data/sentences/:/sentences_data" --rm --name qasrl-automaton hirscheran/qasrl_state_machine_example "file" "/data/generated_predictions.csv" "/data/output_file.csv"'
    else:
        shell_command = 'docker run -it -v "${MODEL_DIR}:/data" --rm --name qasrl-automaton hirscheran/qasrl_state_machine_example "file" "/data/generated_predictions.csv" "/data/output_file.csv"'
    # execute; redirect output
    with open(f"{model_dir}/qasrl_state_machine_output.txt", "w") as fout:
        completed_process = subprocess.run(shell_command, shell=True, capture_output=True, text=True) #stdout=subprocess.PIPE,
        if completed_process.stdout:
            fout.write(completed_process.stdout)
        if completed_process.stderr:
            print("~!!!~   Standard Error from running docker:   ~!!!~ \n", completed_process.stderr)
        completed_process.check_returncode()  # Raise exception if subprocess ended with error
    

# decode_into_qasrl(model_dir, qasrl_test_dataset)

# %% 
# ### (4) Evaluate
def evaluate(model_dir, qasrl_test_dataset, wandb_run_name=None):
    if qasrl_test_dataset == "2020":
        evals = evaluate_qasrl(model_dir, "qasrl_gs/data/gold/all.test.combined.csv", f"{model_dir}/output_file.csv", None, wandb_run_name)
    elif qasrl_test_dataset == "qanom":
        evals = evaluate_qanom(model_dir, wandb_run_name)
    elif qasrl_test_dataset == "qadiscourse":
        # TODO implement evaluation module for qadiscourse
        raise NotImplementedError()
    else:
        raise NotImplementedError(f"evaluate function not implemented for '{qasrl_test_dataset}' test dataset")
    print_evaluations(*evals)
    
    return evals

# evaluate(model_dir, qasrl_test_dataset)


def print_invalid_dist(model_type):
    import pandas as pd
    df=pd.read_csv(f"/home/nlp/kleinay/tmp/{model_type}-tst-summarization/invalid_generated_predictions.csv") 
    print(f"Overall: {len(df)}\n")
    print(df["Error-type"].value_counts(), "\n")
    print(df["Error-type"].value_counts() / len(df))

def full_experiment(model_type: Literal["bart", "t5", "t5-large"] = "bart", 
                    train_dataset: Literal["qanom", "qasrl", "qadiscourse"] = "qanom", 
                    test_dataset: Literal["qanom", "qasrl", "qadiscourse"] = "qanom",
                    dir_switch: str = None,
                    train_epochs: int = 10,
                    description: str = "",
                    finish_wandb = True,
                    fp16 = True,
                    **kwargs):
    
    model_dir = get_model_dir(model_type, train_dataset, dir_switch)
    experiment_kwargs = dict(description=description,
                             model_type=model_type, 
                             train_dataset=train_dataset,
                             test_dataset=test_dataset,
                             train_epochs=train_epochs,
                             fp16=fp16,
                             **kwargs)
    
    qasrl_train_dataset = "2018" if train_dataset == "qasrl" else train_dataset
    qasrl_test_dataset = "2020" if test_dataset == "qasrl" else test_dataset

    run = train(model_type, qasrl_train_dataset, train_epochs, model_dir, **kwargs)
    wandb.config.update(experiment_kwargs, allow_val_change=True)
    # save configuration of experiment
    with open(f"{model_dir}/experiment_kwargs.json", "w") as fout:
        json.dump(experiment_kwargs, fout)
    wandb.save(f"{model_dir}/experiment_kwargs.json")
    run = predict(model_type, qasrl_test_dataset, model_dir=model_dir, run=run, **kwargs)
    decode_into_qasrl(model_dir, qasrl_test_dataset)
    unlabelled_arg, labelled_arg, unlabelled_role = evaluate(model_dir, qasrl_test_dataset, wandb_run_name=run.name)
    
    wandb.save(f"{model_dir}/*.csv")
    wandb.save(f"{model_dir}/*.json")
    wandb.save(f"{model_dir}/*.txt")
    wandb.log({
        "Unlabled Arg f1": unlabelled_arg.f1(),
        "Unlabled Arg precision": unlabelled_arg.prec(),
        "Unlabled Arg recall": unlabelled_arg.recall(),
        "Labled Arg f1": labelled_arg.f1(),
        "Labled Arg precision": labelled_arg.prec(),
        "Labled Arg recall": labelled_arg.recall(),
        "Role f1": unlabelled_role.f1(),
        "Role precision": unlabelled_role.prec(),
        "Role recall": unlabelled_role.recall(),
               })
    
    if finish_wandb:
        wandb.finish()
    return unlabelled_arg, labelled_arg, unlabelled_role

""" 
kwargs for `full_experiment` (and `run_summarization` script):

`preprocess_input_func`: str. Options:
    "input" (default) - encode sentence by repeating the predicate in end of sentence. Also embeds the verb_form if exists (was shown to improve).
    "input_predicate_marker" - encode sentence by prefixing the predicate with a marker. Also embeds the verb_form if exists (was shown to improve).

...      

"""

if __name__ == "__main__":
    # model_type = "bart"
    model_type = "t5"

    model_dir = get_model_dir(model_type)

    # qasrl_train_dataset = "2015"
    # qasrl_train_dataset = "2018"
    # qasrl_train_dataset = "qanom"
    qasrl_train_dataset = "joint_qanom"

    # qasrl_test_dataset = "2015"
    # qasrl_test_dataset = "2020"
    qasrl_test_dataset = "qanom"

    train_epochs = 30
    
    run = train(model_type, qasrl_train_dataset, train_epochs, model_dir,
                overwrite_output_dir=True, 
                wandb_run_name=f"{now()}_{model_dir}_{qasrl_train_dataset}",
                preprocess_input_func="input_predicate_marker")
    predict(model_type, qasrl_test_dataset, model_dir, run)
    decode_into_qasrl(model_dir, qasrl_test_dataset)
    unlabelled_arg, labelled_arg, unlabelled_role = evaluate(model_dir, qasrl_test_dataset, wandb_run=run)