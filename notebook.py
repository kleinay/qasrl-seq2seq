# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from argparse import Namespace
from shutil import Error
from typing import Literal, Any, Dict, List, Optional
from pathlib import Path
from IPython import get_ipython
from traitlets.traitlets import default
import wandb

# Imports

from run_summarization import main
from run_evaluation import run_qasrl_gs_evaluation, run_qanom_evaluation, evaluate_qadiscourse, print_evaluations, write_qasrl_evaluation_to_file
import os
import sys
import json
import datetime
import subprocess
from itertools import product
import pandas as pd
import utils 

# General variables
now = lambda: datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
system = get_ipython().system if get_ipython() else os.system

# tmp_dir = os.environ.get("TMPDIR", "/tmp")
tmp_dir = os.environ.get("HOME") + "/tmp"

### Data params

qasrl_2015_params = ['--dataset_name', 'qa_srl']
qasrl_2018_params = ['--dataset_name', 'biu-nlp/qa_srl2018']
qasrl_2020_params = [
    # "--train_file", "qasrl_gs/data/gold/all.dev.combined.csv",
    # "--validation_file", "qasrl_gs/data/gold/all.dev.combined.csv",
    # "--test_file", "qasrl_gs/data/gold/all.test.combined.csv",
    '--dataset_name', 'biu-nlp/qa_srl2020'
]
qasrl_params = ['--dataset_name', '{train: biu-nlp/qa_srl2018; validation: biu-nlp/qa_srl2020; test: biu-nlp/qa_srl2020}']
qanom_params = ['--dataset_name', 'biu-nlp/qanom']

def get_joint_params(test_task, qanom_factor=1) -> List[str]:
    test_dataset = "biu-nlp/qanom" if test_task=="qanom" else "biu-nlp/qa_srl2020"
    qanom_factor_str = f"{qanom_factor} *" if qanom_factor != 1 else ""
    return ['--dataset_name', '{train:' f'{qanom_factor_str} biu-nlp/qanom, biu-nlp/qa_srl2018; validation: {test_dataset}; test: {test_dataset}' '}']

qadiscourse_params = ['--dataset_name', 'biu-nlp/qa_discourse']

### Model params

def get_model_dir(model_type: str, train_task: str = None, directory_switch: str = None) -> str:
    model_dir = f"{tmp_dir}/{model_type}-tst-summarization"
    if train_task:
        model_dir += f"/{train_task}" 
    if directory_switch:
        model_dir += f"/{directory_switch}"
    return model_dir 

def get_model_params(model_type: str, model_dir: str = None, load_pretrained = True, source_prefix = None) -> List[str]:
    """
    source_prefix string is an actual generic prefix; 
    in the source_prefix, the special "<predicate-type>" token translates into the `predicate_type` ("nominal" or "verbal").`.
    """
    args = ['--output_dir', model_dir]
    
    if "t5" in model_type.lower():
        assert source_prefix
        model_name_to_load = "t5-small" if model_type=="t5" else model_type
        args += ['--model_name_or_path', model_name_to_load if load_pretrained else model_dir,
                     '--model_type', 't5',
                     '--source_prefix', source_prefix,
                     ]

        os.environ["T5_MODEL_DIR"] = model_dir      
    
    elif model_type in ("bart", "bart-large"):
        model_name_to_load = 'facebook/' + ('bart-base' if model_type=="bart" else model_type)
        args += ['--model_name_or_path', model_name_to_load if load_pretrained else model_dir,
                     '--model_type', 'bart',
                     ]
        os.environ["BART_MODEL_DIR"] = model_dir      
    else:
        raise ValueError(f"model_type '{model_type}' is not supported!")
    return args
    
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
                                debug_mode=False,
                                append_verb_form=True,
                                use_bilateral_predicate_marker=True,
                                fp16=True,
                                # under debug:
                                load_best_model_at_end=True,
                                )
    default_non_boolean_args = dict(do_eval_on="validation",
                                    per_device_train_batch_size=8,
                                    per_device_eval_batch_size=8,
                                    save_strategy="steps",
                                    logging_strategy="steps",
                                    evaluation_strategy="steps",
                                    logging_steps=500,
                                    eval_steps=500,
                                    save_steps=500,
                                    metric_for_best_model="eval_loss", # "eval_Wh_and_answer_EM_F1",   # default: "eval_loss"
                                    predicate_marker_type="generic",
                                    )  
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
# def train(model_type, 
#           qasrl_train_dataset, 
#           train_epochs,
#           model_dir,
#           wandb_run_name=None, # a way to name experiment in wandb 
#           limit_train_data=None,
#           limit_eval_data=None,
#           qanom_joint_factor=1, # how many times to duplicate qanom training set in joint training 
#           **kwargs):

#     wandb_run_name = wandb_run_name or f"{now()}_{model_type}_train-on-{qasrl_train_dataset}_{train_epochs}-ep"
#     sys.argv = [
#         'run_summarization.py',
#         '--do_train',
#         '--do_eval',
#         '--num_train_epochs', f'{train_epochs}',
#         '--report_to', 'wandb',
#         '--wandb_run_name', wandb_run_name,
#         # '--debug_mode',
#         # '--n_gpu', '[5,6]'
#     ]
    
#     if "limit_train_data" in kwargs and "overwrite_cache" not in kwargs:
#         kwargs["overwrite_cache"] = True
    
#     if "batch_size" in kwargs and kwargs["batch_size"] is not None:
#         kwargs["per_device_train_batch_size"] = kwargs["batch_size"]
#         kwargs["per_device_eval_batch_size"] = kwargs["batch_size"]
#         kwargs.pop("batch_size")
        
#     defaults = get_default_kwargs()
#     kwargs = dict(defaults, **kwargs)   # integrate deafult kwargs values and override them by **kwargs
    
#     kwargs_to_send = linearize_kwargs_to_send_model(kwargs)
#     sys.argv += kwargs_to_send
    
#     os.makedirs(model_dir, exist_ok=True)
#     load_pretrained_model: bool = kwargs["overwrite_output_dir"]
#     sys.argv += get_model_params(model_type, model_dir, load_pretrained=load_pretrained_model, **kwargs)
    
#     if limit_train_data:
#         sys.argv += ['--limit_train_data', f'{limit_train_data}']
#     if limit_eval_data:
#         sys.argv += ['--limit_eval_data', f'{limit_eval_data}']

#     if qasrl_train_dataset == "2015":
#         sys.argv.extend(qasrl_2015_params)
#     elif qasrl_train_dataset == "2018":
#         sys.argv.extend(qasrl_2018_params)
#     elif qasrl_train_dataset == "qanom":
#         sys.argv.extend(qanom_params)
#     elif qasrl_train_dataset == "joint_qasrl":
#         sys.argv.extend(get_joint_params("qasrl", qanom_joint_factor))
#     elif qasrl_train_dataset == "joint_qanom":
#         sys.argv.extend(get_joint_params("qanom", qanom_joint_factor))
#     elif qasrl_train_dataset == "qadiscourse":
#         sys.argv.extend(qadiscourse_params)
#     else:
#         raise ValueError(f"qasrl_train_dataset doesn't exist ; qasrl_train_dataset {qasrl_train_dataset}")

#     _, run = main()

#     # setup_wandb(True, run.name)
#     wandb.save(f"{model_dir}/pytorch_model.bin")

#     return run


# run = train(model_type, qasrl_train_dataset, train_epochs)
# %%
# ### (2) Predict ***************************

# !python run_summarization.py --model_name_or_path $TMPDIR/tst-summarization --do_predict --dataset_name qa_srl --output_dir $TMPDIR/tst-summarization --source_prefix "summarize: " --predict_with_generate
# def predict(model_type, qasrl_test_dataset, model_dir, wandb_run_name=None, **kwargs):
#     sys.argv = [
#         'run_summarization.py',
#         '--do_predict',
#         '--predict_with_generate',
#         '--eval_accumulation_steps', '10',  # Necessary to avoid OOM where all predictions are kept on one GPU    
#         '--report_to', 'wandb',
#         '--wandb_run_name', wandb_run_name or f"Predict {qasrl_test_dataset} test-set with model {model_dir}"
#     ]
    
#     if "limit_eval_data" in kwargs and "overwrite_cache" not in kwargs:
#         kwargs["overwrite_cache"] = True
#     if qasrl_test_dataset == "qadiscourse":
#         kwargs["preprocess_input_func"] = "qadiscourse_input"
#         kwargs["preprocess_output_func"] = "qadiscourse_output"
        
#     if "batch_size" in kwargs and kwargs["batch_size"] is not None:
#         kwargs["per_device_train_batch_size"] = kwargs["batch_size"]
#         kwargs["per_device_eval_batch_size"] = kwargs["batch_size"]
#         kwargs.pop("batch_size")    
        
#     defaults = get_default_kwargs()
#     kwargs = dict(defaults, **kwargs)   # integrate deafult kwargs values and override them by **kwargs
    
#     kwargs_to_send = linearize_kwargs_to_send_model(kwargs)
#     sys.argv += kwargs_to_send
        
#     # add model name for prediction
#     sys.argv += get_model_params(model_type, model_dir, load_pretrained=False, **kwargs)  # always load the local model from training

#     if qasrl_test_dataset == "2015":
#         sys.argv.extend(qasrl_2015_params)
#     elif qasrl_test_dataset == "2020":
#         sys.argv.extend(qasrl_2020_params)
#     elif qasrl_test_dataset == "qanom":
#         sys.argv.extend(qanom_params)
#     elif qasrl_test_dataset == "qadiscourse":
#         sys.argv.extend(qadiscourse_params)
#     else:
#         raise ValueError(f"qasrl_test_dataset doesn't exist ; qasrl_test_dataset {qasrl_test_dataset}")

#     results, run = main()
#     return run


# predict(model_type, qasrl_test_dataset, run)

# %% 
# ### (3) Run state machine using docker, for parsing the predicted questions into 7 slot format

# def decode_into_qasrl(model_dir, test_dataset):
#     #TODO should we rename `model_dir` to `output_dir`? must it be the real model dir?
#     "Call qasrl_state_machine_example within docker container to verify question formats match QASRL slots. "
#     model_dir = os.path.abspath(model_dir)
#     if wandb.run is None:
#         utils.setup_wandb(True, f"decoding {'/'.join(model_dir.split('/')[-3:])}")
#     # os.environ["MODEL_DIR"] = model_dir 
#     if test_dataset != "qanom":
#         shell_command = f'docker run -it -v "{model_dir}:/data" -v "$(pwd)/../qasrl_bart/qasrl_gs/data/sentences/:/sentences_data" --rm --name qasrl-automaton hirscheran/qasrl_state_machine_example "file" "/data/generated_predictions.csv" "/data/state_machine_output.csv"'
#     else:
#         shell_command = f'docker run -it -v "{model_dir}:/data" --rm --name qasrl-automaton hirscheran/qasrl_state_machine_example "file" "/data/generated_predictions.csv" "/data/state_machine_output.csv"'
#     # execute; redirect output
#     with open(f"{model_dir}/qasrl_state_machine_output.txt", "w") as fout:
#         completed_process = subprocess.run(shell_command, shell=True, capture_output=True, text=True) #stdout=subprocess.PIPE,
#         if completed_process.stdout:
#             fout.write(completed_process.stdout)
#         if completed_process.stderr:
#             print("~!!!~   Standard Error from running docker:   ~!!!~ \n", completed_process.stderr)
#             fout.write("~!!!~   Standard Error from running docker:   ~!!!~ \n" + completed_process.stderr)
#     wandb.save(f"{model_dir}/qasrl_state_machine_output.txt")
#     if completed_process.returncode != 0: # Raise exception if subprocess ended with error
#         print(f"Docker process failed (returncode=={completed_process.returncode}). \n"
#               f"Standard Error: \n\n{completed_process.stderr}\n" 
#               f"Run the command on console to see details: \n\n{shell_command}\n\n"
#               f"Alternatively, run this function from a python notebook: decode_into_qasrl('{model_dir}','{test_dataset}')")
#         raise ChildProcessError(f"Docker process failed with error code {completed_process.returncode}")
#     # copy state_machine_output file to have a non-root priviliges file (docker is run by root)
#     from shutil import copyfile
#     state_machine_output_fn = f"{model_dir}/decoded_output.csv" 
#     copyfile(f"{model_dir}/state_machine_output.csv", state_machine_output_fn)
        
#     # re-format qanom output according to qanom format 
#     if test_dataset == "qanom":
#         from qanom.annotations.common import read_annot_csv, save_annot_csv
#         df_parsed_outputs = read_annot_csv(state_machine_output_fn)
#         df_parsed_outputs = utils.reshape_qasrl_into_qanom(df_parsed_outputs, sentences_df=pd.read_csv(f"{model_dir}/generated_predictions.csv"))
#         save_annot_csv(annot_df=df_parsed_outputs, file_path=state_machine_output_fn)


def print_invalid_dist(model_type):
    import pandas as pd
    df=pd.read_csv(f"/home/nlp/kleinay/tmp/{model_type}-tst-summarization/invalid_generated_predictions.csv") 
    print(f"Overall: {len(df)}\n")
    print(df["Error-type"].value_counts(), "\n")
    print(df["Error-type"].value_counts() / len(df))


""" 
kwargs for `full_experiment` (and `run_summarization` script):

`preprocess_input_func`: str. Options:
    "input" (default) - encode sentence by repeating the predicate in end of sentence. Also embeds the verb_form if exists (was shown to improve).
    "input_predicate_marker" - encode sentence by prefixing the predicate with a marker. Also embeds the verb_form if exists (was shown to improve).

...      

"""
def full_experiment(model_type: str, # e.g. "bart", "t5", "t5-large", "iarfmoose/t5-base-question-generator", 
                    train_dataset: Literal["qanom", "qasrl", "joint_qasrl", "joint_qanom", "qadiscourse"], 
                    dir_switch: str = None,
                    train_epochs: int = 10,
                    description: str = "",
                    finish_wandb = True,
                    wandb_run_name = None,
                    qanom_joint_factor=1,
                    **kwargs):
    
    experiment_kwargs = dict(description=description,
                    model_type=model_type, 
                    train_dataset=train_dataset,
                    train_epochs=train_epochs,
                    qanom_joint_factor=qanom_joint_factor,
                    dir_switch=dir_switch,
                    **kwargs)
    
    model_dir = get_model_dir(model_type, train_dataset, dir_switch)
    wandb_run_name = wandb_run_name or f"{now()}_{model_type}_train-on-{train_dataset}_{train_epochs}-ep"
    args_to_send = [
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
        experiment_kwargs["overwrite_cache"] = True
    
    if "batch_size" in kwargs and kwargs["batch_size"] is not None:
        experiment_kwargs["per_device_train_batch_size"] = kwargs["batch_size"]
        experiment_kwargs["per_device_eval_batch_size"] = kwargs["batch_size"]
        experiment_kwargs.pop("batch_size")
        
    defaults = get_default_kwargs()
    experiment_kwargs = dict(defaults, **experiment_kwargs)   # integrate deafult kwargs values and override them by **kwargs
    
    kwargs_not_to_be_sent = ('description', 'train_dataset', 'train_epochs', 'qanom_joint_factor', 'dir_switch')
    kwargs_to_send = utils.dict_without_keys(experiment_kwargs, kwargs_not_to_be_sent)
    lin_kwargs_to_send = linearize_kwargs_to_send_model(kwargs_to_send)
    args_to_send += lin_kwargs_to_send
    
    os.makedirs(model_dir, exist_ok=True)
    load_pretrained_model: bool = experiment_kwargs["overwrite_output_dir"]
    args_to_send += get_model_params(model_type, model_dir, 
                                     load_pretrained=load_pretrained_model, 
                                     source_prefix=experiment_kwargs.get('source_prefix', None))
    
    if train_dataset == "qasrl2015":
        args_to_send.extend(qasrl_2015_params)
    elif train_dataset == "qasrl2018":
        args_to_send.extend(qasrl_2018_params)
    elif train_dataset == "qanom":
        args_to_send.extend(qanom_params)
    elif train_dataset == "qasrl":
        args_to_send.extend(qasrl_params)
    elif train_dataset == "joint_qasrl":
        args_to_send.extend(get_joint_params("qasrl", qanom_joint_factor))
    elif train_dataset == "joint_qanom":
        args_to_send.extend(get_joint_params("qanom", qanom_joint_factor))
    elif train_dataset == "qadiscourse":
        args_to_send.extend(qadiscourse_params)
    else:
        raise ValueError(f"qasrl_train_dataset doesn't exist ; qasrl_train_dataset {train_dataset}")

    sys.argv = args_to_send
    evals, run = main()
    wandb.config.update(experiment_kwargs, allow_val_change=True)
    test_dataset = train_dataset.split("_")[1] if "joint" in train_dataset else train_dataset
    wandb.config['test_dataset'] = test_dataset
    # save configuration of experiment
    with open(f"{model_dir}/experiment_kwargs.json", "w") as fout:
        json.dump(experiment_kwargs, fout)
    wandb.save(f"{model_dir}/*.json")
    wandb.save(f"{model_dir}/*.csv")
    wandb.save(f"{model_dir}/*.txt")
    wandb.save(f"{model_dir}/pytorch_model.bin")
    if finish_wandb:
        wandb.finish()
    return evals    


def load_and_predict(saved_model_path: str, 
                     test_file, 
                     output_dir=None, 
                     wandb_run_name=None, 
                    #  decode_qasrl: bool = True, 
                     text_column=None, **kwargs):
    
    # load kwargs from the trained model directory
    experiment_kwargs = json.load(open(os.path.join(saved_model_path, "experiment_kwargs.json")))

    # prepare the arguments for the run_summarization script:
    
    # if output_dir not provided, default to model dir/inference_outputs
    test_file_name_stem = Path(test_file).stem
    output_dir = output_dir or os.path.join(saved_model_path, f"inference_outputs/{test_file_name_stem}")
    os.makedirs(output_dir, exist_ok=True)
    wandb_run_name = wandb_run_name or f"{now()} Inference on {test_file}"
    
    args_to_send = [
        'run_summarization.py',
        '--model_name_or_path', saved_model_path,
        '--test_file', test_file,
        '--do_predict',
        '--predict_with_generate',
        '--eval_accumulation_steps', '10',  # Necessary to avoid OOM where all predictions are kept on one GPU    
        '--report_to', 'wandb',
        '--wandb_run_name', wandb_run_name, 
        '--output_dir', output_dir,
        # required:  TODO improve
        # '--pad_to_max_length', 'true',
    ]
        
    if text_column: # default value is "sentence"
        args_to_send += [
            "--text_column", text_column, 
        ]
            
    defaults = get_default_kwargs() # use defaults to account for possibly new kwargs not present in loaded model's experiment_kwargs.json config file
    experiment_kwargs = dict(defaults, **experiment_kwargs) # override defaults with loaded kwargs
    kwargs = dict(experiment_kwargs, **kwargs)   # integrate loaded-experiment kwargs values and override them by function's **kwargs

    if "batch_size" in kwargs and kwargs["batch_size"] is not None:
        kwargs["per_device_eval_batch_size"] = kwargs["batch_size"]
        kwargs.pop("batch_size")    
    
    kwargs_not_to_be_sent = ('description', 'train_dataset', 'test_dataset', 
                             'train_epochs', 'wandb_run_name', 'output_dir',
                             'qanom_joint_factor', 'dir_switch')
    kwargs = utils.dict_without_keys(kwargs, kwargs_not_to_be_sent)
    
    # add model name and model-specific args (model_type, source_perfix)
    model_dir=saved_model_path  
    model_type = kwargs.pop("model_type")
    source_prefix = kwargs.pop("source_prefix", None)
    args_to_send += get_model_params(model_type, model_dir, 
                                     load_pretrained=False,  # always load the local model from training
                                     source_prefix=source_prefix) 
    
    kwargs_to_send = linearize_kwargs_to_send_model(kwargs)
    args_to_send.extend(kwargs_to_send)
    
    # Run predict
    sys.argv = args_to_send
    results, run = main()
     
    wandb.finish()
  

def load_and_evaluate(saved_model_path: str,
                      test_dataset: Literal["qanom", "qasrl", "qadiscourse"] = "qanom",
                      do_eval_on_dev: bool = True, # or on test
                      output_dir=None, 
                      wandb_run_name=None,
                      **kwargs):
    #TODO TEST

    # load kwargs from the trained model directory, change test
    experiment_kwargs = json.load(open(os.path.join(saved_model_path, "experiment_kwargs.json")))
    # prepare the arguments for the run_summarization script:
    
    # if output_dir not provided, default to model dir/inference_outputs
    output_dir = saved_model_path
    # output_dir = output_dir or os.path.join(saved_model_path, f"{test_dataset}_evaluation_outputs")
    # os.makedirs(output_dir, exist_ok=True)
    wandb_run_name = wandb_run_name or f"{now()} Evaluate {saved_model_path} on {test_dataset}"
    
    args_to_send = [
        'run_summarization.py',
        '--model_name_or_path', saved_model_path,
        '--do_eval',
        '--predict_with_generate',
        '--eval_accumulation_steps', '10',  # Necessary to avoid OOM where all predictions are kept on one GPU    
        '--report_to', 'wandb',
        '--wandb_run_name', wandb_run_name, 
        '--output_dir', output_dir,
    ]

    # set evaluation dataset
    qasrl_test_dataset = "2020" if test_dataset == "qasrl" else test_dataset
    if qasrl_test_dataset == "2015":
        args_to_send.extend(qasrl_2015_params)
    elif qasrl_test_dataset == "2020":
        args_to_send.extend(qasrl_2020_params)
    elif qasrl_test_dataset == "qanom":
        args_to_send.extend(qanom_params)
    elif qasrl_test_dataset == "qadiscourse":
        args_to_send.extend(qadiscourse_params)

    # if "batch_size" in kwargs and kwargs["batch_size"] is not None:
    #     kwargs["per_device_train_batch_size"] = kwargs["batch_size"]
    #     kwargs["per_device_eval_batch_size"] = kwargs["batch_size"]
    #     kwargs.pop("batch_size")  

    defaults = get_default_kwargs() # use defaults to account for possibly new kwargs not present in loaded model's experiment_kwargs.json config file
    experiment_kwargs = dict(defaults, **experiment_kwargs) # override defaults with loaded kwargs
    kwargs = dict(experiment_kwargs, **kwargs)   # integrate loaded-experiment kwargs values and override them by function's **kwargs
    if "batch_size" in kwargs and kwargs["batch_size"] is not None:
        kwargs["per_device_eval_batch_size"] = kwargs["batch_size"]
        kwargs.pop("batch_size")   
    
    kwargs["do_eval_on"] = "validation" if do_eval_on_dev else "test"

    # remove args from `experiment_kwargs` and from defaults those kwargs that we want to override here or don't need fro evaluation
    kwargs_not_to_be_sent = ('description', 'train_dataset', 'test_dataset', 
                             'train_epochs', 'wandb_run_name', 'output_dir', 
                             'qanom_joint_factor', 'dir_switch')
    kwargs = utils.dict_without_keys(kwargs, kwargs_not_to_be_sent)
        
    model_dir=saved_model_path  
    model_type = kwargs.pop("model_type")
    source_prefix = kwargs.pop("source_prefix", None)
    # add model name and model-specific args (model_type, source_perfix)
    args_to_send += get_model_params(model_type, model_dir, 
                                     load_pretrained=False,  # always load the local model from training
                                     source_prefix=source_prefix) 

    
    kwargs_to_send = linearize_kwargs_to_send_model(kwargs)
    args_to_send.extend(kwargs_to_send)
    
    # Run main
    sys.argv = args_to_send
    eval_results, run = main()
    wandb.config['test_dataset'] = test_dataset
    wandb.config['model_type'] = model_type
    wandb.config.update(kwargs)
    wandb.save(f"{output_dir}/*.csv")
    wandb.save(f"{output_dir}/*.json")
    wandb.save(f"{output_dir}/*.txt")
    wandb.finish()
    return eval_results
    

from pipeline import load_trained_model
    
def upload_trained_model(saved_model_path, repo_name):
    model, tokenizer = load_trained_model(saved_model_path)
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    # upload also the "experiment_kwargs.json" file for preprocessing and postprocessing switches
    from huggingface_hub import upload_file
    upload_file(f"{saved_model_path}/experiment_kwargs.json", 
                path_in_repo="preprocessing_kwargs.json", 
                repo_id=f"kleinay/{repo_name}")
    print(f"Uploaded to https://huggingface.co/kleinay/{repo_name}")
    

if __name__ == "__main__":
    # model_type = "bart"
    # model_type = "t5"

    # model_dir = get_model_dir(model_type)

    # # qasrl_train_dataset = "2015"
    # # qasrl_train_dataset = "2018"
    # # qasrl_train_dataset = "qanom"
    # qasrl_train_dataset = "joint_qanom"

    # # qasrl_test_dataset = "2015"
    # # qasrl_test_dataset = "2020"
    # qasrl_test_dataset = "qanom"

    # train_epochs = 30
    
    # run = train(model_type, qasrl_train_dataset, train_epochs, model_dir,
    #             overwrite_output_dir=True, 
    #             wandb_run_name=f"{now()}_{model_dir}_{qasrl_train_dataset}",
    #             preprocess_input_func="input_predicate_marker")
    # predict(model_type, qasrl_test_dataset, model_dir, run)
    # decode_into_qasrl(model_dir, qasrl_test_dataset)
    # unlabelled_arg, labelled_arg, unlabelled_role = evaluate(model_dir, qasrl_test_dataset, wandb_run=run)
    model = "trained_models/t5_10ep-joint-qanom_15.12.21"
    load_and_evaluate(model, "qanom")
    