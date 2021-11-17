# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from typing import Literal
from IPython import get_ipython

# Imports

from run_summarization import main
from run_evaluation import evaluate_qasrl, evaluate_qanom
import os
import sys
import json
import datetime

# General variables
now = lambda: datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
system = get_ipython().system if get_ipython() else os.system

run = None  # wandb run
# tmp_dir = os.environ.get("TMPDIR", "/tmp")
tmp_dir = os.environ.get("HOME") + "/tmp"

# Params

### Data params

qasrl_2015_params = ['--dataset_name', 'qa_srl']
qasrl_2018_params = ['--dataset_name', 'biu-nlp/qa_srl2018']
qasrl_2020_params = [
    "--train_file", "qasrl_gs/data/gold/wikinews.dev.combined.csv",
    "--validation_file", "qasrl_gs/data/gold/all.dev.combined.csv",
    "--test_file", "qasrl_gs/data/gold/all.test.combined.csv",
    "--text_column", "sentence", 
    "--summary_column", "answer"
]
qanom_params = ['--dataset_name', 'biu-nlp/qanom']  

### Model params

t5_model_dir = f'{tmp_dir}/t5-tst-summarization'
os.environ["T5_MODEL_DIR"] = t5_model_dir
t5_small_model_train_params = [
    '--model_name_or_path', 't5-small'
]
t5_model_predict_params = [
    '--model_name_or_path', t5_model_dir
]
t5_extra_params = [
    '--model_type', 't5',
    '--source_prefix', 'summarize: ',
    '--output_dir', t5_model_dir
]

bart_model_dir = f'{tmp_dir}/bart-tst-summarization'
os.environ["BART_MODEL_DIR"] = bart_model_dir
bart_base_model_train_params = [
    '--model_name_or_path', 'facebook/bart-base'
]
bart_model_predict_params = [
    '--model_name_or_path', bart_model_dir
]
bart_extra_params = [
    '--model_type', 'bart',
    '--output_dir', bart_model_dir
]


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


# %%
# ### (1) Train ********************************
def train(model_type, 
          qasrl_train_dataset, 
          train_epochs,
          run_name=None, # a way to name experiment in wandb 
          limit_train_data=None,
          limit_eval_data=None,
          **kwargs):
    default_boolean_args = dict(do_eval=True,
                                overwrite_cache=True,
                                overwrite_output_dir=True,
                                predict_with_generate=True,
                                debug_mode=False,)
    sys.argv = [
        'run_summarization.py',
        '--do_train',
        '--per_device_train_batch_size', '12',
        '--per_device_eval_batch_size', '12',
        '--logging_steps', '200',
        '--num_train_epochs', f'{train_epochs}',
        '--report_to', 'wandb',
        '--wandb_run_name', run_name,
        # '--debug_mode',
        # '--n_gpu', '[5,6]'
    ]
    
    kwargs = dict(default_boolean_args, **kwargs)   # integrate deafult (bool) values and override them by **kwargs
    
    kwargs_to_send = []
    for key, value in kwargs.items():
        if type(value) == bool:
            if value is True:
                kwargs_to_send.extend([f'--{key}'])
        else:
            kwargs_to_send.extend([f'--{key}', f'{value}'])
    sys.argv += kwargs_to_send
    
    if limit_train_data:
        sys.argv += ['--limit_train_data', f'{limit_train_data}']
    if limit_eval_data:
        sys.argv += ['--limit_eval_data', f'{limit_eval_data}']

    if model_type == "t5":
        sys.argv.extend(t5_small_model_train_params)
        sys.argv.extend(t5_extra_params)
    elif model_type == "bart":
        sys.argv.extend(bart_base_model_train_params)
        sys.argv.extend(bart_extra_params)
    else:
        raise ValueError(f"model_type doesn't exist ; model_type {model_type}")

    if qasrl_train_dataset == "2015":
        sys.argv.extend(qasrl_2015_params)
    elif qasrl_train_dataset == "2018":
        sys.argv.extend(qasrl_2018_params)
    elif qasrl_train_dataset == "qanom":
        sys.argv.extend(qanom_params)
    else:
        raise ValueError(f"qasrl_train_dataset doesn't exist ; qasrl_train_dataset {qasrl_train_dataset}")

    _, run = main()
    return run


# run = train(model_type, qasrl_train_dataset, train_epochs)
# %%
# ### (2) Predict ***************************

# !python run_summarization.py --model_name_or_path $TMPDIR/tst-summarization --do_predict --dataset_name qa_srl --output_dir $TMPDIR/tst-summarization --source_prefix "summarize: " --predict_with_generate
def predict(model_type, qasrl_test_dataset, run=None):
    sys.argv = [
        'run_summarization.py',
        '--do_predict',
        '--predict_with_generate',
        '--eval_accumulation_steps', '10',  # Necessary to avoid OOM where all predictions are kept on one GPU    
        '--report_to', 'wandb',
        '--wandb_run_name', run.name if run else None
    ]

    if model_type == "t5":
        sys.argv.extend(t5_extra_params)
        sys.argv.extend(t5_model_predict_params)
    elif model_type == "bart":
        sys.argv.extend(bart_extra_params)
        sys.argv.extend(bart_model_predict_params)
    else:
        raise ValueError(f"model_type doesn't exist ; model_type {model_type}")    

    if qasrl_test_dataset == "2015":
        sys.argv.extend(qasrl_2015_params)
    elif qasrl_test_dataset == "2020":
        sys.argv.extend(qasrl_2020_params)
        test_sentences_path = "/sentences_data/wikinews.test.full.csv"
    elif qasrl_test_dataset == "qanom":
        sys.argv.extend(qanom_params)
        test_sentences_path = "/data/generated_predictions.csv" # generated predictions will also hold a "tokens" columns and could serve as sentences-file   
    else:
        raise ValueError(f"qasrl_test_dataset doesn't exist ; qasrl_test_dataset {qasrl_test_dataset}")

    main(generate_sentence_column_in_prediction= qasrl_test_dataset == "qanom" )


# predict(model_type, qasrl_test_dataset, run)

# %% 
# ### (3) Run state machine using docker, for parsing the predicted questions into 7 slot format

def decode_into_qasrl(model_dir, test_dataset):
    os.environ["MODEL_DIR"] = model_dir 
    if test_dataset != "qanom":
        system('docker run -it -v "${MODEL_DIR}:/data" -v "$(pwd)/../qasrl_bart/qasrl_gs/data/sentences/:/sentences_data" --rm --name qasrl-automaton hirscheran/qasrl_state_machine_example "file" "/data/generated_predictions.csv" "$test_sentences_path" "/data/output_file.csv" > /dev/null 2>&1')
    else:
        system('docker run -it -v "${MODEL_DIR}:/data" --rm --name qasrl-automaton hirscheran/qasrl_state_machine_example "file" "/data/generated_predictions.csv" "/data/output_file.csv" > /dev/null 2>&1')


# decode_into_qasrl(model_dir, qasrl_test_dataset)

# %% 
# ### (4) Evaluate
def evaluate(model_dir, qasrl_test_dataset):
    from run_evaluation import evaluate_qasrl, evaluate_qanom
    if qasrl_test_dataset != "qanom":
        return evaluate_qasrl(model_dir, "qasrl_gs/data/gold/wikinews.test.gold.csv", f"{model_dir}/output_file.csv", None, None)
    else:
        return evaluate_qanom(model_dir, None)


# evaluate(model_dir, qasrl_test_dataset)


def print_invalid_dist(model_type):
    import pandas as pd
    df=pd.read_csv(f"/home/nlp/kleinay/tmp/{model_type}-tst-summarization/invalid_generated_predictions.csv") 
    print(f"Overall: {len(df)}\n")
    print(df["Error-type"].value_counts(), "\n")
    print(df["Error-type"].value_counts() / len(df))

def full_experiment(model_type: Literal["bart", "t5"] = "bart", 
                    train_dataset: Literal["qanom", "qasrl"] = "qanom", 
                    test_dataset: Literal["qanom", "qasrl"] = "qanom",
                    train_epochs: int = 10,
                    **kwargs):
    
    model_dir = t5_model_dir if model_type == "t5" else bart_model_dir
    qasrl_train_dataset = "qanom" if train_dataset == "qanom" else "2018"
    qasrl_test_dataset = "qanom" if test_dataset == "qanom" else "2020"

    run = train(model_type, qasrl_train_dataset, train_epochs, **kwargs)
    predict(model_type, qasrl_test_dataset, run)
    decode_into_qasrl(model_dir, qasrl_test_dataset)
    unlabelled_arg, labelled_arg, unlabelled_role = evaluate(model_dir, qasrl_test_dataset)
    return unlabelled_arg, labelled_arg, unlabelled_role


if __name__ == "__main__":
    # model_type = "bart"
    model_type = "t5"

    model_dir = t5_model_dir if model_type == "t5" else bart_model_dir

    # qasrl_train_dataset = "2015"
    # qasrl_train_dataset = "2018"
    qasrl_train_dataset = "qanom"

    # qasrl_test_dataset = "2015"
    # qasrl_test_dataset = "2020"
    qasrl_test_dataset = "qanom"

    train_epochs = 3
    
    run = train(model_type, qasrl_train_dataset, train_epochs)
    predict(model_type, qasrl_test_dataset, run)
    decode_into_qasrl(model_dir, qasrl_test_dataset)
    unlabelled_arg, labelled_arg, unlabelled_role = evaluate(model_dir, qasrl_test_dataset)