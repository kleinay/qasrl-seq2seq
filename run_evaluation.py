from argparse import ArgumentParser
from typing import Optional
import pandas as pd, numpy as np

import wandb

from qasrl_gs.scripts.evaluate_dataset import main as qasrl_eval
from utils import setup_wandb, reshape_qasrl_into_qanom
from qanom.evaluation.evaluate import Metrics, BinaryClassificationMetrics

def evaluate_qadiscourse(model_dir, wandb_run_name: Optional[str]):
    if not wandb_run_name:
        setup_wandb(wandb_run_name is not None, wandb_run_name)   
    
    # TODO implement qadiscourse evaluations
    
    input_prediction_file_name = f"{model_dir}/generated_qadiscourse_predictions.csv" 
    predicted_df = pd.read_csv(input_prediction_file_name)
     
    if len(predicted_df)==0:
        return Metrics(0,0,1), Metrics(0,0,1)
    eval_fn = f"{model_dir}/evaluation_qadiscourse.txt"
    raise NotImplementedError()

    
    
def evaluate_qasrl(model_dir, input_gold_file, input_prediction_file, input_sentences_path: Optional[str], wandb_run_name: Optional[str]):
    if not wandb_run_name:
        setup_wandb(wandb_run_name is not None, wandb_run_name)
    arg, larg, role = qasrl_eval(proposed_path=input_prediction_file, reference_path=input_gold_file, sents_path=input_sentences_path)
    eval_fn = f"{model_dir}/evaluation_qasrl.txt"
    write_qasrl_evaluation_to_file(eval_fn, arg, larg, role)
    wandb.save(eval_fn)
    return arg, larg, role

def evaluate_qanom(model_dir: str, wandb_run_name: Optional[str]):
    if not wandb_run_name:
        setup_wandb(wandb_run_name is not None, wandb_run_name)
    from qanom.annotations.common import read_annot_csv
    from qanom.evaluation.evaluate import eval_datasets as qanom_evaluate, get_recall_and_precision_mistakes
    
    if len(pd.read_csv(f"{model_dir}/generated_predictions.csv"))==0 or len(pd.read_csv(f"{model_dir}/decoded_output.csv"))==0:
        return Metrics(0,0,1), Metrics(0,0,1), Metrics(0,0,1)
    
    qanom_test_df = read_annot_csv("QANom/dataset/annot.test.csv")
    df_parsed_outputs = read_annot_csv(f"{model_dir}/generated_predictions.csv")
    # adjust `df_parsed_outputs` to qanom format for qanom evaluation package requirements 
    from qanom.utils import rename_column
    rename_column(df_parsed_outputs, 'verb_idx', 'target_idx')
    rename_column(df_parsed_outputs, 'verb', 'noun')
    df_parsed_outputs['is_verbal'] = True
    df_parsed_outputs['verb_prefix'] = ''
    df_parsed_outputs['verb_slot_inflection'] = ''
    for slot in ['wh', 'subj', 'obj', 'obj2', 'aux', 'prep']:   # replace underscore with empty string
        df_parsed_outputs[slot] = df_parsed_outputs[slot].apply(lambda s: '' if s=='_' else s)
    
    # Add empty predictions for all instances from gold test not in predicted output, so that evaluation will count them as FN
    all_keys = set(qanom_test_df.key)
    missing_keys = all_keys - set(df_parsed_outputs.key)
    df_to_add = qanom_test_df[qanom_test_df.key.isin(missing_keys)].drop_duplicates('key')[df_parsed_outputs.columns]
    # set prediction columns empty
    for c in ['wh', 'subj', 'obj', 'obj2', 'aux', 'prep', 'verb_prefix', 'verb_slot_inflection']:
        df_to_add[c] = ''
    for c in ['is_passive','is_negated']:
        df_to_add[c] = False
    for c in ['answer','answer_range']:    
        df_to_add[c] = np.empty((len(df_to_add), 0)).tolist()    # set as empty lists
    df_to_add['question'] = np.nan    
    # Concat df_to_add to predictions
    df_parsed_outputs = pd.concat((df_parsed_outputs, df_to_add))
    
    # Print evaluation measures
    arg, larg, role, is_verbal, _ = qanom_evaluate(df_parsed_outputs, qanom_test_df)
    eval_fn = f"{model_dir}/evaluation_qanom.txt"
    write_qasrl_evaluation_to_file(eval_fn, arg, larg, role, is_verbal)
    wandb.save(eval_fn)
    # recall_mistakes_df, precision_mistakes_df = get_recall_and_precision_mistakes(df_parsed_outputs, qanom_test_df)
    return arg, larg, role

def write_qasrl_evaluation_to_file(fn, arg, larg, role, pred_detection = None):
    with open(fn, "w") as fout:
        fout.write("\n\t\t\tPrec\tRecall\tF1\n") 
        fout.write(f"arg-f1 \t\t\t{100*arg.prec():.1f}\t{100*arg.recall():.1f}\t{100*arg.f1():.2f}\n") 
        fout.write(f"labeled-arg-f1 \t\t{100*larg.prec():.1f}\t{100*larg.recall():.1f}\t{100*larg.f1():.2f}\n") 
        fout.write(f"role-f1 \t\t{100*role.prec():.1f}\t{100*role.recall():.1f}\t{100*role.f1():.2f}\n")
        if pred_detection:
            fout.write(f"predicate-detection (acc.)\t{100*pred_detection.accuracy():.2f}\n")
         
        
def print_evaluations(arg, larg, role):
    print("\n\t\t\tPrec\tRecall\tF1\n") 
    print(f"arg-f1 \t\t\t{100*arg.prec():.1f}\t{100*arg.recall():.1f}\t{100*arg.f1():.2f}\n") 
    print(f"labeled-arg-f1 \t\t{100*larg.prec():.1f}\t{100*larg.recall():.1f}\t{100*larg.f1():.2f}\n") 
    print(f"role-f1 \t\t{100*role.prec():.1f}\t{100*role.recall():.1f}\t{100*role.f1():.2f}\n") 

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("sys_path")
    ap.add_argument("ground_truth_path")
    ap.add_argument("-s","--sentences_path", required=False)
    ap.add_argument("--wandb_run_name", required=False)
    args = ap.parse_args()
    evaluate_qasrl(args.sys_path, args.ground_truth_path, args.sentences_path, args.wandb_run_name)
