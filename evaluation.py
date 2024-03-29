from argparse import ArgumentParser
from typing import Optional
import pandas as pd, numpy as np

import wandb

from qasrl_gs.scripts.evaluate_dataset import eval_datasets as qasrl_evaluate
from utils import setup_wandb, reshape_qasrl_into_qanom
from qanom.evaluation.evaluate import eval_datasets as qanom_evaluate, Metrics, get_recall_and_precision_mistakes
from qanom.annotations.decode_encode_answers import decode_qasrl
from qanom.annotations.common import set_key_column
from qanom.utils import rename_column

from dfa_fill_qasrl_slots import extract_is_negated

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

    
def run_qasrl_gs_evaluation(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame):
    # Adjust to qasrl format for qasrl-gs evaluation package requirements 
    rename_column(ground_truth_df, 'sent_id', 'qasrl_id')
    rename_column(ground_truth_df, 'answers', 'answer') 
    rename_column(ground_truth_df, 'answer_ranges', 'answer_range') 
    rename_column(ground_truth_df, 'predicate_idx', 'verb_idx')
    # set question slots at ground_truth_df
    for i, slot in enumerate(['wh', 'aux', 'subj', 'verb', 'obj', 'prep', 'obj2']):   # replace underscore with empty string
        ground_truth_df.loc[:, slot] = ground_truth_df['question'].apply(lambda q: q[i] if q else '')
    # clean questoin slots from '_'
    for slot in ['wh', 'aux', 'subj', 'obj', 'prep', 'obj2']:   # replace underscore with empty string
        predictions_df.loc[:, slot] = predictions_df[slot].apply(lambda s: '' if s=='_' else s)
        ground_truth_df.loc[:, slot] = ground_truth_df[slot].apply(lambda s: '' if s=='_' else s)
    # add additional binary slots
    ground_truth_df.loc[:, 'is_negated'] = ground_truth_df.apply(extract_is_negated, axis=1)    
    # TODO make 'is_passive' to work correctly - currently ignoring 
    ground_truth_df.loc[:, 'is_passive'] = False   
    predictions_df.loc[:, 'is_passive'] = False   
    # linearize question column in `ground_truth_df`
    def q_slot_to_q_str(q_slots) -> str:
        q_slots = [sl for sl in q_slots[:-1] if sl != '_']
        return ' '.join(q_slots) + "?" if q_slots else ''
    ground_truth_df.loc[:, 'question'] = ground_truth_df['question'].apply(q_slot_to_q_str) 
    
    # decode answers on `predictions_df` - to be lists
    predictions_df = decode_qasrl(predictions_df)
    # replace lists in tuples in answers of ground_truth_df (taken from Dataset)
    ground_truth_df.loc[:, 'answer_range'] = ground_truth_df['answer_range'].apply(lambda l: [tuple(ans) for ans in l])     

    arg, larg, role = qasrl_evaluate(grt_df=ground_truth_df, sys_df=predictions_df)
    
    # copied from qanom - not sure it works properly
    recall_mistakes_df, precision_mistakes_df = get_recall_and_precision_mistakes(predictions_df, ground_truth_df)

    return arg, larg, role, (recall_mistakes_df, precision_mistakes_df)

def run_qanom_evaluation(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame):   
    if len(predictions_df)==0:
        return Metrics(0,0,1), Metrics(0,0,1), Metrics(0,0,1)
    
    # Adjust to qanom format for qanom evaluation package requirements 
    rename_column(ground_truth_df, 'sent_id', 'qasrl_id') 
    rename_column(ground_truth_df, 'answers', 'answer') 
    rename_column(ground_truth_df, 'answer_ranges', 'answer_range') 
    rename_column(ground_truth_df, 'predicate_idx', 'target_idx') 
    set_key_column(ground_truth_df)
    rename_column(predictions_df, 'verb', 'predicate') 
    rename_column(predictions_df, 'verb_idx', 'target_idx')
    
    # rename_column(predictions_df, 'verb', 'noun')
    
    # Temporal? for working on analyses
    ground_truth_df.loc[:,'qa_position'] = ground_truth_df.groupby(['qasrl_id', 'target_idx']).cumcount()
    ground_truth_df["qa_position_group"] = ground_truth_df.qa_position.apply(lambda n: str(n+1) if n<3 else ">3") 
    ground_truth_df.loc[:,'num_qas'] =  ground_truth_df.groupby(['qasrl_id', 'target_idx']).qa_position.transform('max') + 1
    ground_truth_df["num_qas_group"] = ground_truth_df.num_qas.apply(lambda n: str(n) if n<=3 else ">3") 
    # ans_index = ground_truth_df.answer_range.str.split("~!~").str.get(0).str.split(":").str.get(0).astype(int)
    # dep_range = np.absolute(ans_index - ground_truth_df.verb_idx)
    # dependency_range_buckets = {(0,3): "1-2", (3,6):"3-5", (6,9):"6-8", (9,300):"9-Inf"}
    # from utils import arr_to_buckets
    ground_truth_df["dep_range_group"] = ''
    ground_truth_df["sentence_length_group"] = ''
    ground_truth_df["raw_question"] = ''
    
    ground_truth_df.loc[:,'verb_prefix'] = ''
    ground_truth_df.loc[:,'verb_slot_inflection'] = ''
    predictions_df.loc[:,'is_verbal'] = True
    predictions_df.loc[:,'verb_prefix'] = ''
    predictions_df.loc[:,'verb_slot_inflection'] = ''
    
    # set question slots at ground_truth_df
    for i, slot in enumerate(['wh', 'aux', 'subj', 'verb', 'obj', 'prep', 'obj2']):   # replace underscore with empty string
        ground_truth_df.loc[:, slot] = ground_truth_df['question'].apply(lambda q: q[i] if q else '')
    # clean questoin slots from '_'
    for slot in ['wh', 'aux', 'subj', 'obj', 'prep', 'obj2']:   # replace underscore with empty string
        predictions_df.loc[:, slot] = predictions_df[slot].apply(lambda s: '' if s=='_' else s)
        ground_truth_df.loc[:, slot] = ground_truth_df[slot].apply(lambda s: '' if s=='_' else s)
    
    ground_truth_df.loc[:, 'is_negated'] = ground_truth_df.apply(extract_is_negated, axis=1)   
    ground_truth_df.loc[:, 'is_passive'] = False   # TODO make it work correctly
    # linearize question column in `ground_truth_df`
    def q_slot_to_q_str(q_slots) -> str:
        q_slots = [sl for sl in q_slots[:-1] if sl != '_']
        return ' '.join(q_slots) + "?" if q_slots else ''
    ground_truth_df.loc[:, 'question'] = ground_truth_df['question'].apply(q_slot_to_q_str)   
    
    # keep in ground_truth_df only positive predicates (non-predicates shouldn't count as FN)
    ground_truth_df = ground_truth_df[ground_truth_df['is_verbal']]

    # decode answers on `predictions_df` - to be lists
    predictions_df = decode_qasrl(predictions_df)
    # replace lists in tuples in answers of ground_truth_df (taken from Dataset)
    ground_truth_df.loc[:, 'answer_range'] = ground_truth_df['answer_range'].apply(lambda l: [tuple(ans) for ans in l])     
    
    # Add empty predictions for all instances from gold test not in predicted output, so that evaluation will count them as FN
    all_keys = set(ground_truth_df.key)
    missing_keys = all_keys - set(predictions_df.key)
    df_to_add = ground_truth_df[ground_truth_df.key.isin(missing_keys)].drop_duplicates('key')[predictions_df.columns].copy()
    # set prediction columns empty
    for c in ['wh', 'subj', 'obj', 'obj2', 'aux', 'prep', 'verb_prefix', 'verb_slot_inflection']:
        df_to_add.loc[:,c] = ''
    for c in ['is_passive','is_negated']:
        df_to_add.loc[:,c] = False
    for c in ['answer','answer_range']:    
        df_to_add[c] = np.empty((len(df_to_add), 0)).tolist()    # set as empty lists
    df_to_add.loc[:, 'question'] = np.nan    
    # Concat df_to_add to predictions
    predictions_df = pd.concat((predictions_df, df_to_add))
    
    # Compute evaluation measures
    arg, larg, role, is_verbal, _ = qanom_evaluate(predictions_df, ground_truth_df)

    recall_mistakes_df, precision_mistakes_df = get_recall_and_precision_mistakes(predictions_df, ground_truth_df)
    return arg, larg, role, (recall_mistakes_df, precision_mistakes_df)

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


