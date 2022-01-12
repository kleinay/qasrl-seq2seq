from typing import Optional, Union
import wandb
import pandas as pd


def setup_wandb(use_wandb: bool, wandb_run_name: str):
    return wandb.init(name=wandb_run_name, project="qasrl", reinit=True, mode="online" if use_wandb else "disabled")

def reshape_qasrl_into_qanom(df: pd.DataFrame, sentences_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    from qanom.annotations.common import set_sentence_columns, set_key_column
    from qanom.utils import rename_column
    
    if sentences_df is not None:      
        set_sentence_columns(df, sentences_df)
    set_key_column(df)
    rename_column(df, 'verb_idx', 'target_idx')
    rename_column(df, 'verb', 'noun')
    df['is_verbal'] = True
    df['verb_form'] = ' '
    return df    

def replaceKeys(orig_dict, oldKeys2NewKeys, inplace=True):
    """ replace keys with new keys using oldKeys2NewKeys mapping. """
    target_dict = orig_dict if inplace else orig_dict.copy()
    for oldKey, newKey in oldKeys2NewKeys.items():
        if oldKey in target_dict:
            target_dict[newKey] = target_dict.get(oldKey)
            if oldKey != newKey: target_dict.pop(oldKey)
    return target_dict

def str2num(s: str) -> Union[int, float]:
    try:
        num = int(s)
    except ValueError:
        num = float(s)
    return num
    