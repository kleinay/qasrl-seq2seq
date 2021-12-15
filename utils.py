from typing import Optional
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