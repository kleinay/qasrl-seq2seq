from typing import Optional, Union, Iterable, List, Any
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

def dict_without_keys(d: dict, kwargs_to_remove) -> dict:
    ret = d.copy()
    for kw in kwargs_to_remove:
        ret.pop(kw, None)
    return ret    

# available only as of python 3.9
def without_prefix(s: str, prefix: str) -> str:
    if not prefix: return s
    return s[len(prefix):] if s.startswith(prefix) else s
def without_suffix(s: str, suffix: str) -> str:
    if not suffix: return s
    return s[:-len(suffix)] if s.endswith(suffix) else s
 
def duplicate_by_per_element_factor(lst: Iterable[Any], factors: Iterable[int]):
    """ Returns a list where each element `lst[i]` is repeated `factors[i]` times. 
    Note:
    len(factors) must be == len(lst).
    Example:
    duplicate_by_per_element_factor(["a", "b", "c"], [2,3,1]) -> ['a', 'a', 'b', 'b', 'b', 'c'] 
    """
    ret = []
    for element, factor in zip(lst, factors):
        ret.extend([element] * factor)
    return ret

def df_to_row_list(df: pd.DataFrame) -> List[pd.Series]:
    "Return the list of rows of the `df` "
    return list(list(zip(*df.iterrows()))[1])

def stack_rows(rows: Iterable[pd.Series], ignore_index=True) -> pd.DataFrame:
    return pd.concat(rows, ignore_index=ignore_index, axis=1).T
