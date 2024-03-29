from typing import Optional, Union, Iterable, List, Any, Dict, Tuple
import itertools, math, random
import pandas as pd
import numpy as np
import wandb
import torch


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

def listSplit(lst, delimeterElement):
    " as str.split(); return a splitted list (list of sub-lists), splitted by the delimeter. "
    return [list(y) 
            for x, y in itertools.groupby(lst, lambda z: z == delimeterElement) 
            if not x]

def all_indices(lst: Iterable[Any], element: Any) -> List[int]:
    " Return a list of all indices where `element` occur in `lst`" 
    return [i for i,e in enumerate(lst) if e==element]

def split_by_indices(lst: List[Any], sep_indices: List[int]) -> List[List[Any]]:
    " Split `lst` to a list of sub-lists using indices of separators"
    sublists = []
    seps = [-1] + sep_indices + [len(lst)]
    for i in range(len(seps)-1):
        sublists.append(lst[seps[i]+1:seps[i+1]])
    return sublists 

def strip_sequence(seq: torch.Tensor, val_to_strip: int = 0) -> torch.Tensor:
    return seq[seq!=val_to_strip]

def arr_to_buckets(arr: Iterable[int], buckets: Dict[Tuple[int, int], str]):
    def to_bucket(e):
        for (start, end), label in buckets.items():
            if e in list(range(start, end)):
                return label
        return None
    return [to_bucket(e) for e in arr]

def sample_permutations(lst: List[Any], k: int, with_replacement = False) -> List[List[Any]]:
    " Return a sample of `k` random permutations of `lst` "
    n_all_permutations = math.factorial(len(lst))
    if k / n_all_permutations < 0.1 or n_all_permutations > 100000:
        return _sample_permutations_without_enumeration(lst, k, with_replacement) 
    else:
        return _sample_permutations_with_enumeration(lst, k, with_replacement) 
         

def _sample_permutations_without_enumeration(lst: List[Any], k: int, with_replacement = False) -> List[List[Any]]:
    """ 
    Return a sample of `k` random permutations of `lst`, without enumerating all possible permutations. 
    Warning: assuming collision probability is tiny.
    """
    selected_permutations = []
    while len(selected_permutations) < k:
        candidate_permutation = np.random.choice(lst, size=len(lst), replace=False).tolist()
        if with_replacement or candidate_permutation not in selected_permutations:
            selected_permutations.append(candidate_permutation)
    return selected_permutations

def _sample_permutations_with_enumeration(lst: List[Any], k: int, with_replacement = False) -> List[List[Any]]:
    """ 
    Return a sample of `k` random permutations of `lst`, by sampling from enumertaed list of all possible permutations.
    Warning: Can be VERY memory demanding (and may crash) for a large `lst`.
    """
    permutations = list(itertools.permutations(lst))
    if not with_replacement:
        k = min(k, len(permutations))
        return random.sample(permutations, k=k)
    else:
        return random.choices(permutations, k=k)