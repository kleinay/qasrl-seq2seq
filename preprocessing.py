from builtins import ValueError
from dataclasses import dataclass
from typing import List, Literal, Any
from argparse import Namespace
import pandas as pd
import random
import itertools
import math

import utils

class Preprocessor:
    def __init__(self,
                 data_args,
                 special_tokens
                 ):
        if isinstance(data_args, dict):
            data_args = Namespace(**data_args)
        self.data_args = data_args
        self.special_tokens = special_tokens
        self.sample_k_permutations_per_instance = 3 
        
        self.preprocess_input_function_map = {
            "input_predicate_repeated": self.extract_inputs_predicate_repeated,
            "input_predicate_marker": self.extract_inputs_predicate_inline_marker,
            "qadiscourse_input": self.extract_qadiscourse_inputs,
        }
        self.preprocess_output_function_map = {
            "first_two_question_answer": self.extract_targets_only_first_two_question_answers,
            "permutate_sample_num_of_qas": self.extract_qa_permutations_as_qas,
            "permutate_sample_fixed": self.extract_qa_permutations_fixed,
            "permutate_all": self.extract_all_permutations,
            "all_shuffled": self.extract_targets_shuffled_order,
            "all_random_order": self.extract_targets_arbitrary_order,
            "all_by_answer_ordering": self.extract_targets_by_answer_ordering,
            "qadiscourse_output": self.extract_qadiscourse_targets,
            }
        # List of preprocessing functions that "duplicate", i.e. return multiple sequences for some instances.
        # We use this to distinguish preprocessing of the training set (duplicate) and dev/test (don't duplicate)
        self.duplicating_preprocessing_output_functions = ("permutate_sample_num_of_qas",
                                                           "permutate_sample_fixed",
                                                           "permutate_all")
        # where there a a lot of QAs, enumerating their permutations is intractible
        self.MAX_PERMUTATIONS = 10
     
    """
    External API:
    """
    def preprocess_input(self, x: pd.DataFrame) -> str:
        if self.data_args.preprocess_input_func not in self.preprocess_input_function_map:
            raise ValueError(f"input preprocessing function '{self.data_args.preprocess_input_func}' not supported; "
                             f"options are: {list(self.preprocess_input_function_map.keys())}")
        preprocessing_function = self.preprocess_input_function_map[self.data_args.preprocess_input_func]
        input_seq = preprocessing_function(x)
        # prepend prefix (used in T5 models to specify the task, e.g. "summarize: ")
        prefix = self.get_sequence_prefix(x)
        return prefix + input_seq
    
    def preprocess_output(self, x: pd.DataFrame, is_training: bool) -> str:
        row = x.iloc[0]
        if self.data_args.preprocess_output_func not in self.preprocess_output_function_map:
            raise ValueError(f"output preprocessing function {self.data_args.preprocess_output_func} not supported; "
                             f"options are: {list(self.preprocess_output_function_map.keys())}")
        
        preprocessing_function = self.preprocess_output_function_map[self.data_args.preprocess_output_func]
        if not is_training:
            if self.data_args.preprocess_output_func in self.duplicating_preprocessing_output_functions:
                preprocessing_function = self.preprocess_output_function_map["all_random_order"]
            
        target_seqs = preprocessing_function(x)
        
        def further_target_seq_processing(target_seq: str) -> str:
            if self.data_args.learn_predicate_type == "pre" and "predicate_type" in row:
                target_seq = row["predicate_type"] + " | " + target_seq 
            elif self.data_args.learn_predicate_type == "post" and "predicate_type" in row:
                target_seq = target_seq + " | " + row["predicate_type"] 
            return target_seq
        
        return [further_target_seq_processing(seq) for seq in target_seqs]
    
    def reverse_input_preprocessing(self, processed_sentence: str) -> str:
        """
        Return the original sentence from the preprocessed input sequence.
        Args:
            processed_sentence (str): the preprocessed input sequence, given
                by `tokenizer.decode(token_ids, skip_special_tokens=True)`.
        """
        orig_sentence = processed_sentence
        if self.data_args.preprocess_input_func == "input_predicate_repeated":
            raise NotImplemented
        elif self.data_args.preprocess_input_func == "qadiscourse_input":
            raise NotImplemented
        elif self.data_args.preprocess_input_func == "input_predicate_marker":
            #TODO
            # strip prefix (identified using colons ':')
            if self.data_args.source_prefix is not None:
                orig_sentence = ':'.join(orig_sentence.split(':')[1:])
            # strip verb_form (last word)
            orig_sentence = ' '.join(orig_sentence.split(' ')[:-1])
        return orig_sentence
    
    """
    Prefix for Sequence preprocessing:
    """
    def get_sequence_prefix(self, x: pd.DataFrame) -> str:
        row = x.iloc[0]
        if self.data_args.source_prefix is None:
            return ''
        if "<predicate-type>" in self.data_args.source_prefix:
            if "predicate_type" not in row or row["predicate_type"] is None:
                raise ValueError("source_prefix includes '<predicate-type>' but input row has no 'predicate_type' attribute.")
            pred_type = row["predicate_type"] 
            if self.data_args.source_prefix == "<predicate-type>": # backwrad compatibility - "<predicate-type>" alone was a sign for a longer prefix 
                return f"Generate QAs for {pred_type} QASRL: "
            else:
                return self.data_args.source_prefix.replace("<predicate-type>", pred_type)
        else:
            return self.data_args.source_prefix

    """
    Input Sequence preprocessing:
    """
    
    def extract_inputs_predicate_repeated(self, x: pd.DataFrame) -> str:
        """ Encode predicate by repeating it at the end of sequence """
        # all rows have the same index values (and predicate-tailored info) because of the groupby
        row = x.iloc[0]
        sentence_before_predicate, predicate, sentence_after_predicate = self._get_splitted_sentence_by_predicate(row)
        seq = f"{sentence_before_predicate} {predicate} {sentence_after_predicate} {self.special_tokens.separator_input_question_predicate} {predicate}"
        # embed also the verb_form
        seq = self._append_verb_form(seq, row)
        
        # append predicate_type
        if "predicate_type" in row:   # only if we train on joint-qasrl/joint-qanom datatset
            seq = f'{row["predicate_type"]} | {seq}' 
        return seq
    
    def extract_inputs_predicate_inline_marker(self, x: pd.DataFrame) -> str:
        """ Encode predicate by prefixing it with a marker """
        # all rows have the same index values (and predicate-tailored info) because of the groupby
        row = x.iloc[0]
        sentence_before_predicate, predicate, sentence_after_predicate = self._get_splitted_sentence_by_predicate(row)
        
        # prepare predicate marker
        #  In case we want a generic marker for all predicate types: """
        if self.data_args.predicate_marker_type == "generic":
            predicate_marker = self.special_tokens.predicate_generic_marker    
        #  In case we want special marker for each predicate type: """
        elif self.data_args.predicate_marker_type == "pred_type" \
            and "predicate_type" in row:
            predicate_marker = {"verbal": self.special_tokens.predicate_verb_marker , 
                                "nominal": self.special_tokens.predicate_nominalization_marker 
                                }[row["predicate_type"]]
        else:
            raise ValueError(f"invalid value for `data_args.predicate_marker_type`: {self.data_args.predicate_marker_type}")

        if self.data_args.use_bilateral_predicate_marker:
            seq = f"{sentence_before_predicate} {predicate_marker} {predicate} {predicate_marker} {sentence_after_predicate}"
        else:
            seq = f"{sentence_before_predicate} {predicate_marker} {predicate} {sentence_after_predicate}"
        
        # embed also the verb_form
        # In this function, since we don't repeat the predicate, separator_input_question_predicate prefixes the verb_form
        seq = self._append_verb_form(seq, row)
        
        # append predicate_type (if not captured by in predicate_marker)
        # if "predicate_type" in row and predicate_marker == self.special_tokens.predicate_generic_marker :
        #     seq = f'{row["predicate_type"]} | {seq}' 
        return seq
    
    def _get_splitted_sentence_by_predicate(self, row: pd.Series):
        sent_tokens = row.sentence.split(" ") 
        sentence_before_predicate = " ".join([sent_tokens[i] for i in range(int(row.predicate_idx))])
        predicate = sent_tokens[int(row.predicate_idx)]
        sentence_after_predicate = " ".join([sent_tokens[i] for i in range(int(row.predicate_idx)+1, len(sent_tokens))])
        return sentence_before_predicate, predicate, sentence_after_predicate
        
    def _append_verb_form(self, seq: str, df_row: pd.Series):
        if not self.data_args.append_verb_form or \
                "verb_form" not in df_row or \
                df_row.verb_form is None:
            return f"{seq} "
        else:
            return f"{seq} {self.special_tokens.separator_input_question_predicate} {df_row.verb_form} "
    
    def extract_qadiscourse_inputs(self, x: pd.DataFrame) -> str:
        #TODO
        pass
    
    """
    Output Sequence preprocessing:
    """           
    def extract_all_permutations(self, x: pd.DataFrame) -> List[str]:
        """
        Extracts !n output sequences for every predicate, where n=|QAs|. 
        Each output sequence is a permutation of the QA order. 
        """
        qa_reprs = [f"{q}{self.special_tokens.separator_output_question_answer}{self._flatten_answers(t)}" 
                    for q, t in zip(x.question, x.answer)]
        permutations = utils.sample_permutations(qa_reprs, 
                                                 k=min(self.MAX_PERMUTATIONS, math.factorial(len(qa_reprs))))
        return [f"{self.special_tokens.separator_output_pairs}".join(permuted_qas)
                for permuted_qas in permutations]
        
    def extract_qa_permutations_as_qas(self, x: pd.DataFrame) -> List[str]:
        """
        Extracts n output sequences for every predicate, where n=|QAs|. 
        Each output sequence is sampled from the permutations of the QA order. 
        """
        qa_reprs = [f"{q}{self.special_tokens.separator_output_question_answer}{self._flatten_answers(t)}" 
                    for q, t in zip(x.question, x.answer)]
        n_permutations_to_sample = len(qa_reprs)
        sampled_permutations = utils.sample_permutations(qa_reprs, k=n_permutations_to_sample)
        return [f"{self.special_tokens.separator_output_pairs}".join(permuted_qas)
                for permuted_qas in sampled_permutations]
        
    def extract_qa_permutations_fixed(self, x: pd.DataFrame) -> List[str]:
        """
        Extracts k output sequences for every predicate. 
        Each output sequence is sampled from the permutations of the QA order. 
        """
        qa_reprs = [f"{q}{self.special_tokens.separator_output_question_answer}{self._flatten_answers(t)}" 
                    for q, t in zip(x.question, x.answer)]
        n_permutations_to_sample = self.sample_k_permutations_per_instance
        sampled_permutations = utils.sample_permutations(qa_reprs, k=n_permutations_to_sample, 
                                                         with_replacement=True)
        return [f"{self.special_tokens.separator_output_pairs}".join(permuted_qas)
                for permuted_qas in sampled_permutations]
        
    def extract_targets_arbitrary_order(self, x: pd.DataFrame) -> str:
        """
        Extracts ((question, answers), ...)
        """
        qa_reprs = [f"{q}{self.special_tokens.separator_output_question_answer}{self._flatten_answers(t)}" for q, t in zip(x.question, x.answer)]
        return [f"{self.special_tokens.separator_output_pairs}".join(qa_reprs)]
    
    def extract_targets_shuffled_order(self, x: pd.DataFrame) -> str:
        """
        Extracts ((question, answers), ...) in a randomized order.
        In order to take effect, one also needs to repeat preprocessing every epoch... 
        """
        qa_reprs = [f"{q}{self.special_tokens.separator_output_question_answer}{self._flatten_answers(t)}" for q, t in zip(x.question, x.answer)]
        random.shuffle(qa_reprs)
        return [f"{self.special_tokens.separator_output_pairs}".join(qa_reprs)]

    def extract_targets_by_answer_ordering(self, x: pd.DataFrame) -> str:
        """
        Extracts ((question, answers), ...)
        """
        qas = list(zip(x.question, x.answer, x.answer_range))
        # sort QAs by answer indices
        def sort_by_range(triplet):
            q,a,ranges=triplet
            return min(ranges) if ranges else 0
        qas = sorted(qas, key=sort_by_range)
        qa_reprs = [f"{q}{self.special_tokens.separator_output_question_answer}{self._flatten_answers(t)}" for q, t, _ in qas]
        return [f"{self.special_tokens.separator_output_pairs}".join(qa_reprs)]
     
    def extract_targets_only_answers(self, x: pd.DataFrame) -> str:
        """
        Extracts (answer, answer, ...)
        """

        return [f"{self.special_tokens.separator_output_answers}".join([f"{self._flatten_answers(t)}" 
                                                                       for q, t in zip(x.question, x.answer)])]

    def extract_targets_only_questions(self, x: pd.DataFrame) -> str:
        """
        Extracts (question, question, ...)
        """

        return [f"{self.special_tokens.separator_output_questions}".join([f"{q}" for q, t in zip(x.question, x.answer)])]

    def extract_targets_only_questions_first_word(self, x: pd.DataFrame) -> str:
        """
        Extracts (question, question, ...)
        """

        return [f"{self.special_tokens.separator_output_questions}".join([f"{q.split(' ')[0]}" for q, t in zip(x.question, x.answer)])]

    def extract_targets_only_first_two_question_answers(self, x: pd.DataFrame) -> str:
        """
        Extracts ((question, answer), (question, answer))
        """

        return [f"{self.special_tokens.separator_output_pairs}".join([f"{q}{self.special_tokens.separator_output_question_answer}{t[0]}" for q, t in list(zip(x.question, x.answer))[:2]])]

    def extract_targets_single(self, x: pd.DataFrame) -> str:
        """
        Extracts (quesiton, answers)
        """

        x = x.iloc[0]

        return [str([f"{q}{self.special_tokens.separator_output_question_answer}{t[0]}" for q, t in zip([x.question], [x.answer])])]

    def extract_qadiscourse_targets(self, x: pd.DataFrame) -> str:
        #TODO
        pass

    """ Helpers """

    def _flatten_answers(self, targets: List[str]) -> str:
        return f"{self.special_tokens.separator_output_answers}".join(targets)
    
