from builtins import ValueError
from dataclasses import dataclass
from typing import List

import pandas as pd

# @dataclass
# class Separators:
#     separator_input_question_predicate: str
#     separator_output_answers: str
#     separator_output_questions: str
#     separator_output_question_answer: str
#     separator_output_pairs: str
#     eos_token: str


class Preprocessor:
    def __init__(self,
                 separator_input_question_predicate: str,
                 separator_output_answers: str,
                 separator_output_questions: str,
                 separator_output_question_answer: str,
                 separator_output_pairs: str,
                 marker_generic_predicate: str,
                 marker_verbal_predicate: str,
                 marker_nominalization_predicate: str,
                 eos_token: str
                 ):
        self.separator_input_question_predicate = separator_input_question_predicate
        self.separator_output_answers = separator_output_answers
        self.separator_output_questions = separator_output_questions
        self.separator_output_question_answer = separator_output_question_answer
        self.separator_output_pairs = separator_output_pairs
        self.marker_generic_predicate = marker_generic_predicate
        self.marker_verbal_predicate = marker_verbal_predicate
        self.marker_nominalization_predicate = marker_nominalization_predicate
        self.eos_token = eos_token

    """
    Prefix for Sequence preprocessing:
    """
    def get_sequence_prefix(self, x: pd.DataFrame, prefix_or_prefixing_method: str) -> str:
        row = x.iloc[0]
        if prefix_or_prefixing_method is None:
            return ''
        if not prefix_or_prefixing_method.startswith("<"):  # Regular prefix - not dependent on input row x
            return prefix_or_prefixing_method
        if prefix_or_prefixing_method == "<predicate-type>":
            if "predicate_type" not in row:
                raise ValueError("source_prefix is '<predicate-type>' but input row has no 'predicate_type' attribute.")
            else:
                pred_type = x["predicate_type"]
                return f"Generate QAs for {pred_type} QASRL: "
        
        
        raise ValueError(f"source_prefix '{prefix_or_prefixing_method}' starts with '<' but does not correspond to a valid prefixing method. ")

    """
    Input Sequence preprocessing:
    """
    
    def extract_inputs(self, x: pd.DataFrame) -> str:
        """ Encode predicate by repeating it at the end of sequence """
        # all rows have the same index values (and predicate-tailored info) because of the groupby
        row = x.iloc[0]
        sentence_before_predicate, predicate, sentence_after_predicate = self._get_splitted_sentence_by_predicate(row)
        seq = f"{sentence_before_predicate} {predicate} {sentence_after_predicate} {self.separator_input_question_predicate} {predicate}"
        # embed also the verb_form
        seq = self._append_verb_form(seq, row)
        
        # append predicate_type
        # if "predicate_type" in row:   # only if we train on joint-qasrl/joint-qanom datatset
        #     seq = f'{row["predicate_type"]} | {seq}' 
        return seq
    
    def extract_inputs_predicate_inline_marker(self, x: pd.DataFrame) -> str:
        """ Encode predicate by prefixing it with a marker """
        # all rows have the same index values (and predicate-tailored info) because of the groupby
        row = x.iloc[0]
        sentence_before_predicate, predicate, sentence_after_predicate = self._get_splitted_sentence_by_predicate(row)
        
        # prepare predicate marker
        """ In case we want a generic marker for all predicate types: """
        predicate_marker = self.marker_generic_predicate    
        """ In case we want special marker for each predicate type: """
        if "predicate_type" in row:
            predicate_marker = {"verbal": self.marker_verbal_predicate, 
                                "nominal": self.marker_nominalization_predicate
                                }[row["predicate_type"]]
        seq = f"{sentence_before_predicate} {predicate_marker} {predicate} {sentence_after_predicate}"
        
        # embed also the verb_form
        # In this function, since we don't repeat the predicate, separator_input_question_predicate prefixes the verb_form
        seq = self._append_verb_form(seq, row)
        
        # append predicate_type (if not captured by in predicate_marker)
        if "predicate_type" in row and predicate_marker == self.marker_generic_predicate:
            seq = f'{row["predicate_type"]} | {seq}' 
        return seq
    
    def _get_splitted_sentence_by_predicate(self, row: pd.Series):
        sentence = row.input
        sent_tokens = sentence.split(" ") 
        sentence_before_predicate = " ".join([sent_tokens[i] for i in range(int(row.predicate_idx))])
        predicate = row.predicate
        sentence_after_predicate = " ".join([sent_tokens[i] for i in range(int(row.predicate_idx)+1, len(sent_tokens))])
        return sentence_before_predicate, predicate, sentence_after_predicate
        
    def _append_verb_form(self, seq: str, df_row: pd.Series):
        if "verb_form" not in df_row or df_row.verb_form is None:
            return f"{seq} "
        else:
            return f"{seq} {self.separator_input_question_predicate} {df_row.verb_form} "
    
    def extract_qadiscourse_inputs(self, x: pd.DataFrame) -> str:
        #TODO
        pass
    
    """
    Output Sequence preprocessing:
    """        

    def extract_targets_all(self, x: pd.DataFrame) -> str:
        """
        Extracts ((question, answers), ...)
        """
        qa_reprs = [f"{q}{self.separator_output_question_answer}{self._flatten_targets(t)}" for q, t in zip(x.question, x.target)]
        return f"{self.separator_output_pairs}".join(qa_reprs)

    def extract_targets_all_by_answer_ordering(self, x: pd.DataFrame) -> str:
        """
        Extracts ((question, answers), ...)
        """
        qas = list(zip(x.question, x.target, x.answer_ranges))
        # sort QAs by answer indices
        def sort_by_range(triplet):
            q,a,ranges=triplet
            return min(ranges) if ranges else 0
        qas = sorted(qas, key=sort_by_range)
        qa_reprs = [f"{q}{self.separator_output_question_answer}{self._flatten_targets(t)}" for q, t, _ in qas]
        return f"{self.separator_output_pairs}".join(qa_reprs)
     
    def extract_targets_only_answers(self, x: pd.DataFrame) -> str:
        """
        Extracts (answer, answer, ...)
        """

        return f"{self.separator_output_answers}".join([f"{self._flatten_targets(t)}" for q, t in zip(x.question, x.target)])

    def extract_targets_only_questions(self, x: pd.DataFrame) -> str:
        """
        Extracts (question, question, ...)
        """

        return f"{self.separator_output_questions}".join([f"{q}" for q, t in zip(x.question, x.target)])

    def extract_targets_only_questions_first_word(self, x: pd.DataFrame) -> str:
        """
        Extracts (question, question, ...)
        """

        return f"{self.separator_output_questions}".join([f"{q.split(' ')[0]}" for q, t in zip(x.question, x.target)])

    def extract_targets_only_first_two_question_answers(self, x: pd.DataFrame) -> str:
        """
        Extracts ((question, answer), (question, answer))
        """

        return f"{self.separator_output_pairs}".join([f"{q}{self.separator_output_question_answer}{t[0]}" for q, t in list(zip(x.question, x.target))[:2]])

    def extract_targets_single(self, x: pd.DataFrame) -> str:
        """
        Extracts (quesiton, answers)
        """

        x = x.iloc[0]

        return str([f"{q}{self.separator_output_question_answer}{t[0]}" for q, t in zip([x.question], [x.target])])

    def extract_qadiscourse_targets(self, x: pd.DataFrame) -> str:
        #TODO
        pass

    def _flatten_targets(self, targets: List[str]) -> str:
        return f"{self.separator_output_answers}".join(targets)
