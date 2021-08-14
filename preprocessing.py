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
                 eos_token: str
                 ):
        self.separator_input_question_predicate = separator_input_question_predicate
        self.separator_output_answers = separator_output_answers
        self.separator_output_questions = separator_output_questions
        self.separator_output_question_answer = separator_output_question_answer
        self.separator_output_pairs = separator_output_pairs
        self.eos_token = eos_token

    def extract_inputs(self, x: pd.DataFrame) -> str:
        # all rows have the same index values because of the groupby
        any_row = x.iloc[0]
        return f"{any_row['input']}{self.separator_input_question_predicate}{any_row['predicate']}"
        #             return f"{any_row['input']}{tokenizer.eos_token}{any_row['predicate']}"

    def extract_targets_all(self, x: pd.DataFrame) -> str:
        """
        Extracts ((question, answers), ...)
        """

        return f"{self.separator_output_pairs}".join([f"{q}{self.separator_output_question_answer}{self._flatten_targets(t)}" for q, t in zip(x.question, x.target)])

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

    def _flatten_targets(self, targets: List[str]) -> str:
        return f"{self.separator_output_answers}".join(targets)
