import logging
from typing import List, Tuple, Dict, Optional

from datasets import Dataset
from spacy.matcher.phrasematcher import PhraseMatcher

from qasrl_gs.scripts.common import Role, Question, QuestionAnswer, STR_FORMAT_ANSWER_SEPARATOR

QASRL_UNUSED_SLOT = "_"


class StringsToObjectsParser:
    """
    This class converts from prediction strings to classes used in evaluation
    """

    def __init__(self,
                 separator_input_question_predicate: str,
                 separator_output_answers: str,
                 separator_output_questions: str,
                 separator_output_question_answer: str,
                 separator_output_pairs: str,
                 bos_token: Optional[str],
                 eos_token: str,
                 pad_token: str
                 ):
        self.separator_input_question_predicate = separator_input_question_predicate
        self.separator_output_answers = separator_output_answers
        self.separator_output_questions = separator_output_questions
        self.separator_output_question_answer = separator_output_question_answer
        self.separator_output_pairs = separator_output_pairs
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

    def to_qasrl_gs_csv_format(self, predict_dataset: Dataset, predictions: List[str]) -> List[QuestionAnswer]:
        qasrl_predictions: List[QuestionAnswer] = []
        predictions = [x.replace(self.pad_token, "").strip() for x in predictions]

        for prediction, sentence, qasrl_idx, predicate_idx, predicate in zip(predictions, predict_dataset['sentence'], predict_dataset['qasrl_indices'], predict_dataset['predicates_indices'], predict_dataset['predicates']):
            qasrl_predictions.extend(self._str_to_qasrl_gs_arguments(prediction, sentence, qasrl_idx, predicate_idx, predicate))

        return qasrl_predictions

    def _str_to_qasrl_gs_arguments(self, question_str: str, sentence: str, qasrl_idx: str, verb_idx: int, verb:str) -> List[QuestionAnswer]:
        questions_answers = []
        skipped_pairs_strs = []
        pairs_strs = question_str.split(self.separator_output_pairs)
        for pair_str in pairs_strs:
            try:
                question_str, arguments_strs = pair_str.split(self.separator_output_question_answer)
                clean_question_str = self._clean_question(question_str)
                arguments = arguments_strs.split(self.separator_output_answers)
                clean_arguments_objs = [self._clean_generated_string(argument) for argument in arguments]
                arguments_ranges_objs = [find_argument_answer_range(argument, sentence) for argument in clean_arguments_objs]
                arguments_str = QuestionAnswer.answer_obj_to_str(clean_arguments_objs)
                arguments_ranges_str = QuestionAnswer.answer_range_obj_to_str(arguments_ranges_objs)
                question = QuestionAnswer(qasrl_id=qasrl_idx, verb_idx=verb_idx, verb=verb, question=clean_question_str, answer=arguments_str, answer_range=arguments_ranges_str)
                questions_answers.append(question)
            except:
                skipped_pairs_strs.append(pair_str)

        logging.info(f"Skipped invalid QASRL format pairs ; len(skipped_pairs_strs) {len(skipped_pairs_strs)} ; example {skipped_pairs_strs[:5]}")
        return questions_answers

    def _clean_question(self, question_str: str) -> str:
        return self._clean_generated_string(question_str.replace(f"{QASRL_UNUSED_SLOT} ","").replace(f"{QASRL_UNUSED_SLOT}?", "?").replace(f" ?", "?"))

    def _clean_generated_string(self, generated_string: str) -> str:
        if self.bos_token is not None:
            generated_string = generated_string.replace(self.bos_token, "")
        return generated_string.replace(self.eos_token, "").strip()


SPACY_MODELS = {}


def get_spacy(lang, **kwargs):
    import spacy

    if lang not in SPACY_MODELS:
        SPACY_MODELS[lang] = spacy.load(lang, **kwargs)
    return SPACY_MODELS[lang]


def find_argument_answer_range(argument: str, input: str) -> Tuple[int, int]:
    """
    Given an argument (predicted answer) and an input (original sentence), finds the range
    """

    nlp = get_spacy('en_core_web_sm')
    input_spacy = nlp(input)
    argument_spacy = nlp(argument)
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add("argument", [argument_spacy])
    matches = matcher(input_spacy)

    if not any(matches):
        raise ValueError(f"No matches found ; argument {argument} ; input {input}")

    first_match = matches[0]
    return first_match[1], first_match[2]
