import logging
from typing import List, Tuple

from spacy.matcher.phrasematcher import PhraseMatcher

from qasrl_gs.scripts.common import Role, Question


class StringsToObjectsParser:
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

    def to_qasrl_gs_arguments(self, inputs: List[str], labels: List[str], predictions: List[str]) -> Tuple[
        List[Role], List[Role]]:
        labels_parsed = []
        predictions_parsed = []
        for input, label, prediction in zip(inputs, labels, predictions):
            labels_parsed.extend(self._str_to_qasrl_gs_arguments(label, input))
            predictions_parsed.extend(self._str_to_qasrl_gs_arguments(prediction, input))

        return labels_parsed, predictions_parsed

    def _str_to_qasrl_gs_arguments(self, role_str: str, input: str) -> List[Role]:
        roles = []
        skipped_pairs_strs = []
        pairs_strs = role_str.split(self.separator_output_pairs)
        for pair_str in pairs_strs:
            try:
                question_str, arguments_strs = pair_str.split(self.separator_output_question_answer)
                arguments = arguments_strs.split(self.separator_output_answers)
                question_split_str = question_str.strip().split(" ")
                if len(question_split_str) != 7:
                    raise ValueError("Not a valid QASRL question format")
                question_dict = {
                    'wh': question_split_str[0].strip(),
                    'subj': question_split_str[2].strip(),
                    'obj': question_split_str[4].strip(),
                    'aux': question_split_str[1].strip(),
                    'prep': question_split_str[5].strip(),
                    'obj2': question_split_str[6].replace("?", "").strip(),
                    'is_passive': False,
                    'is_negated': False
                }

                question = Question(text=question_str, **question_dict)
                arguments = tuple([find_argument_answer_range(argument.replace(self.eos_token, "").strip(), input) for argument in arguments])
                roles.append(Role(question, arguments))
            except:
                skipped_pairs_strs.append(pair_str)
        logging.info(f"Skipped invalid QASRL format pairs ; len(skipped_pairs_strs) {len(skipped_pairs_strs)} ; example {skipped_pairs_strs[:5]}")
        return roles


SPACY_MODELS = {}


def get_spacy(lang, **kwargs):
    import spacy

    if lang not in SPACY_MODELS:
        SPACY_MODELS[lang] = spacy.load(lang, **kwargs)
    return SPACY_MODELS[lang]


def find_argument_answer_range(argument: str, input: str) -> Tuple[int, int]:
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
