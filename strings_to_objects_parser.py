import logging
from shutil import Error
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from datasets import Dataset
from spacy.matcher.phrasematcher import PhraseMatcher

from qasrl_gs.scripts.common import Role, Question, QuestionAnswer as OrigQASRLGSQuestionAnswer, STR_FORMAT_ANSWER_SEPARATOR
from constrained_decoding.qasrl import get_qasrl_question_dfa
from constrained_decoding.dfa import DFA
from dfa_fill_qasrl_slots import dfa_fill_qasrl_slots, extract_is_negated, SLOT_TO_STATE
QASRL_UNUSED_SLOT = "_"

# extend OrigQASRLGSQuestionAnswer for our needs here
@dataclass_json
@dataclass
class QuestionAnswer(OrigQASRLGSQuestionAnswer):
    raw_question: str = ''  # keeps further information, since model's raw generated question separates question-slots   

class S2SOutputError(Error):
    def __init__(self, *args: object, error_type='') -> None:
        super().__init__(*args)
        self.error_type = error_type

class StringsToObjectsParser:
    """
    This class converts from prediction strings to classes used in evaluation
    """

    def __init__(self,
                 special_tokens,
                 tokenizer
                 ):
        self.special_tokens = special_tokens
        self.qasrl_q_dfa: DFA = get_qasrl_question_dfa(constrain_verb=False)
        # Define qasrl_q_find_verb_dfa for finding verbs within the question
        # self.qasrl_q_find_verb_dfa = self.qasrl_q_dfa.copy()
        # self.qasrl_q_find_verb_dfa.accept_states = [3]    # verb slot

    def to_qasrl_gs_csv_format(self, predict_dataset: Dataset, predictions: List[str]) -> Tuple[List[QuestionAnswer], List[str]]:
        qasrl_predictions: List[QuestionAnswer] = []
        skipped_predictions = []
        predictions = [x.replace(self.special_tokens.pad_token, "").strip() for x in predictions]

        for prediction, sentence, qasrl_idx, predicate_idx, predicate in zip(predictions, predict_dataset['sentence'], predict_dataset['qasrl_id'], predict_dataset['predicate_idx'], predict_dataset['predicate']):
            questions_answers, skipped_pairs_strs = self._str_to_qasrl_gs_arguments(prediction, sentence, qasrl_idx, predicate_idx, predicate)
            qasrl_predictions.extend(questions_answers)
            skipped_predictions.extend(skipped_pairs_strs)

        logging.info(f"Skipped invalid QASRL format pairs ; len(skipped_pairs_strs) {len(skipped_pairs_strs)} ; for example, {skipped_pairs_strs[:3]}")

        return qasrl_predictions, skipped_predictions

    def _str_to_qasrl_gs_arguments(self, prediction_seq: str, sentence: str, qasrl_idx: str, predicate_idx: int, predicate:str) -> Tuple[List[QuestionAnswer], List[str]]:
        questions_answers = []
        invalid_qa_pairs_strs = []
        pairs_strs = prediction_seq.split(self.special_tokens.separator_output_pairs)
        for pair_str in pairs_strs:
            try:
                question_str, arguments_strs = pair_str.split(self.special_tokens.separator_output_question_answer)
                # since every slot can be multiple word (e.g. "how much", "should not"), we will use the slot-based qasrl automaton for identifying the verb slot. 
                question_slots, is_valid_qasrl_question = self._get_question_slots(question_str)
                clean_question_str = self._clean_question(question_str)
                arguments = arguments_strs.split(self.special_tokens.separator_output_answers)
                clean_arguments_objs = [self._clean_generated_string(argument) for argument in arguments]
                arguments_ranges_objs = [find_argument_answer_range(argument, sentence) for argument in clean_arguments_objs]
                arguments_str = QuestionAnswer.answer_obj_to_str(clean_arguments_objs)
                arguments_ranges_str = QuestionAnswer.answer_range_obj_to_str(arguments_ranges_objs)
                if not is_valid_qasrl_question:
                    # log that question is invalid w.r.t. QASRL format, but also keep QA in returned predicted QAs,
                    #  so that one can make use of the argument with the invalid (yet perhaps useful) questions
                    invalid_qa_pairs_strs.append(("Invalid QASRL question format", pair_str, sentence))
                qa = QuestionAnswer(qasrl_id=qasrl_idx, verb_idx=predicate_idx, verb=predicate, 
                                    question=clean_question_str, raw_question=question_str,
                                          answer=arguments_str, answer_range=arguments_ranges_str, **question_slots)
                questions_answers.append(qa)
            except S2SOutputError as e:
                logging.debug("Bad output, error: ", e)
                invalid_qa_pairs_strs.append((e.error_type, pair_str, sentence))
            except Exception as e:
                logging.debug("Bad output-format, error: ", e)
                invalid_qa_pairs_strs.append(("Bad-format", pair_str, sentence))

        return questions_answers, invalid_qa_pairs_strs


    def _get_question_slots(self, question_str: str) -> Tuple[Dict[str, str], bool]:
        try:
            question_slots = dfa_fill_qasrl_slots(question_str.strip(), self.qasrl_q_dfa)
        except Exception as e:
            raise S2SOutputError(f"QASRL-Automaton error. Predicted question: '{question_str}', error: {e}", 
                                 error_type="QASRL question automaton error")

        if question_slots is None:
            question_invalid_slots = {sl: "" for sl in SLOT_TO_STATE.keys()}
            question_invalid_slots["verb_form"] = ""
            return question_invalid_slots, False
                
        question_slots["is_negated"] = extract_is_negated(question_slots)
        question_slots["verb_form"] = question_slots.pop("verb")
        return question_slots, True
        

    # def _clean_question_and_verb(self, question_str: str) -> Tuple[str, str]:
    #     clean_question = self._clean_question(question_str)
    #     verb = self._find_verb_by_slot_dfa(question_str)
    #     return clean_question, verb

    # def _find_verb_by_slot_dfa(self, question_str: str) -> str:
    #     words_of_q = question_str.split(" ")
    #     for i in range(len(words_of_q)):
    #         automaton_input = ' '.join(words_of_q[:i])
    #         if self.qasrl_q_find_verb_dfa(automaton_input)[2]:
    #             return words_of_q[i]
    #     # verb not found
    #     raise S2SOutputError(f"No verb found in question. Predicted question: '{question_str}'", error_type="No verb in question") 
        
    def _clean_question(self, question_str: str) -> str:
        return self._clean_generated_string(question_str.replace(f"{QASRL_UNUSED_SLOT} ","").replace(f"{QASRL_UNUSED_SLOT}?", "?").replace(f" ?", "?"))

    def _clean_generated_string(self, generated_string: str) -> str:
        if self.special_tokens.bos_token is not None:
            generated_string = generated_string.replace(self.special_tokens.bos_token, "")
        return generated_string.replace(self.special_tokens.eos_token, "").strip()


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
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("argument", [argument_spacy])
    matches = matcher(input_spacy)

    if not any(matches):
        raise S2SOutputError(f"No matches found ; argument: '{argument}' ; input: '{input}'", error_type="Answer not found in sentence")

    first_match = matches[0]
    return first_match[1], first_match[2]

# def find_verb_in_question(question: str) -> str:
#     """
#     Given a QASRL question (e.g. predicted as full-string by model), return the verb inside it.
#     Returns the last token identified as 'Verb' by Spacy's POS-tagger.
#     """ 
#     nlp = get_spacy('en_core_web_sm')
#     doc = nlp(question)
#     verbs_in_question = [str(t) for t in doc if t.pos_ == "VERB"]
#     if not any(verbs_in_question):
#         raise S2SOutputError(f"No verb found in question. Predicted question: '{question}'", error_type="No verb in question")
#     return verbs_in_question[-1]