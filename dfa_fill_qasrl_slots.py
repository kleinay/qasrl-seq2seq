from typing import List, Optional, Dict, Union

from qasrl_gs.scripts.common import QuestionAnswer
from seq2seq_constrained_decoding.constrained_decoding.dfa import DFA

STATE_TO_SLOT = {
    0: "wh",
    1: "aux",
    2: "subj",
    3: "verb",
    "<1>_0": "obj",
    "<1>_1": "prep",
    "<1>_2": "obj2",
}

SLOT_TO_STATE = {slot: state for state, slot in STATE_TO_SLOT.items()}
Slots = Dict[str, str]


def extract_is_negated(q_slots: Slots) -> bool:
    " Return `is_negated` based on the 7 surface question slots (aux and verb). "
    return (q_slots["aux"].endswith("n't") or
            (len(q_slots["verb"].split(" "))>1 and q_slots["verb"].split(" ")[0] == "not") )

def dfa_fill_qasrl_slots(predicted_question: str, question_dfa: DFA) -> Optional[Slots]:
    """
    Use DFA to fill QASRL slots
    """

    lowered_question = predicted_question.lower()
    tokenized_question = lowered_question.split(" ")
    # handle '?' edge cases
    if tokenized_question[-1].endswith('?'):
        if not tokenized_question[-1] == '?':
            # seperate '?' to be the 8th slot when there is no space before it
            tokenized_question = tokenized_question[:-1] + [tokenized_question[-1][:-1], '?']
    else:
    # add '? as the 8th slot if non existent
        tokenized_question.append('?')
        
    slots: Optional[Slots] = _parse_token(tokenized_question, 0, {}, [], question_dfa)

    if slots:
        return slots


def _parse_token(tokenized_question, curr_state, slots: Slots, previous_tokens: List[str], question_dfa: DFA) -> Optional[Slots]:
    if not(any(tokenized_question)):
        return None
    elif tokenized_question[0] == "?":
        if len(slots) == 7:
            return slots
        else:
            return None

    token = tokenized_question.pop(0)
    success, next_state, _ = question_dfa(previous_tokens + [token])
    if success:
        # Save slot
        slots[STATE_TO_SLOT[curr_state]] = token

        # Keep track of previous tokens (for next inference of DFA)
        previous_tokens.append(token)

        # Update state
        curr_state = next_state

        # Recursive call
        return _parse_token(tokenized_question, curr_state, slots, previous_tokens, question_dfa)
    else:
        # verb is a wildcard. In this case, it is possible this token belongs to the previous part
        is_previous_slot_wildcard = curr_state == SLOT_TO_STATE['obj']
        if is_previous_slot_wildcard:
            # Remove last token and concatenate with current token
            tokenized_question_copy = tokenized_question.copy()
            previous_tokens_copy = previous_tokens.copy()
            previous_token = previous_tokens_copy.pop()
            tokenized_question_copy.insert(0, " ".join([previous_token, token]))

            # Recursive call
            slots_by_verb = _parse_token(tokenized_question_copy, SLOT_TO_STATE['verb'], slots.copy(), previous_tokens_copy, question_dfa)
            if slots_by_verb:
                return slots_by_verb

        if not any(tokenized_question):
            return None

        # Try adding next token to current token
        tokenized_question[0] = " ".join([token, tokenized_question[0]])

        # Recursive call
        return _parse_token(tokenized_question, curr_state, slots, previous_tokens, question_dfa)
