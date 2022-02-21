
from constrained_decoding.qasrl_constrained_decoding import get_qasrl_question_dfa
from dfa_fill_qasrl_slots import dfa_fill_qasrl_slots


def test_dfa_fill_qasrl_slots__valid():
    predicted_qa = f"Who might _ be orientated _ _ _ ?"
    question_dfa = get_qasrl_question_dfa(constrain_verb=False)
    actual_slots = dfa_fill_qasrl_slots(predicted_qa, question_dfa)

    assert actual_slots == {
        "wh": "who",
        "aux": "might",
        "subj": "_",
        "verb": "be orientated",
        "obj": "_",
        "obj2": "_",
        "prep": "_"
    }


def test_dfa_fill_qasrl_slots__invalid_missing_slots():
    predicted_qa = f"Who might _ be orientated _ _ ?"
    question_dfa = get_qasrl_question_dfa(constrain_verb=False)
    actual_slots = dfa_fill_qasrl_slots(predicted_qa, question_dfa)

    assert actual_slots is None


def test_dfa_fill_qasrl_slots__unfamiliar_wh():
    predicted_qa = f"Whom might _ be orientated _ _ _ ?"
    question_dfa = get_qasrl_question_dfa(constrain_verb=False)
    actual_slots = dfa_fill_qasrl_slots(predicted_qa, question_dfa)

    assert actual_slots is None

