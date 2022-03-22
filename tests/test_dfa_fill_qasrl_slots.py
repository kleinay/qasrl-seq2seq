
# from qasrl_constrained_decodingseq2seq_constrained_decoding.constrained_decoding.qasrl_constrained_decoding import get_qasrl_question_dfa
from seq2seq_constrained_decoding.constrained_decoding.qasrl_constrained_decoding import get_qasrl_question_dfa
from dfa_fill_qasrl_slots import dfa_fill_qasrl_slots


def test_dfa_fill_qasrl_slots__valid():
    question_dfa = get_qasrl_question_dfa(constrain_verb=False)

    predicted_qa = f"When might someone orient _ _ _ ?"
    assert dfa_fill_qasrl_slots(predicted_qa, question_dfa) is not None #["wh"] == "when"
    predicted_qa = f"How long might someone be orientated _ _ _ ?"
    assert dfa_fill_qasrl_slots(predicted_qa, question_dfa) is not None #["wh"] == "how long"
    predicted_qa = f"How might someone be orientated _ _ _ ?"
    assert dfa_fill_qasrl_slots(predicted_qa, question_dfa) is not None #["wh"] == "how"
    
    predicted_qa = f"Who might _ be orientated _ _ _ ?"
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

if __name__ == "__main__":
    test_dfa_fill_qasrl_slots__valid()
