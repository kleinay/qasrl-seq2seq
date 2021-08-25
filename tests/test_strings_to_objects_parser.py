from qasrl_gs.scripts.common import QuestionAnswer, Argument
from strings_to_objects_parser import find_argument_answer_range, StringsToObjectsParser
from mock import MagicMock


def test_find_argument_answer_range():
    argument = "the French Community of Belgium"
    input = "summarize: Additionally, the French Community of Belgium has controversially begun referring to itself exclusively as the <unk> Wallonia-Brussels Federation'to emphasize the links between the French Community, Wallonia and Brussels.<extra_id_1> referring</s>"
    result = find_argument_answer_range(argument, input)
    assert result == (4, 9)


class TestStringsToObjectsParser:
    def test_to_qasrl_gs_csv_format(self):
        predictions = ["</s><s>who did someone lodge _ with _?<extra_id_5>other medical students <extra_id_3> Henry Stephens<extra_id_7>where did someone lodge _ _ _ ?<extra_id_5>in Southwark</s><pad><pad><pad><pad>"]

        dataset = {
            "sentence": ["He lodged near the hospital at 28 St Thomas \'s Street in Southwark , with other medical students , including Henry Stephens who became a famous inventor and ink magnate ."],
            "qasrl_indices": ["some_id"],
            "predicates": ["lodged"],
            "predicates_indices": [1]
        }

        separator_output_answers = "<extra_id_3>"
        separator_output_question_answer = "<extra_id_5>"
        separator_output_pairs = "<extra_id_7>"
        bos_token = "<s>"  # Only relevant for BART
        eos_token = "</s>"
        pad_token = "<pad>"
        strings_to_objects_parser = StringsToObjectsParser(MagicMock(), separator_output_answers, MagicMock(), separator_output_question_answer, separator_output_pairs, bos_token, eos_token, pad_token)

        result = strings_to_objects_parser.to_qasrl_gs_csv_format(dataset, predictions)

        assert result == [
            QuestionAnswer(
                qasrl_id="some_id",
                verb_idx=1,
                verb="lodged",
                question="who did someone lodge with?",
                answer="other medical students~!~Henry Stephens",
                answer_range="15:18~!~20:22"
            ),
            QuestionAnswer(
                qasrl_id="some_id",
                verb_idx=1,
                verb="lodged",
                question="where did someone lodge?",
                answer="in Southwark",
                answer_range="11:13"
            )
        ]
