from strings_to_objects_parser import find_argument_answer_range


def test_find_argument_answer_range():
    argument = "the French Community of Belgium"
    input = "summarize: Additionally, the French Community of Belgium has controversially begun referring to itself exclusively as the <unk> Wallonia-Brussels Federation'to emphasize the links between the French Community, Wallonia and Brussels.<extra_id_1> referring</s>"
    result = find_argument_answer_range(argument, input)
    assert result == (4, 9)
