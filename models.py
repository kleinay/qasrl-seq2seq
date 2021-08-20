from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json, LetterCase


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class InputExample:
    sentence: str
    predicate: str
    predicate_idx: int
    gold_question: Optional[str] = None
    predicted_question: Optional[str] = None
