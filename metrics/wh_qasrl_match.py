# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" 
Simple strict evaluation for QASRL - Exact match on answer jointly with match on WH-word in question .
"""

import datasets


_CITATION = ""

# TODO: Add description of the metric here
_DESCRIPTION = """\
Simple strict evaluation for QASRL - Exact match on answer jointly with match on WH-word in question ."""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string (predicted output sequence).
    references: list of reference for each prediction. Each
        reference should be a string (reference output sequence).
    qa_pairs_sep (str): the string by which separates between QA pairs [QA sep QA].
    qa_sep (str): the string by which separates between the question and the answer [Q sep A].  
Returns:
    precision:
    recall: 
    f1: 
Examples:
    >>> metric = datasets.load_metric("wh_qasrl_match")
    >>> results = metric.compute(predictions=["when? % 1990 | what was there? % the student | Why? % not too long"], 
                                 references=["when? % 1990 | who was there? % the student | Why? % not long"],
                                 qa_pairs_sep="|", 
                                 qa_sep="%")
    >>> print(results)
    {'precision': 0.3333333333333333, 'recall': 0.3333333333333333, 'f1': 0.3333333333333333}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ElementExactMatch(datasets.Metric):
    """A simple metric for computing how many elements in predicted sequences were excatly the same as the elements in ground truth."""

    def _info(self):
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            }),
        )

    def _compute(self, predictions, references, qa_pairs_sep: str, qa_sep: str):
        """Returns the scores"""
        tp, fp, fn = 0, 0, 0
        for pred, ref in zip(predictions, references):
            pred_qas = pred.split(qa_pairs_sep)
            ref_qas = ref.split(qa_pairs_sep)
            pred_qas = [qa.lstrip().split(qa_sep) for qa in pred_qas if len(qa.lstrip().split(qa_sep))==2]
            ref_qas = [qa.lstrip().split(qa_sep) for qa in ref_qas if len(qa.lstrip().split(qa_sep))==2]
            pred_wh_and_answer = [(question.lstrip().split(" ")[0], answer) for question, answer in pred_qas]
            ref_wh_and_answer = [(question.lstrip().split(" ")[0], answer) for question, answer in ref_qas]

            tp += len(set(pred_wh_and_answer) & set(ref_wh_and_answer))
            fp += len(set(pred_wh_and_answer) - set(ref_wh_and_answer))
            fn += len(set(ref_wh_and_answer) - set(pred_wh_and_answer))
  
        recall = 0 if (tp + fn)==0 else tp / (tp + fn) 
        precision = 0 if (tp + fp)==0 else tp / (tp + fp)
        f1 = 0 if (recall==0 or precision==0) else  2*(recall * precision) / (recall + precision) 

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }