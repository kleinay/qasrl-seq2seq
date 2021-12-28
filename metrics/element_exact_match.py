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
Element Exact Match for seq2seq model - 
a simple metric for computing how many elements in predicted sequences were excatly the same as the elements in ground truth; 
simple string match for each element.
An element is deterimned by splitting the sequence using a separator string, given as parameter.
"""

import datasets


_CITATION = ""

# TODO: Add description of the metric here
_DESCRIPTION = """\
Element Exact Match is a simple metric for computing how many elements in predicted sequences were excatly the same as the elements in ground truth; 
simple string match for each element.
An element is deterimned by splitting the sequence(s) using a `separator` string, given as parameter.
The metric also support an `as_set` mode, to regard the elements as a set (not considering order)."""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string (predicted output sequence).
    references: list of reference for each prediction. Each
        reference should be a string (reference output sequence).
    separator (str, default=' '): the string by which sequences should be splitted to elements.
    as_set (bool, default=True): whether to regard the elements as a set (not ordered) or as a sequence (ordered)  
Returns:
    precision:
    recall: 
    f1: 
Examples:
    >>> metric = datasets.load_metric("element_exact_match")
    >>> results = metric.compute(predictions=["let me lead you"], references=["don't you lead"])
    >>> print(results)
    {'precision': 0.5, 'recall': 0.6666666666666666, 'f1': 0.5714285714285715}
    >>> results = metric.compute(predictions=["let me lead you"], references=["don't you lead"], as_set=False)
    >>> print(results)
    {'precision': 0.25, 'recall': 0.3333333333333333, 'f1': 0.28571428571428575}
    
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

    def _compute(self, predictions, references, separator=' ', as_set=True):
        """Returns the scores"""
        tp, fp, fn = 0, 0, 0
        for pred, ref in zip(predictions, references):
            pred_elements = pred.split(separator)
            ref_elements = ref.split(separator)
            if as_set:
                tp += len(set(pred_elements) & set(ref_elements))
                fp += len(set(pred_elements) - set(ref_elements))
                fn += len(set(ref_elements) - set(pred_elements))
            else:
                # match with keeping order
                cur_ref_index, matches = 0, 0
                for pred_element in pred_elements:
                    if pred_element in ref_elements[cur_ref_index:]:
                        matches += 1
                        cur_ref_index = ref_elements.index(pred_element)
                    else:
                        fp += 1
                tp += matches
                fn += len(ref_elements) - matches
        
        recall = 0 if (tp + fn)==0 else tp / (tp + fn) 
        precision = 0 if (tp + fp)==0 else tp / (tp + fp)
        f1 = 0 if (recall==0 or precision==0) else  2*(recall * precision) / (recall + precision) 

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }