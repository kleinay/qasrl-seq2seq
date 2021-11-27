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
""" Exact Match for seq2seq model - a simple metric for computing how many predicted sequences were excatly the same as the ground truth; simple string match."""

import datasets


_CITATION = ""

# TODO: Add description of the metric here
_DESCRIPTION = """\
Exact Match is a simple metric for computing how many predicted sequences were excatly the same as the ground truth output sequences.
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string (predicted output sequence).
    references: list of reference for each prediction. Each
        reference should be a string (reference output sequence).
Returns:
    accuracy: description of the first score
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> metric = datasets.load_metric("exact_match")
    >>> results = metric.compute(references=['hi', 'there', '!'], predictions=['hi', 'their', '!'])
    >>> print(results)
    {'accuracy': 0.6666666666666666}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ExactMatch(datasets.Metric):
    """A simple metric for computing exact string match between prediction and reference (on the instance level)."""

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
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

    def _compute(self, predictions, references):
        """Returns the scores"""
        accuracy = sum(i == j for i, j in zip(predictions, references)) / len(predictions)

        return {
            "accuracy": accuracy,
        }