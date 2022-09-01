# QASRL Seq2Seq
Parse QASRL using Seq2Seq technologies.

## Important files and paths
* `notebook.py` - convenient entry point, with 3 relevant API functions:
  * `full_experiment` - train and evaluate a new model. See a few example calls at [small_experiments.py](small_experiments.py).
  * `load_and_evaluate` - evaluate an existing model.
  * `load_and_predict` - run inference on new texts using an existing model.
* `run_summarization.py` - main file for train, evaluate and predict logics. It is adapted from Huggingface script but quite havily refactored. 
* `pipeline.py` - introduces a wrapper pipeline class for easily running text-to-text inference. Can load a model from Huggingface Hub or from a local directory. 
* [QASRL State machine example](https://github.com/eranhirs/qasrl_state_machine_example) - analyze the question format for evaluation.
* `run_evaluation` - main file for the evaluation logics. Since we compare with previous QA-SRL and QANom works, we utilize their evaluation functionality from their own pakages, and just adapt the input format per package.
* `qasrl-gs` submodule - used for evaluation of the results.


