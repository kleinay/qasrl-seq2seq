# QASRL Seq2Seq
Parse QASRL using Seq2Seq technologies.

# Relevant files and paths
* `run_summarization.py` - main file for train and predict.
* [QASRL State machine example](https://github.com/eranhirs/qasrl_state_machine_example) - analyze the question format for evaluation.
* `run_evaluation` - main file for evaluation of the results.
* `qasrl-gs` submodule - used for evaluation of the results.

# How to run
Run `qasrl_notebook.ipynb`. It will download whatever is necessary, train, and evaluate.

# Combine QASRL-gs files
Run `python scripts/qasrl_gs_utils.py <input_tag_file> <input_sentences_file> <output_combined_file>`

Example `python scripts/qasrl_gs_utils.py qasrl_gs/data/gold/wikinews.dev.gold.csv qasrl_gs/data/sentences/wikinews.dev.full.csv qasrl_gs/data/gold/wikinews.dev.combined.csv`