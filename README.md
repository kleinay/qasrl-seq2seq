# QASRL Seq2Seq
Parse QASRL using Seq2Seq technologies.

Relevant files and paths
* `run_summarization.py` - main file for train and predict.
* `qasrl-gs` submodule - used for evaluation of the results.
* [QASRL State machine example](https://github.com/eranhirs/qasrl_state_machine_example) - creates the necessary csv format for evaluation

# Combine QASRL-gs files
Run `python scripts/qasrl_gs_utils.py <input_tag_file> <input_sentences_file> <output_combined_file>`

Example `python scripts/qasrl_gs_utils.py qasrl_gs/data/gold/wikinews.dev.gold.csv qasrl_gs/data/sentences/wikinews.dev.full.csv qasrl_gs/data/gold/wikinews.dev.combined.csv`