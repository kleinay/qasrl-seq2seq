from argparse import ArgumentParser
from typing import Optional

import wandb

from qasrl_gs.scripts.evaluate_dataset import main
from utils import setup_wandb


def evaluate(input_gold_file, input_prediction_file, input_sentences_path: Optional[str], wandb_run_name: Optional[str]):
    setup_wandb(wandb_run_name is not None, wandb_run_name)

    unlabelled_arg_metrics = main(proposed_path=input_prediction_file, reference_path=input_gold_file, sents_path=input_sentences_path)

    print(f"Result: {unlabelled_arg_metrics}")
    unlabelled_arg_metrics_dict = {
        "prec": unlabelled_arg_metrics.prec(),
        "recall": unlabelled_arg_metrics.recall(),
        "f1": unlabelled_arg_metrics.f1()
    }
    wandb.log(unlabelled_arg_metrics_dict)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("sys_path")
    ap.add_argument("ground_truth_path")
    ap.add_argument("-s","--sentences_path", required=False)
    ap.add_argument("--wandb_run_name", required=False)
    args = ap.parse_args()
    evaluate(args.sys_path, args.ground_truth_path, args.sentences_path, args.wandb_run_name)
