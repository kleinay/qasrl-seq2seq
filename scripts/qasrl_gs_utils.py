from sys import argv
from typing import Optional
import pandas as pd


def combine_files_tag_with_sentences(tags_file: str, sentences_file: str, combined_output_file: Optional[str] = None):
    """
    Transformers expect one file and there are currently two files. This method combines them
    """

    tags_df = pd.read_csv(tags_file)
    sentences_df = pd.read_csv(sentences_file)
    combined_df = pd.merge(tags_df, sentences_df, on="qasrl_id")
    combined_df.to_csv(combined_output_file, index=False)


if __name__ == "__main__":
    combine_files_tag_with_sentences(argv[1], argv[2], argv[3])
