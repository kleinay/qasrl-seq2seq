import numpy as np
import json
import torch
import utils


class MetaBeam():
    def __init__(self, data_args, tokenizer, special_tokens_constants, clean_seq_func) -> None:
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.special_tokens_constants = special_tokens_constants
        self.clean_seq_func = clean_seq_func
        
        self.sep_id = tokenizer.convert_tokens_to_ids(special_tokens_constants.separator_output_pairs)
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        self.num_return_sequences = data_args.num_beams or 1
        
        # information collecting data-structures 
        self.records = []  # TODO for research

    def save(self, out_filename: str):
        with open(out_filename, "w") as fout:
            json.dump(self.records, fout)
            
    def collect_info(self, generated, input_sequences):
        " Get one batch of inference. Scrape subsequences information. "
        sequences = generated["sequences"]  # (eval_batch_size * num_return_sequences, seq_length)
        scores = torch.stack(generated["scores"]) # (seq_length (-1 ?), eval_batch_size * num_return_sequences, vocab_size)
        sequences_scores = generated["sequences_scores"] # (eval_batch_size * num_return_sequences)
        # define sizes
        n_seqs_in_batch, seq_length = sequences.shape
        eval_batch_size = input_sequences.shape[0]
        # align dimensions
        sequences = sequences.reshape(eval_batch_size, self.num_return_sequences, seq_length)
        sequences_scores = sequences_scores.reshape(eval_batch_size, self.num_return_sequences)
        # Debug Note: sometimes the sequence-length axis may have a mismatch between `scores`
        #  and `sequences` - such that `scores` is provided with a sequence shorter by 1 than `sequences`.
        # The reason mentioned here (https://discuss.huggingface.co/t/scores-in-generate/3450) is that
        #  "the first token, the `decoder_start_token_id` is not generated, meaning that no scores can be calculated".   
        # But there are some batches where len(scores) == seq_length (from `sequences.shape[-1]`), 
        #  and I can't understand why. TODO understand the meaning of this phenomena. 
        # Note: From looking at examples, scores[0] is always full with -100000000 for non-first-beam sequences (but regular for first beams).
        # Meantime if where there is a mismatch, remove last token of all `sequences` (minimal harm- removes <EOS> of longest sequence in batch)
        if len(scores) == seq_length-1:
            # raise ValueError()
            sequences = sequences[:,:,:-1]
            seq_length = seq_length-1
        vocab_size = scores.shape[2]
        scores = scores.transpose(0,1)
        scores = scores.reshape(eval_batch_size, self.num_return_sequences, seq_length, vocab_size)
        
        # Use `sequences` to index the selected-tokens scores from `scores`
        #  and save them to record       
        bch_token_scores = []     
        for bch_idx in range(eval_batch_size):
            ins_token_scores = []
            for beam_idx in range(self.num_return_sequences):
                seq_token_scores = scores[bch_idx,beam_idx][range(seq_length), sequences[bch_idx,beam_idx]]    
                ins_token_scores.append(seq_token_scores)
            ins_token_scores = torch.stack(ins_token_scores)
            bch_token_scores.append(ins_token_scores)
        bch_token_scores = torch.stack(bch_token_scores) # (eval_batch_size, num_return_sequences, seq_length)
        
        # Split each sequence to subsequences (QAs) and compute joint-probability (sum-of-scores) for each
        for bch_idx, (ins_token_scores, ins_sequences) in enumerate(zip(bch_token_scores, sequences)):
            ins_subsequences, ins_subseq_tok_scores, ins_subseq_scores = [], [], []
            # mainly for debug, research and investigation:
            ins_decoded_qas = []
            ins_subseq_mean_scores = []
            ins_subseq_idxs = []
            # prepare information of each beam seprately
            for beam_token_scores, beam_sequence in zip(ins_token_scores, ins_sequences):
                sep_indices = utils.all_indices(beam_sequence, self.sep_id)
                subseqs_indices = utils.split_by_indices(range(seq_length), sep_indices)
                subsequences_info = [   [(beam_sequence[idx].cpu().item(), beam_token_scores[idx].cpu().item(), idx) 
                                            for idx in subseq_indices 
                                            if beam_sequence[idx] not in [self.pad_id, self.eos_id] ]
                                        for subseq_indices in subseqs_indices]
                # each of the three list-of-lists have the same "shape" and correspond to subsequences information
                subsequences, subseqs_tok_scores, subseqs_indices = zip(*[zip(*subinf) for subinf in subsequences_info])
                subseqs_sum_scores = [sum(subseq_tok_scores) for subseq_tok_scores in subseqs_tok_scores] 
                subseqs_mean_scores = [np.mean(subseq_tok_scores) for subseq_tok_scores in subseqs_tok_scores] 
                # save to aggregate over all beams of instances 
                ins_subsequences.append(subsequences)
                ins_subseq_tok_scores.append(subseqs_tok_scores)
                ins_subseq_scores.append(subseqs_sum_scores)    # the total score of a subsequence (i.e. a QA)
                ins_subseq_mean_scores.append(subseqs_mean_scores)    # the total score of a subsequence (i.e. a QA)
                ins_subseq_idxs.append(subseqs_indices)    # the total score of a subsequence (i.e. a QA)
                ins_decoded_qas.append([self.tokenizer.decode(sub, skip_special_tokens=False) 
                                        for sub in subsequences])
            
            # collect QA infos 
            input_seq: str = self.clean_seq_func(self.tokenizer.decode(input_sequences[bch_idx]))
            record = {"input": input_seq,
                        "sequence-scores": sequences_scores[bch_idx].tolist(),
                        "subsequences": ins_subsequences,
                        "subsequences-tok-idxs": ins_subseq_idxs,
                        "subsequences-tok-scores": ins_subseq_tok_scores,
                        "subsequences-scores": ins_subseq_scores,
                        "subsequences-mean-scores": ins_subseq_mean_scores,
                        "QAs": ins_decoded_qas
                        }
            self.records.append(record)