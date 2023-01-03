""" A custom Seq2Seq model for qasrl. """

from ast import Pass
from typing import Literal, Optional, List, Dict, Tuple, Callable, Union
from dataclasses_json import config

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Config
)

import torch
import torch.distributed as dist
from torch import nn

from transformers import (
    BeamScorer, 
    LogitsProcessorList, 
    StoppingCriteriaList,
    T5TokenizerFast
)
import transformers
from transformers.generation_utils import (
    BeamSearchOutput,
    BeamSearchDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    GreedySearchEncoderDecoderOutput
)
from transformers.generation_stopping_criteria import validate_stopping_criteria
import warnings
import numpy as np

import itertools       
from constrained_decoding.dfa import DFA 
from constrained_decoding.autoregressive_dfa_constraining import dfa_constrained_generate 
    


class QASRLSeq2SeqModel(T5ForConditionalGeneration):
    BRANCHING_STRATEGIES = (
        "standard_beam_search",
        "at_first_token",
        # "at_subsequences_start",
        # "not_during_answer"
    )
    DEFAULT_BRANCHING_STRATEGY = BRANCHING_STRATEGIES[0]
    
    def register_constraining_dfa(self, dfa=None, dfa_factory=None, multiple_dfas=None, multiple_dfa_factories=None):

        # Init `multiple_dfa_factories` list if doesn't exist
        self.multiple_dfa_factories = vars(self).get("multiple_dfa_factories", [])
        
        # Collect all dfas that would be applied per beam into `self.multiple_dfa_factories`
        multiple_dfas = multiple_dfas or []
        if dfa: multiple_dfas.append(dfa)
        if dfa_factory: self.multiple_dfa_factories.append(dfa_factory)
        for a_dfa in multiple_dfas:
            self.multiple_dfa_factories.append(lambda x: a_dfa)
        self.multiple_dfa_factories.extend(multiple_dfa_factories)
    
    def set_branching_strategy(self, strategy: Optional[str] = None):
        strategy = strategy or QASRLSeq2SeqModel.DEFAULT_BRANCHING_STRATEGY
        assert strategy in QASRLSeq2SeqModel.BRANCHING_STRATEGIES, \
            f"branching strategy must be one of following: {QASRLSeq2SeqModel.BRANCHING_STRATEGIES}"
        self.branching_strategy = strategy 
    
    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        encoder_input_ids: Optional[torch.LongTensor] = None, # @dfa
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        This is an adapted beam_search method for contraining the search according to a custom Deterministic Finite Automaton (DFA).
        The DFA should be provided as a method attribute, by setting `dfa_constrained_beam_search.dfa = DFA(...)`.
        The original method is copied from transformers.generation_utils.GenerationMixins.beam_search(...).
            Github Ref: https://github.com/huggingface/transformers/blob/05fa1a7ac17bb7aa07b9e0c1e138ecb31a28bbfe/src/transformers/generation_utils.py#L1730
        Additions are denoted with a preceding "# @dfa:" comment.  
        ---------------
        Generates sequences for models with a language modeling head using beam search decoding.

        Parameters:

            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are
                constructed, stored and sorted during generation. For more information, the documentation of
                [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from
                [`LogitsProcessor`] used to modify the prediction scores of the language modeling
                head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from
                [`StoppingCriteria`] used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of
                generated tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to *False*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to *False*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to *False*):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to *False*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If
                model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation_utilsBeamSearchDecoderOnlyOutput`],
            [`~generation_utils.BeamSearchEncoderDecoderOutput`] or obj:*torch.LongTensor*: A
            `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.BeamSearchDecoderOnlyOutput`] if
            `model.config.is_encoder_decoder=False` and `return_dict_in_generate=True` or a
            [`~generation_utils.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import (
        ...    AutoTokenizer,
        ...    AutoModelForSeq2SeqLM,
        ...    LogitsProcessorList,
        ...    MinLengthLogitsProcessor,
        ...    BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList([
        ...     MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ... ])

        >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        ```"""
        # @dfa: get DFA from function attribute 'dfa' or `dfa_factory` if exists
        multiple_dfa_factories = vars(self).get("multiple_dfa_factories", [])
        is_dfa = bool(multiple_dfa_factories)
        
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        
        # @dfa: Init an array of iterators (holding current_states) for running the dfa efficiently per beam
        # To support applying multiple dfas concurrently on every beam, `current_state_iterators` would have 
        #  the shape (`n_dfas_per_beam` X `batch_beam_size`) 
        n_dfas_per_beam = len(multiple_dfa_factories)
        if multiple_dfa_factories: # === if is_dfa:    
            current_state_iterators = []
            for dfa_factory in multiple_dfa_factories:
                dfas = [[dfa_factory(input_seq)] * num_beams 
                        for input_seq in encoder_input_ids] 
                dfas = list(itertools.chain(*dfas)) # length batch_beam_size
                current_state_iterators.append([dfa.iterator() for dfa in dfas])
            
        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            # @dfa: in every iteration we are directly constraining next tokens to dfa transitions
            if is_dfa:
                # to constrain to conjunction of all dfas (per beam), we will apply -inf masks iteratively
                for per_dfa_current_state_iterators in current_state_iterators:
                    mask = torch.ones_like(next_token_scores) # (batch_beam_size, vocab_size); 1 will denote forbidden tokens
                    for i, current_state_iterator in enumerate(per_dfa_current_state_iterators):
                        allowed_word_ids = list(current_state_iterator.get_allowed_transitions().keys())
                        if DFA.WILDCARD in allowed_word_ids: 
                        # allow all words
                            mask[i] = 0
                        else:
                        # allow only those words
                            mask[i, allowed_word_ids] = 0 # put zeros in allowed token ids       
                    next_token_scores = next_token_scores.masked_fill(mask.bool(), -float("inf"))
            
            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # Ayal Comment: The following lines represent "branching", and should be refactored for partial-beam-search strategies 

            #TODO currently implementing global branching strategies only - same "is_branching" decision for all the batch
            
            # Set `branch_method` for this iteration (i.e. next token), according to `self.branching_strategy`:
            #  "diverse" --- take diverse top num_beams of the best beam; suitable for first token
            #  "no_branching" --- every beam will continue independantly via greedy decoding
            #  "standard" --- regular beam search algorithm, taking top num_beams for union of beams    
            
            if self.branching_strategy == "standard_beam_search":
                branch_method = "standard"
            
            elif self.branching_strategy == "at_first_token":
                if cur_len <= 1: 
                    branch_method = "standard" # "diverse"
                else:
                    branch_method = "no_branching"
            else:
                raise NotImplementedError()        
                    
            
            vocab_size = next_token_scores.shape[-1]
            if branch_method == "standard":
                # Original beam_search code:
                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                )

                next_indices = (next_tokens / vocab_size).long()
                next_tokens = next_tokens % vocab_size

            elif branch_method == "no_branching":
                # DO NOT BRANCH OUT - apply greedy decoding for each beam separately
                next_token_scores = next_token_scores.view(batch_size, num_beams, vocab_size)
                next_token_scores, next_tokens = next_token_scores.topk(
                    2, dim=2, largest=True, sorted=True
                )
                next_token_scores = next_token_scores.view(batch_size, num_beams*2)
                next_tokens = next_tokens.view(batch_size, num_beams*2)
                # next_indices = torch.arange(0,num_beams).repeat(batch_size,1)
                next_indices = torch.arange(0,num_beams).repeat(2,1).T.reshape(-1).repeat(batch_size,1)
            
            elif branch_method == "diverse":
                # Take `num_beams` different tokens by applying top_k on best beam 
                pass
                # TODO prepare: 
                #   `next_token_scores` (batch_size, num_beams*2)
                #   `next_tokens` (batch_size, num_beams*2)
                #   `next_indices` (batch_size, num_beams*2) - indices of beams from from which to take prev token              
            
            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"] # (batch_beam_size == batch_size*num_beams,)
            beam_idx = beam_outputs["next_beam_indices"] # (batch_beam_size == batch_size*num_beams,)

            # @dfa: update automatons state
            if is_dfa:
                # copy object when indexed (to avoid references to same iter object) 
                current_state_iterators = [ [current_state_iterators[dfa_i][beam_i].copy() 
                                            for beam_i in beam_idx.cpu()]  # take the previous states of the selected beams 
                                        for dfa_i in range(n_dfas_per_beam)]
                for per_dfa_current_state_iterators in current_state_iterators:
                    for i, dfa_iter in enumerate(per_dfa_current_state_iterators):
                        # step is applied internally in iterators
                        success, new_state = dfa_iter.step(beam_next_tokens[i].item())             
                    
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]     
    
    
    # Other Seq2seq Utilities, not related to constrained beam search
    
    def get_sequence_score(self, 
                           encoder_input_ids: torch.LongTensor, 
                           output_seq_ids: torch.LongTensor,
                           agg: Callable = torch.mean,
                           **model_kwargs) -> float:
        """ Return the "confidnece" score of a given output sequence, as the minimum of token-wise posterior probailities. 
        See "Modeling Confidence in Sequence-to-Sequence Models" (https://aclanthology.org/W19-8671.pdf) for definition (\S 2.2).
        Assuming an encoder-decoder model. 

        Args:
            encoder_input_ids (LongTensor): the input sequence (context) - 1D
            output_seq_ids (LongTensor): the output sequence for which we want to get the model's score - 1D, no trailing <pad>
            agg (Callable): aggregation function for posterior probabilites
            **model_kwargs: additional kwargs for model's forward method.
        """
        seq_probs = self.get_sequence_posteriors(encoder_input_ids, output_seq_ids, **model_kwargs)
        agg_probs = agg(seq_probs)
        return agg_probs
        
    def get_sequence_posteriors(self, 
                                encoder_input_ids: torch.LongTensor,
                                output_seq_ids: torch.LongTensor,
                                **model_kwargs) -> torch.Tensor:
        """ Return the token-wise posterior probabilities of a given output sequence (). 
        See "Modeling Confidence in Sequence-to-Sequence Models" (https://aclanthology.org/W19-8671.pdf) for definition (\S 2.2).
        Assuming an encoder-decoder model. 

        Args:
            encoder_input_ids (LongTensor): the input sequence (context) - 1D
            output_seq_ids (LongTensor): the output sequence for which we want to get the model's score - 1D, no trailing <pad>
            **model_kwargs: additional kwargs for model's forward method.
        """
        assert self.config.is_encoder_decoder, "This method is implemented only for encoder-decoder models"
        bos_token_id = self.config.bos_token_id
        scores = ()
        
        if encoder_input_ids.dim() == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        
        # put the input sequence (`encoder_input_ids`) into model_kwargs, as done in `generate` code: 
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(encoder_input_ids, bos_token_id, model_kwargs)
        if "encoder_outputs" not in model_kwargs:
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )
            
        batch_size = inputs_tensor.shape[0] # 1!
    
        # Prepare `decoder_input_ids` which will be used for auto-regressive generation
        # decoder_input_ids = self._prepare_decoder_input_ids_for_generation(
        #     batch_size,
        #     decoder_start_token_id=None,
        #     bos_token_id=bos_token_id,
        #     model_kwargs=model_kwargs,
        # )    
        decoder_input_ids = torch.cat([torch.tensor([0]), output_seq_ids]).view(1,-1) 
        
        # auto-regressive loop   
        for i in range(1,decoder_input_ids.size(1)): 

            # prepare model inputs (including both encoder-input and current decoder-input info)
            model_inputs = self.prepare_inputs_for_generation(decoder_input_ids[:,:i], **model_kwargs)
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True
            )            
            next_token_logits = outputs.logits[:, -1, :]
            scores += (next_token_logits,)
            
            # update generated ids, model inputs, and length for next step
            # decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.view(1,1)], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )  
            # don't include final <pad> tokens          
            if decoder_input_ids[0,i] == 1:
                break

        
        # If sequences_scores computed as, \sum_{t} \log p(y_{t} | x, y_{t<}) --- let's implement it by myself and extract it from `scores`
        scores = torch.stack(scores).squeeze(1)
        probabilities = torch.softmax(scores, dim=-1)
        seq_probs = torch.tensor([prob[tok_id] 
                                  for prob, tok_id in zip(probabilities, output_seq_ids)])
        return seq_probs

        
    
    
if __name__ == "__main__":
    # debug & test `get_sequence_score`
    model_name_or_path = f"/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/qanom/linearization/permutate_sample_num_of_qas"
    # model_name_or_path = f"/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/qanom/linearization/all_by_answer_ordering"
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = QASRLSeq2SeqModel.from_pretrained(model_name_or_path,
        config=config,
    )
    input_str = "parse: A device known as the Mosquito, set to be trialed in New South Wales ( NSW ), Australia by RailCorp in a bid to deter vandals from areas frequently the<extra_id_10> target<extra_id_10> of graffiti, is attracting criticism.<extra_id_1> target"
    input_ids = torch.tensor(tokenizer(input_str).input_ids).unsqueeze(0)
    generated = model.generate(input_ids, max_length=120, return_dict_in_generate=True)
    def print_seq(seq):
        print(tokenizer.decode(seq))
    print_seq(generated["sequences"][0])
    output_str= "what was _ targeted _ _ _?<extra_id_7> graffiti<extra_id_9> where was something targeted _ _ _?<extra_id_7> areas<extra_id_9> who was something targeted _ for _?<extra_id_7> RailCorp<extra_id_3> New South Wales<extra_id_3> Australia</s>"
    output_ids = torch.tensor(tokenizer(output_str).input_ids)
    score = model.get_sequence_score(generated["sequences"], output_ids)
    print(score)