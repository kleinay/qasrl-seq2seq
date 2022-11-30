#%%
from torch import dropout
from notebook import *

# Prototype - for listing the valid keyword args (given with default values)

# model_type, epochs = "t5", 30
# wandb_run_name=f"{now()}_{epochs}ep_{model_type}_"
# full_experiment(model_type=model_type, # e.g. "t5", "bart", "iarfmoose/t5-base-question-generator"
#                     train_dataset="joint_qasrl",
#                     do_eval_on="validation", # or "test" # whether to do the final evaluation ("do eval") on validation set or on test set
#                     qanom_joint_factor=1, # how many times to duplicate qanom training set in joint training
#                     train_epochs=epochs,
#                     batch_size=12,
#                     gradient_accumulation_steps=1,
#                     learning_rate=.00005,
#                     fp16=True,
#                     source_prefix="Generate QAs for <predicate_type> QASRL: ",
#                     preprocess_input_func="input_predicate_marker",
#                     preprocess_output_func="all_by_answer_ordering", # "permutate_sample_fixed", "permutate_sample_num_of_qas", "permutate_all", "all_shuffled", "all_random_order", "all_by_answer_ordering", "all_by_role_ordering"
#                     append_verb_form=True,
#                     predicate_marker_type="generic", # or "pred_type"
#                     use_bilateral_predicate_marker=False,
#                     load_best_model_at_end=True,
#                     metric_for_best_model="rouge1", # or "eval_loss" / "rougeL"
#                     learn_predicate_type=None, # or "pre" or "post"
#                     limit_train_data=1.0,
#                     limit_eval_data=1.0,
#                     logging_steps=500,
#                     eval_steps=500,
#                     save_steps=500,
#                     overwrite_output_dir=True,
#                     wandb_run_name=wandb_run_name,
#                     dir_switch="qasrl-no-order",
#                     description="",
#                     )

## Actual Experiments:

# train best single-domain models for cross-domain experiments

best_hparams = {
    "TQA": dict(
        gradient_accumulation_steps=14,
        learning_rate=0.005, 
        dropout_rate=0.15,
    ),
    # "wikinews": dict(
    #     gradient_accumulation_steps=8,
    #     learning_rate=0.001, 
    #     dropout_rate=0.10,
    # ),
    # "wikipedia": dict(
    #     gradient_accumulation_steps=14,
    #     learning_rate=0.005, 
    #     dropout_rate=0.10,
    # )   
}

for domain, hparams in best_hparams.items():
    wandb_run_name=f"{now()} qasrl (large) trained on {domain} (eval on wikipedia test)"
    full_experiment(model_type="t5",
                    train_dataset="qasrl",
                    training_domains=domain,
                    validation_domains="wikipedia",
                    test_domains="wikipedia",
                    do_eval_on="test",
                    train_epochs=20,
                    batch_size=12,
                    seed=44,
                    source_prefix="parse: ",
                    preprocess_input_func="input_predicate_marker",
                    preprocess_output_func="all_by_answer_ordering",
                    use_bilateral_predicate_marker=True,
                    overwrite_output_dir=True,
                    num_beams=5,
                    logging_steps=500,
                    eval_steps=500,
                    save_steps=500,
                    wandb_run_name=wandb_run_name,
                    # max_train_samples=15079,
                    dir_switch=f"qasrl/cross_domain/best/{domain}",
                    description=f"optimal large qasrl {domain} model, based on grid sweep",
                    **hparams
                    )

# for output_linearization in ["permutate_sample_fixed", "permutate_sample_num_of_qas", 
#                              "all_shuffled", "all_by_answer_ordering", 
#                              "permutate_all"]:
#     wandb_run_name=f"{now()}_5ep_joint_{output_linearization}"
#     full_experiment(model_type="t5",
#                 train_dataset="joint_qanom",
#                 train_epochs=5,
#                 batch_size=12,
#                 gradient_accumulation_steps=14,
#                 learning_rate=0.001,
#                 dropout_rate=0.1,
#                 seed=44,
#                 source_prefix="parse: ",
#                 preprocess_input_func="input_predicate_marker",
#                 preprocess_output_func=output_linearization, # "permutate_sample_fixed", "permutate_sample_num_of_qas", "permutate_all", "all_shuffled", "all_random_order", "all_by_answer_ordering"
#                 use_bilateral_predicate_marker=True,
#                 overwrite_output_dir=True,
#                 num_beams=3,
#                 logging_steps=500,
#                 eval_steps=500,
#                 save_steps=500,
#                 wandb_run_name=wandb_run_name,
#                 qanom_joint_factor=14,

#                 dir_switch=f"linearization/{output_linearization}",
#                 description=f"best joint with output_linearization={output_linearization}, with best hyperparamters for 'by_answer_order' (5ep, accum=14, lr=.001) ",
#                 )

# for output_linearization in ["permutate_sample_num_of_qas", "permutate_sample_fixed", 
#                              "all_by_answer_ordering", "all_shuffled", "permutate_all"]: # "all_shuffled", "all_by_answer_ordering",
# for output_linearization in ["all_by_role_ordering"]: 
# output_linearization = "all_by_role_ordering" 
    # wandb_run_name=f"{now()} joint, {output_linearization} - on qanom test"
    # full_experiment(model_type="t5",
    #             train_dataset="joint_qanom",
    #             train_epochs=20,
    #             batch_size=12,
    #             gradient_accumulation_steps=14,
    #             learning_rate=0.001,
    #             dropout_rate=0.1,
    #             seed=44,
    #             do_eval_on="test",
    #             source_prefix="parse: ",
    #             preprocess_input_func="input_predicate_marker",
    #             preprocess_output_func=output_linearization, # "permutate_sample_fixed", "permutate_sample_num_of_qas", "permutate_all", "all_shuffled", "all_random_order", "all_by_answer_ordering"
    #             use_bilateral_predicate_marker=True,
    #             overwrite_output_dir=True,
    #             num_beams=5,
    #             logging_steps=500,
    #             eval_steps=500,
    #             save_steps=500,
    #             wandb_run_name=wandb_run_name,
    #             # dir_switch=f"debug/{output_linearization}",
    #             dir_switch=f"lin-{output_linearization}",
    #             # limit_train_data=0.07,
    #             qanom_joint_factor=14,
    #             description=f"joint with output_linearization={output_linearization}",
    #             )
    # model_dir=f"/home/nlp/kleinay/tmp/t5-tst-summarization/joint_qanom/lin-{output_linearization}"              
    # load_and_evaluate(model_dir,
    #                 test_dataset = "qanom",
    #                 output_dir=None, 
    #                 wandb_run_name=f"evaluate joint {output_linearization} on qanom test",
    #                 # wandb_run_name=f"debug evaluate",
    #                 do_eval_on_dev=False,
    #                 evaluation_protocol="qanom",
    #                 constrain_generation=False,
    #                 # limit_eval_data=0.05,
    #                 batch_size=12,
    #                 num_beams=5,
    #                 )
    # load_and_evaluate(model_dir,
    #                 test_dataset = "qasrl",
    #                 output_dir=None, 
    #                 wandb_run_name=f"evaluate joint {output_linearization} on qasrl test",
    #                 # wandb_run_name=f"debug evaluate",
    #                 do_eval_on_dev=False,
    #                 evaluation_protocol="qanom",
    #                 constrain_generation=False,
    #                 # limit_eval_data=0.05,
    #                 batch_size=12,
    #                 num_beams=5,
    #                 )

# optimal qasrl-baseline based on sweep:
# wandb_run_name=f"{now()}_t5_baseline_qasrl_small"
# full_experiment(model_type="t5",
#                 train_dataset="qasrl",
#                 train_epochs=20,
#                 batch_size=12,
#                 gradient_accumulation_steps=8,
#                 learning_rate=0.005, # ACTUALLY 0.0062
#                 dropout_rate=0.1,
#                 seed=44,
#                 source_prefix="parse: ",
#                 preprocess_input_func="input_predicate_marker",
#                 use_bilateral_predicate_marker=True,
#                 overwrite_output_dir=True,
#                 num_beams=5,
#                 logging_steps=500,
#                 eval_steps=500,
#                 save_steps=500,
#                 wandb_run_name=wandb_run_name,
#                 dir_switch="small_train",
#                 limit_train_data=0.07,
#                 description="qasrl baseline trained on 0.07 of the data, approx. as qanom data",
#                 )


# #%%  best joint model so far
# for learn_predicate_type in [None, "pre", "post"]:
# model_type = "t5-base"
# epochs=20
# wandb_run_name=f"{now()}_{epochs}ep_{model_type}_joint-qanom answer-order dev"
# full_experiment(model_type=model_type,
#                     train_dataset="joint_qanom",
#                     train_epochs=epochs,
#                     batch_size=4,
#                     gradient_checkpointing=True,
#                     # learn_predicate_type=learn_predicate_type,
#                     gradient_accumulation_steps=42,
#                     learning_rate=0.001,
#                     dropout_rate=0.1,
#                     seed=44,
#                     append_verb_form=True,
#                     source_prefix="parse: ",
#                     preprocess_input_func="input_predicate_marker",
#                     use_bilateral_predicate_marker=True,
#                     overwrite_output_dir=True,
#                     num_beams=3,
#                     logging_steps=500,
#                     eval_steps=500,
#                     save_steps=500,
#                     wandb_run_name=wandb_run_name,
#                     preprocess_output_func="all_by_answer_ordering", # "permutate_sample_fixed", "permutate_sample_num_of_qas", "permutate_all", "all_shuffled", "all_random_order", "all_by_answer_ordering"
#                     do_eval_on="validation",
#                     # dir_switch="j",
#                     qanom_joint_factor=14,
#                     description="""t5-base using best joint of t5-small hparams, answer-ordering""",
#                     )
    
  

""" 
*** Saved Configs ***
 best QASRL baseline configuration (sweep baseline finer1):
 
wandb_run_name=f"{now()}_t5_baseline_qasrl"
full_experiment(model_type="t5",
                train_dataset="qasrl",
                train_epochs=5,
                batch_size=12,
                gradient_accumulation_steps=8,
                learning_rate=0.005, # ACTUALLY 0.0062
                dropout_rate=0.1,
                seed=44,
                source_prefix="parse: ",
                preprocess_input_func="input_predicate_marker",
                use_bilateral_predicate_marker=True,
                overwrite_output_dir=True,
                num_beams=5,
                logging_steps=500,
                eval_steps=500,
                save_steps=500,
                wandb_run_name=wandb_run_name,
                dir_switch="qasrl/baseline",
                description="optimal qasrl baseline config based on finer sweep1",
                )


 best qanom baseline configuration (sweep baseline finer1):

full_experiment(model_type="t5",
                train_dataset="qanom",
                train_epochs=30,
                batch_size=12,
                gradient_accumulation_steps=8,
                learning_rate=0.001,
                dropout_rate=0.15,
                seed=44,
                source_prefix="parse: ",
                preprocess_input_func="input_predicate_marker",
                use_bilateral_predicate_marker=True,
                overwrite_output_dir=True,
                num_beams=3,
                logging_steps=500,
                eval_steps=500,
                save_steps=500,
                wandb_run_name=wandb_run_name,
                dir_switch="baseline",
                description="optimal qanom baseline config from finer sweep1",
                )


 best joint configuration (sweep joint finer1):

full_experiment(model_type="t5",
                train_dataset="joint_qanom",
                train_epochs=20,
                batch_size=12,
                gradient_accumulation_steps=14,
                learning_rate=0.001,
                dropout_rate=0.1,
                seed=44,
                source_prefix="parse: ",
                preprocess_input_func="input_predicate_marker",
                use_bilateral_predicate_marker=True,
                overwrite_output_dir=True,
                num_beams=5,
                logging_steps=500,
                eval_steps=500,
                save_steps=500,
                wandb_run_name=wandb_run_name,
                dir_switch="joint_optimal",
                qanom_joint_factor=14,
                description="optimal joint config from finer sweep1, mainly for qanom",
                )
                


"""
