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
#                     preprocess_output_func="all_by_answer_ordering", # "permutate_sample_fixed", "permutate_sample_num_of_qas", "permutate_all", "all_shuffled", "all_random_order", "all_by_answer_ordering"
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

# #%%  best joint model so far
for model_type in ["t5"]:
    epochs=20
    wandb_run_name=f"{now()}_{epochs}ep_{model_type}_joint_qanom"
    full_experiment(model_type=model_type,
                        train_dataset="joint_qanom",
                        train_epochs=epochs,
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
                        description="""optimal joint config from sweep, mainly for qanom""",
                        )
    # wandb_run_name=f"{now()}_{epochs}ep_{model_type}_qanom_joint"
    # full_experiment(model_type=model_type,
    #                     train_dataset="joint_qanom",
    #                     train_epochs=epochs,
    #                     batch_size=12,
    #                     gradient_accumulation_steps=8,
    #                     learning_rate=0.001,
    #                     dropout_rate=0.15,
    #                     # metric_for_best_model="eval_rouge1",
    #                     source_prefix="parse <predicate_type>: ",
    #                     preprocess_input_func="input_predicate_marker",
    #                     use_bilateral_predicate_marker=True,
    #                     overwrite_output_dir=True,
    #                     num_beams=3,
    #                     append_verb_form=True,
    #                     logging_steps=500,
    #                     eval_steps=500,
    #                     save_steps=500,
    #                     wandb_run_name=wandb_run_name,
    #                     dir_switch="qanom_joint",
    #                     qanom_joint_factor=14,
    #                     description="""joint after refactoring evaluations and hyper-parameter tuning""",
    #                     # limit_train_data=0.01,
    #                     # limit_eval_data=0.01,
    #                     )
  

""" 
*** Saved Configs ***

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
