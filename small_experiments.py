#%%
from notebook import *

# Prototype - for listing the valid keyword args (given with default values)

# model_type, epochs = "t5", 30
# wandb_run_name=f"{now()}_{epochs}ep_{model_type}_"
# full_experiment(model_type=model_type, # e.g. "t5", "bart", "iarfmoose/t5-base-question-generator"
#                     train_dataset="joint_qasrl",
#                     qanom_joint_factor=1, # how many times to duplicate qanom training set in joint training
#                     test_dataset="qasrl",
#                     train_epochs=epochs,
#                     batch_size=12,
#                     gradient_accumulation_steps=1,
#                     learning_rate=.00005,
#                     fp16=True,
#                     source_prefix="Generate QAs for <predicate_type> QASRL: ",
#                     preprocess_input_func="input_predicate_marker",
#                     preprocess_output_func="all_random_order",
#                     append_verb_form=True,
#                     predicate_marker_type="generic", # or "pred_type"
#                     use_bilateral_predicate_marker=False,
#                     learn_predicate_type=None, # or "pre" or "post"
#                     limit_train_data=1.0,
#                     logging_steps=500,
#                     eval_steps=500,
#                     save_steps=500,
#                     overwrite_output_dir=True,
#                     wandb_run_name=wandb_run_name,
#                     dir_switch="qasrl-no-order",
#                     description="",
#                     )

# #%%  best joint model so far
for model_type in ["t5", "bart"]:
    epochs=50
    wandb_run_name=f"{now()}_{epochs}ep_{model_type}_qanom_baseline_long"
    full_experiment(model_type=model_type,
                        train_dataset="qanom",
                        test_dataset="qanom",
                        train_epochs=epochs,
                        batch_size=12,
                        source_prefix="Generate: ",
                        preprocess_input_func="input_predicate_marker",
                        use_bilateral_predicate_marker=True,
                        overwrite_output_dir=True,
                        num_beams=3,
                        logging_steps=500,
                        eval_steps=500,
                        save_steps=500,
                        wandb_run_name=wandb_run_name,
                        dir_switch="qanom_long",
                        # qanom_joint_factor=14,
                        description="""basline with short generic prefix, load_best_model=True""",
                        )
    
    # epochs = 60
    # wandb_run_name=f"{now()}_{epochs}ep_{model_type}_qanom_long"
    # full_experiment(model_type=model_type,
    #                     train_dataset="qanom",
    #                     test_dataset="qanom",
    #                     train_epochs=epochs,
    #                     batch_size=12,
    #                     source_prefix="ask <predicate_type>: ",
    #                     preprocess_input_func="input_predicate_marker",
    #                     use_bilateral_predicate_marker=True,
    #                     overwrite_output_dir=True,
    #                     num_beams=3,
    #                     logging_steps=500,
    #                     wandb_run_name=wandb_run_name,
    #                     dir_switch="joint_qanom_short-prefix",
    #                     qanom_joint_factor=14,
    #                     description="""shorter pred-type prefix, bilateral marker, beams=3, qanom_factor=14""",
    #                     )
    
model_type="t5"
epochs=10
wandb_run_name=f"{now()}_{epochs}ep_{model_type}_qanom_exp-baseline"
full_experiment(model_type=model_type,
                    train_dataset="qanom",
                    test_dataset="qanom",
                    train_epochs=epochs,
                    batch_size=12,
                    source_prefix="Generate: ",
                    preprocess_input_func="input_predicate_marker",
                    use_bilateral_predicate_marker=True,
                    overwrite_output_dir=True,
                    num_beams=3,
                    logging_steps=200,
                    eval_steps=200,
                    save_steps=200,
                    wandb_run_name=wandb_run_name,
                    dir_switch="qanom_exp",
                    # qanom_joint_factor=14,
                    description="""experiment: baseline""",
                    )

wandb_run_name=f"{now()}_{epochs}ep_{model_type}_qanom_exp-batch=120"
full_experiment(model_type=model_type,
                    train_dataset="qanom",
                    test_dataset="qanom",
                    train_epochs=epochs,
                    batch_size=12,
                    gradient_accumulation_steps=10,
                    source_prefix="Generate: ",
                    preprocess_input_func="input_predicate_marker",
                    use_bilateral_predicate_marker=True,
                    overwrite_output_dir=True,
                    num_beams=3,
                    logging_steps=200,
                    eval_steps=200,
                    save_steps=200,
                    wandb_run_name=wandb_run_name,
                    dir_switch="qanom_exp",
                    # qanom_joint_factor=14,
                    description="""experiment: batch=120""",
                    )

#%%
# prefix & decoding experiments

# model_type, epochs = "t5", 30
# wandb_run_name=f"{now()}_{epochs}ep_{model_type}_qanom_baseline"
# full_experiment(model_type=model_type,
#                     train_dataset="qanom",
#                     test_dataset="qanom",
#                     train_epochs=epochs,
#                     batch_size=12,
#                     source_prefix="Generate QAs for <predicate_type> QASRL: ",
#                     preprocess_input_func="input_predicate_marker",
#                     overwrite_output_dir=True,
#                     wandb_run_name=wandb_run_name,
#                     dir_switch="qanom_baseline",
#                     description="supposed to be best qanom baseline - trained on qanom, prefix is <predicate-type> dependent, preprocessing_input_func is predicate_marker. ",
#                     )

# wandb_run_name=f"{now()}_{epochs}ep_{model_type}_qanom_new-generic-prefix"
# full_experiment(model_type=model_type,
#                     train_dataset="qanom",
#                     test_dataset="qanom",
#                     train_epochs=epochs,
#                     batch_size=12,
#                     source_prefix="Ask and answer: ",
#                     preprocess_input_func="input_predicate_marker",
#                     overwrite_output_dir=True,
#                     wandb_run_name=wandb_run_name,
#                     dir_switch="qanom_prefix",
#                     description="same as baseline- trained on qanom, preprocessing_input_func is predicate_marker; But with prefix 'ask and answer' ",
#                     )
# wandb_run_name=f"{now()}_{epochs}ep_{model_type}_qanom_short-generic-prefix"
# full_experiment(model_type=model_type,
#                     train_dataset="qanom",
#                     test_dataset="qanom",
#                     train_epochs=epochs,
#                     batch_size=12,
#                     source_prefix="Parse: ",
#                     preprocess_input_func="input_predicate_marker",
#                     overwrite_output_dir=True,
#                     wandb_run_name=wandb_run_name,
#                     dir_switch="qanom_prefix",
#                     description="same as baseline- trained on qanom, preprocessing_input_func is predicate_marker; But with prefix 'Parse' ",
#                     )
# wandb_run_name=f"{now()}_{epochs}ep_{model_type}_qanom_beams=3"
# full_experiment(model_type=model_type,
#                     train_dataset="qanom",
#                     test_dataset="qanom",
#                     train_epochs=epochs,
#                     batch_size=12,
#                     source_prefix="Generate QAs for <predicate_type> QASRL: ",
#                     preprocess_input_func="input_predicate_marker",
#                     overwrite_output_dir=True,
#                     wandb_run_name=wandb_run_name,
#                     dir_switch="qanom_beams",
#                     num_beams=3,
#                     description="trained on qanom, prefix is <predicate-type> dependent, preprocessing_input_func is predicate_marker; ",
#                     )

# wandb_run_name=f"{now()}_{epochs}ep_{model_type}_qanom_beams=10"
# full_experiment(model_type=model_type,
#                     train_dataset="qanom",
#                     test_dataset="qanom",
#                     train_epochs=epochs,
#                     batch_size=12,
#                     source_prefix="Generate QAs for <predicate_type> QASRL: ",
#                     preprocess_input_func="input_predicate_marker",
#                     overwrite_output_dir=True,
#                     wandb_run_name=wandb_run_name,
#                     dir_switch="qanom_beams",
#                     num_beams=10,
#                     description="trained on qanom, prefix is <predicate-type> dependent, preprocessing_input_func is predicate_marker; ",
#                     )



#%% 
# Test effect of ordering QAs by indices

# epochs = 30
# for model_type in ["t5", "bart"]:
    # wandb_run_name=f"{now()}_{epochs}ep_{model_type}_qasrl_no-order"
    # full_experiment(model_type=model_type,
    #                 train_dataset="qasrl",
    #                 test_dataset="qasrl",
    #                 train_epochs=epochs,
    #                 wandb_run_name=wandb_run_name,
    #                 source_prefix="Generate QAs: ",
    #                 dir_switch="qasrl-no-order",
    #                 preprocess_input_func="input_predicate_marker",
    #                 preprocess_output_func="all_random_order",
    #                 description=""
    #                 )
    # print(f"\n\n\n Experiment {wandb_run_name} end: {now()}\n\n\n\n")
    
    # wandb_run_name=f"{now()}_{epochs}ep_{model_type}_qasrl_ordered"
    # full_experiment(model_type=model_type,
    #                 train_dataset="qasrl",
    #                 test_dataset="qasrl",
    #                 train_epochs=epochs,
    #                 wandb_run_name=wandb_run_name,
    #                 source_prefix="Generate QAs: ",
    #                 dir_switch="qasrl-ordered",
    #                 preprocess_input_func="input_predicate_marker",
    #                 preprocess_output_func="all_by_answer_ordering",
    #                 description="should be the new best baseline for qasrl"
    #                 )
    # print(f"\n\n\n Experiment {wandb_run_name} end: {now()}\n\n\n\n")


# qasrl joint - first experiment (t5 prefixes)
# for model_type in ["t5"]:
#     wandb_run_name=f"{now()}_{epochs}ep_{model_type}_joint-qasrl_generic-prefix"
#     full_experiment(model_type=model_type,
#                     train_dataset="joint_qasrl",
#                     test_dataset="qasrl",
#                     train_epochs=epochs,
#                     wandb_run_name=wandb_run_name,
#                     source_prefix="Generate QAs: ",
#                     dir_switch="qasrl-joint",
#                     preprocess_input_func="input_predicate_marker",
#                     preprocess_output_func="all_by_answer_ordering",
#                     description=""
#                     )
#     print(f"\n\n\n Experiment {wandb_run_name} end: {now()}\n\n\n\n")
    
#     wandb_run_name=f"{now()}_{epochs}ep_{model_type}_joint-qasrl_pred-type-prefix"
#     full_experiment(model_type=model_type,
#                     train_dataset="joint_qasrl",
#                     test_dataset="qasrl",
#                     train_epochs=epochs,
#                     wandb_run_name=wandb_run_name,
#                     source_prefix="Generate QAs for <predicate_type> QASRL: ",
#                     dir_switch="qasrl-joint",
#                     preprocess_input_func="input_predicate_marker",
#                     preprocess_output_func="all_by_answer_ordering",
#                     description="should be the new best baseline for joint-qasrl"
#                     )
#     print(f"\n\n\n Experiment {wandb_run_name} end: {now()}\n\n\n\n")

# for model_type in ["t5", "bart"]:

#     wandb_run_name=f"{now()}_{epochs}ep_{model_type}_joint_qanom_predicate-repeat"
#     full_experiment(model_type=model_type,
#                     train_dataset="joint_qanom",
#                     test_dataset="qanom",
#                     train_epochs=epochs,
#                     wandb_run_name=wandb_run_name,
#                     source_prefix="Generate QAs: ",
#                     dir_switch="joint-qanom-pred-repeat",
#                     preprocess_input_func="input_predicate_repeated",
#                     description="using generic prefix, but inserting the predicate_type in the input_preprocessing at beginnning of sequence (but after 'prefix')"
#                     )
#     print(f"\n\n\n Experiment {wandb_run_name} end: {now()}\n\n\n\n")

#     wandb_run_name=f"{now()}_{epochs}ep_{model_type}_joint_qanom_predicate-marker"
#     full_experiment(model_type=model_type,
#                     train_dataset="joint_qanom",
#                     test_dataset="qanom",
#                     train_epochs=epochs,
#                     wandb_run_name=wandb_run_name,
#                     source_prefix="Generate QAs for <predicate_type> QASRL: ",
#                     dir_switch="joint-qanom-pred-marker",
#                     preprocess_input_func="input_predicate_marker",
#                     description="After commenting out pred-type-dependent predicate marker, staying only with marker_generic_predicate."
#                     )
#     print(f"\n\n\n Experiment {wandb_run_name} end: {now()}\n\n\n\n")


    
