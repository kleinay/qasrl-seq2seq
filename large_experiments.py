# Design large-scale experiments (leanring curve analysis, mixed training, etc.)
#%%
from notebook import *


#%%

# learning rate

for model_type in ["t5", "iarfmoose/t5-base-question-generator"]:
    epochs=30
    wandb_run_name=f"{now()}_{epochs}ep_{model_type}_qanom"
    full_experiment(model_type=model_type,
                        train_dataset="qanom",
                        test_dataset="qanom",
                        train_epochs=epochs,
                        batch_size=2,
                        source_prefix="Generate: ",
                        preprocess_input_func="input_predicate_marker",
                        use_bilateral_predicate_marker=True,
                        overwrite_output_dir=True,
                        num_beams=1,
                        logging_steps=500,
                        eval_steps=500,
                        save_steps=500,
                        wandb_run_name=wandb_run_name,
                        dir_switch="qanom_quesgen",
                        # qanom_joint_factor=14,
                        description="""basline vs. quesgen""",
                        )
    wandb_run_name=f"{now()}_{epochs}ep_{model_type}_qanom_lr=.0001"
    full_experiment(model_type=model_type,
                        train_dataset="qanom",
                        test_dataset="qanom",
                        train_epochs=epochs,
                        batch_size=2,
                        source_prefix="Generate: ",
                        preprocess_input_func="input_predicate_marker",
                        use_bilateral_predicate_marker=True,
                        overwrite_output_dir=True,
                        num_beams=1,
                        logging_steps=500,
                        eval_steps=500,
                        save_steps=500,
                        wandb_run_name=wandb_run_name,
                        dir_switch="qanom_quesgen",
                        # qanom_joint_factor=14,
                        learning_rate=.0001,
                        description="""basline vs. quesgen, lr=.0001""",
                        )
    
    wandb_run_name=f"{now()}_{epochs}ep_{model_type}_qanom_lr=.001"
    full_experiment(model_type=model_type,
                        train_dataset="qanom",
                        test_dataset="qanom",
                        train_epochs=epochs,
                        batch_size=2,
                        source_prefix="Generate: ",
                        preprocess_input_func="input_predicate_marker",
                        use_bilateral_predicate_marker=True,
                        overwrite_output_dir=True,
                        num_beams=1,
                        logging_steps=500,
                        eval_steps=500,
                        save_steps=500,
                        wandb_run_name=wandb_run_name,
                        dir_switch="qanom_quesgen",
                        # qanom_joint_factor=14,
                        learning_rate=.001,
                        description="""basline vs. quesgen, lr=.001""",
                        )





# model_type="t5"
# train_dataset="qanom"
# test_dataset="qanom"
# model_dir = t5_model_dir if model_type == "t5" else bart_model_dir
# qasrl_train_dataset = "qanom" if train_dataset == "qanom" else "2018"
# qasrl_test_dataset = "qanom" if test_dataset == "qanom" else "2020"

# kwargs=dict(train_epochs=10, wandb_run_name=f"{now()}_debug_with_joint_and_em")

# run = train(model_type, qasrl_train_dataset, **kwargs)
# predict(model_type, qasrl_test_dataset, run)
# decode_into_qasrl(model_dir, qasrl_test_dataset)
# unlabelled_arg, labelled_arg, unlabelled_role = evaluate(model_dir, qasrl_test_dataset)


#%% 
# Joint Training - start with QASRL, then finetune on qanom

# model_type="t5"
# wandb_run_name=f"{now()}_debug"   
# full_experiment(model_type=model_type,
#                 train_dataset="qanom",
#                 test_dataset="qanom",
#                 train_epochs=1,
#                 wandb_run_name=wandb_run_name,
#                 )
# full_experiment(model_type=model_type,
#                 train_dataset="qanom",
#                 test_dataset="qanom",
#                 overwrite_output_dir=False,
#                 train_epochs=25,
#                 wandb_run_name=wandb_run_name + "_qanom-25",
#                 )


# model_type="bart"
# wandb_run_name=f"{now()}_bart_mixed_qasrl-5_qanom-15"
# full_experiment(model_type=model_type,
#                 train_dataset="qasrl",
#                 test_dataset="qanom",
#                 train_epochs=5,
#                 wandb_run_name=wandb_run_name,
#                 )
# full_experiment(model_type=model_type,
#                 train_dataset="qanom",
#                 test_dataset="qanom",
#                 overwrite_output_dir=False,
#                 train_epochs=15,
#                 wandb_run_name=wandb_run_name,
#                 )

#%%
# Test performance after different training epochs




#%%
# Accumulative learning-curve estimation (training set size) 

# model_type="bart"
# data = "qasrl"
# epochs = 10
# fn = f"learning_curve_{data}_{model_type}_{epochs}-epochs_sample-size-to-arg-f1.json"
# arg_f1 = json.load(open(fn))
# for train_sample_size in [45000, 60000, 100000]:  # 1000 samples and less result in 0.0 performance 
#     wandb_run_name=f"{now()}_{data}_{model_type}_{train_sample_size}-samples_learning_curve" 
#     metrics = full_experiment(model_type=model_type,
#                     train_dataset=data,
#                     test_dataset=data,
#                     train_epochs=epochs,
#                     wandb_run_name=wandb_run_name,
#                     max_train_samples=train_sample_size,
#                     dir_switch="data_learning_curve"
#                     )
#     arg_f1[train_sample_size] = metrics[0].f1()
#     print(f"\n\n\n\n\n !!!!!!!! Performance for training on {train_sample_size} samples:  {metrics[0]}  !!!!!!!!!!  \n\n\n\n\n")
# json.dump(arg_f1, open(fn, "w"))
