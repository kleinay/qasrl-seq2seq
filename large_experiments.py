# Design large-scale experiments (leanring curve analysis, mixed training, etc.)
#%%
from notebook import *

#%%
# Simple experiments

# wandb_run_name=f"{now()}_40ep_t5_qasrl"
# full_experiment(model_type="t5",
#                 train_dataset="qasrl",
#                 test_dataset="qasrl",
# #                 overwrite_output_dir=True, # if False, continue with last model in model_dir; then need to set `train_epochs` to be the cummulative num of desired epochs.
#                 train_epochs=40,
#                 fp16=True,
#                 wandb_run_name=wandb_run_name,
#                 # limit_train_data=0.5,
#                 logging_steps=200,
#                 )
# print(f"\n\n\n Experiment {wandb_run_name} end: {now()}\n\n\n\n")

# wandb_run_name=f"{now()}_40ep_t5_qanom_pred-marker"
# full_experiment(model_type="t5",
#                 train_dataset="qanom",
#                 test_dataset="qanom",
#                 train_epochs=40,
#                 fp16=True,
#                 wandb_run_name=wandb_run_name,
#                 logging_steps=200,
#                 preprocess_input_func="input_predicate_marker"
#                 )
# print(f"\n\n\n Experiment {wandb_run_name} end: {now()}\n\n\n\n")

# wandb_run_name=f"{now()}_40ep_t5_qasrl_pred-marker"
# full_experiment(model_type="t5",
#                 train_dataset="qasrl",
#                 test_dataset="qasrl",
#                 train_epochs=40,
#                 fp16=True,
#                 wandb_run_name=wandb_run_name,
#                 logging_steps=200,
#                 preprocess_input_func="input_predicate_marker"
#                 )
# print(f"\n\n\n Experiment {wandb_run_name} end: {now()}\n\n\n\n")

# wandb_run_name=f"{now()}_40ep_t5_joint_qanom"
# full_experiment(model_type="t5",
#                 train_dataset="joint_qanom",
#                 test_dataset="qanom",
# #                 overwrite_output_dir=True, # if False, continue with last model in model_dir; then need to set `train_epochs` to be the cummulative num of desired epochs.
#                 train_epochs=40,
#                 fp16=True,
#                 wandb_run_name=wandb_run_name,
#                 # limit_train_data=0.5,
#                 logging_steps=200,
#                 )
# print(f"\n\n\n Experiment {wandb_run_name} end: {now()}\n\n\n\n")

# wandb_run_name=f"{now()}_40ep_t5_joint_qasrl"
# full_experiment(model_type="t5",
#                 train_dataset="joint_qasrl",
#                 test_dataset="qanom",
#                 train_epochs=40,
#                 fp16=True,
#                 wandb_run_name=wandb_run_name,
#                 logging_steps=200,
#                 )
# print(f"\n\n\n Experiment {wandb_run_name} end: {now()}\n\n\n\n")

# wandb_run_name=f"{now()}_40ep_t5_zero-shot_qasrl->qanom"
# full_experiment(model_type="t5",
#                 train_dataset="qasrl",
#                 test_dataset="qanom",
#                 train_epochs=40,
#                 fp16=True,
#                 wandb_run_name=wandb_run_name,
#                 logging_steps=200,
#                 )
# print(f"\n\n\n Experiment {wandb_run_name} end: {now()}\n\n\n\n")

# wandb_run_name=f"{now()}_40ep_t5_zero-shot_qanom->qasrl"
# full_experiment(model_type="t5",
#                 train_dataset="qanom",
#                 test_dataset="qasrl",
#                 train_epochs=40,
#                 fp16=True,
#                 wandb_run_name=wandb_run_name,
#                 logging_steps=200,
#                 )
# print(f"\n\n\n Experiment {wandb_run_name} end: {now()}\n\n\n\n")

# wandb_run_name=f"{now()}_40ep_t5_zero-shot_qanom->qasrl"
# full_experiment(model_type="t5",
#                 train_dataset="qanom",
#                 test_dataset="qasrl",
#                 train_epochs=40,
#                 fp16=True,
#                 wandb_run_name=wandb_run_name,
#                 logging_steps=200,
#                 )
# print(f"\n\n\n Experiment {wandb_run_name} end: {now()}\n\n\n\n")




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
model_type="bart"
for task in ["qanom", "qasrl"]: 
    f1s = {}
    for i in [2,3,5,7,10,15,20,25,30,35,40]:
        wandb_run_name=f"test {task} performance per epoch - {model_type} {i}-epochs"
        evaluations = full_experiment(model_type=model_type,
                        train_dataset=task,
                        test_dataset=task,
                        overwrite_output_dir=i==2,
                        train_epochs=i,
                        wandb_run_name=wandb_run_name,
                        finish_wandb=False
                        )
        f1s[i] = evaluations[0].f1()
        print(f"\n\n\n\n\n !!!!!!!! Performance for training on {task} for {i} epochs:  {evaluations[0]}  !!!!!!!!!!  \n\n\n\n\n")
    json.dump(f1s, open(f"epoch-wise-learning_curve_{task}_{model_type}_epochs-to-arg-f1.json", "w"))




#%%
# Accumulative learning-curve estimation (training set size) 

# model_type="t5"
# data = "qasrl"
# epochs = 30
# wandb_run_name=f"{now()}_{data}_{model_type}_{epochs}-epochs_learning_curve" 
# arg_f1 = {}
# for train_sample_size in [2000, 4000, 6000, 8000, 11000, 15000, 20000, 25000, 30000]:  # 1000 samples and less result in 0.0 performance 
#     metrics = full_experiment(model_type=model_type,
#                     train_dataset=data,
#                     test_dataset=data,
#                     train_epochs=epochs,
#                     wandb_run_name=wandb_run_name,
#                     max_train_samples=train_sample_size
#                     )
#     arg_f1[train_sample_size] = metrics[0].f1()
#     print(f"\n\n\n\n\n !!!!!!!! Performance for training on {train_sample_size} samples:  {metrics[0]}  !!!!!!!!!!  \n\n\n\n\n")
# json.dump(arg_f1, open(f"learning_curve_{data}_{model_type}_{epochs}-epochs_sample-size-to-arg-f1.json", "w"))
