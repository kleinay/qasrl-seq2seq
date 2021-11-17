# Design large-scale experiments (leanring curve analysis, mixed training, etc.)
#%%
from notebook import *

#%%
# Simple experiment

# full_experiment(model_type="t5",
#                 train_dataset="qanom",
#                 test_dataset="qanom",
#                 overwrite_output_dir=True, # if False, continue with last model in model_dir; then need to set `train_epochs` to be the cummulative num of desired epochs.
#                 train_epochs=1,
#                 run_name=f"{now()}_debug",
#                 limit_train_data=0.5,
#                 logging_steps=20,
#                 )

#%% 
# Joint Training - start with QASRL, then finetune on qanom

# model_type="t5"
# run_name=f"{now()}_t5_mixed_qasrl-15"   
# full_experiment(model_type=model_type,
#                 train_dataset="qasrl",
#                 test_dataset="qanom",
#                 train_epochs=15,
#                 run_name=run_name,
#                 )
# full_experiment(model_type=model_type,
#                 train_dataset="qanom",
#                 test_dataset="qanom",
#                 overwrite_output_dir=False,
#                 train_epochs=25,
#                 run_name=run_name + "_qanom-25",
#                 )


# model_type="bart"
# run_name=f"{now()}_bart_mixed_qasrl-5_qanom-15"
# full_experiment(model_type=model_type,
#                 train_dataset="qasrl",
#                 test_dataset="qanom",
#                 train_epochs=5,
#                 run_name=run_name,
#                 )
# full_experiment(model_type=model_type,
#                 train_dataset="qanom",
#                 test_dataset="qanom",
#                 overwrite_output_dir=False,
#                 train_epochs=15,
#                 run_name=run_name,
#                 )

#%%
# Accumulative learning-curve estimation 

model_type="t5"
data = "qasrl"
run_name=f"{now()}_{data}_{model_type}_learning_curve" 
arg_f1 = {}
for train_sample_size in [12000, 20000, 30000]:  # 1000 samples and less result in 0.0 performance 
    metrics = full_experiment(model_type=model_type,
                    train_dataset=data,
                    test_dataset=data,
                    train_epochs=10,
                    run_name=run_name,
                    max_train_samples=train_sample_size
                    )
    arg_f1[train_sample_size] = metrics[0].f1()
    print(f"\n\n\n\n\n !!!!!!!! Performance for training on {train_sample_size} samples:  {metrics[0]}  !!!!!!!!!!  \n\n\n\n\n")
json.dump(arg_f1, open(f"learning_curve_{data}_{model_type}_epochs-to-arg-f1.json", "w"))
