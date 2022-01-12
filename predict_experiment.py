from notebook import *

#%%

# Prediction-Only experiment

# model_dir = "/home/nlp/kleinay/tmp/bart-tst-summarization/qasrl/qasrl_long"
model_dir = "trained_models/t5_10ep-joint-qanom_15.12.21"
for model_dir in ["joint_qanom_short-prefix", "qanom_long"]:
    load_and_evaluate("/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/" + model_dir,
                    test_dataset = "qanom",
                    output_dir=None, 
                    wandb_run_name=f"{now()} Evaluate {model_dir}"
                    )