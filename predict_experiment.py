from notebook import *

#%%

# Prediction-Only experiment

# model_dir = "/home/nlp/kleinay/tmp/bart-tst-summarization/qasrl/qasrl_long"
model_dir = "trained_models/t5_10ep-joint-qanom_15.12.21"
# model_dir = "trained_models/t5_30ep-qanom-baseline-17.12.21"
# model_dir = "/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/qanom_baseline"

load_and_predict(model_dir,
                test_file = "tst2.csv",
                output_dir=None, 
                wandb_run_name=f"{now()} testing Predict (debug)",
                # constrain_generation=False,
                # limit_eval_data=0.05,
                num_beams=3,
                )