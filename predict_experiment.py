from notebook import *

#%%

# Prediction-Only experiment

# model_dir = "/home/nlp/kleinay/tmp/bart-tst-summarization/qasrl/qasrl_long"
model_dir = "trained_models/t5_qanom-joint-23.03.22"
# model_dir = "trained_models/t5_30ep-qanom-baseline-13.03.22"
# model_dir = "trained_models/t5_30ep-qanom-baseline-17.12.21"
# model_dir = "trained_models/t5_10ep-joint-qanom_15.12.21"
# model_dir = "/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/qanom_baseline"
# model_dir = "/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/debug"

# test_file = "tst2.csv"
test_file = "QANom/dataset/annot.dev.csv"

load_and_predict(model_dir,
                test_file = test_file,
                output_dir=None, 
                wandb_run_name=f"{now()} testing Predict (debug)",
                limit_eval_data=0.1,
                batch_size=5,
                num_beams=4,
                # constrain_generation=False,
                beam_search_method="at_first_token",
                )