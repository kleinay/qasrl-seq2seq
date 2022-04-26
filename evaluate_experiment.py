from notebook import *

#%%

# Prediction-Only experiment
# model_dir = "/home/nlp/kleinay/tmp/t5-tst-summarization/qasrl/qasrl_baseline"
# model_dir = "/home/nlp/kleinay/tmp/t5-tst-summarization/joint_qanom/qanom_joint"
# model_dir = "/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/custom_debug"
model_dir = "trained_models/t5_qanom-joint-23.03.22"
# model_dir = "trained_models/t5_10ep-joint-qanom_15.12.21"
# model_dir = "trained_models/t5_30ep-qanom-baseline-17.12.21"
# model_dir = "trained_models/t5_30ep-qanom-baseline-13.03.22"
# for model_dir in ["joint_qanom_short-prefix", "qanom_long"]:
#   load_and_evaluate("/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/" + model_dir,

load_and_evaluate(model_dir,
                test_dataset = "qanom",
                output_dir=None, 
                wandb_run_name=f"{now()} debug evaluate model ",
                do_eval_on_dev=True,
                evaluation_protocol="qanom",
                constrain_generation=False,
                limit_eval_data=0.05,
                batch_size=8,
                num_beams=3,
                )
# load_and_evaluate(model_dir,
#                 test_dataset = "qasrl",
#                 output_dir=None, 
#                 wandb_run_name=f"{now()} Evaluate qasrl joint (qasrl protocol) - constrained",
#                 do_eval_on_dev=True,
#                 evaluation_protocol="qasrl",
#                 constrain_generation=True,
#                 # limit_eval_data=0.05,
#                 num_beams=5,
#                 )
