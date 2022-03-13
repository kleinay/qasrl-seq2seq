from notebook import *

#%%

# Prediction-Only experiment

# model_dir = "/home/nlp/kleinay/tmp/bart-tst-summarization/qasrl/qasrl_long"
# model_dir = "trained_models/t5_10ep-joint-qanom_15.12.21"
# model_dir = "trained_models/t5_30ep-qanom-baseline-17.12.21"
model_dir = "trained_models/t5_30ep-qanom-baseline-13.03.22"
# for model_dir in ["joint_qanom_short-prefix", "qanom_long"]:
#   load_and_evaluate("/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/" + model_dir,
load_and_evaluate(model_dir,
                test_dataset = "qanom",
                output_dir=None, 
                wandb_run_name=f"{now()} Evaluate qanom baseline",
                do_eval_on_dev=True,
                evaluation_protocol="qanom",
                # constrain_generation=False,
                # limit_eval_data=0.05,
                num_beams=3,
                )
# load_and_evaluate(model_dir,
#                 test_dataset = "qasrl",
#                 output_dir=None, 
#                 wandb_run_name=f"{now()} qasrl protocol",
#                 do_eval_on_dev=True,
#                 evaluation_protocol="qasrl",
#                 # constrain_generation=False,
#                 # limit_eval_data=0.1,
#                 num_beams=3,
#                 )
# load_and_evaluate(model_dir,
#                 test_dataset = "qanom",
#                 output_dir=None, 
#                 wandb_run_name=f"{now()} Evaluate {model_dir} on qanom",
#                 # constrain_generation=False,
#                 num_beams=3,
#                 )
# load_and_evaluate(model_dir,
#                 test_dataset = "qasrl",
#                 output_dir=None, 
#                 wandb_run_name=f"{now()} Evaluate {model_dir} on qasrl",
#                 # constrain_generation=True,
#                 num_beams=3,
#                 )