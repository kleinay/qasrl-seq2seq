from notebook import *

#%%

# Prediction-Only experiment
# model_dir = "/home/nlp/kleinay/tmp/t5-tst-summarization/qasrl/qasrl_baseline"
# model_dir = "/home/nlp/kleinay/tmp/t5-tst-summarization/joint_qanom/qanom_joint"
# model_dir = "/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/custom_debug"
# model_dir = "trained_models/t5_qanom-joint-23.03.22"
# model_dir = "trained_models/t5_30ep-qanom-baseline-13.03.22"
model_dir = "trained_models/t5_qasrl-baseline-10.05.22"

# model_dir = "/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/qanom_baseline"

# for model_dir in ["joint_qanom_short-prefix", "qanom_long"]:
#   load_and_evaluate("/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/" + model_dir,
# for method in (  'permutate_sample_num_of_qas',): # 'permutate_sample_fixed',
            #    'all_shuffled', 'all_by_answer_ordering' ):
    # model_dir = f"/home/nlp/kleinay/tmp/t5-tst-summarization/joint_qanom/linearization/{method}"
# for model_dir in ("trained_models/t5_qanom-joint-23.03.22",
#                   "trained_models/t5_qasrl-baseline-10.05.22",
                   # "trained_models/t5_30ep-qanom-baseline-13.03.22",
                #   ):
    # for protocol in ("qanom",):
    #     load_and_evaluate(model_dir,
    #                     test_dataset = "qanom",
    #                     output_dir=None, 
    #                     wandb_run_name=f"evaluate joint {model_dir.split('/')[-1]}, qanom test, {protocol} protocol, keep errors ",
    #                     do_eval_on_dev=False,
    #                     evaluation_protocol=protocol,
    #                     constrain_generation=False,
    #                     # limit_eval_data=0.05,
    #                     batch_size=12,
    #                     num_beams=5,
    #                     )
#         load_and_evaluate(model_dir,
#                         test_dataset = "qanom",
#                         output_dir=None, 
#                         wandb_run_name=f"evaluate: {model_dir[-23:]}, qanom test, {protocol} protocol ",
#                         do_eval_on_dev=False,
#                         evaluation_protocol=protocol,
#                         constrain_generation=False,
#                         # limit_eval_data=0.05,
#                         batch_size=12,
#                         num_beams=3,
#                         )
    
for model_dir in (
    "trained_models/t5_qanom-joint-23.03.22",
    "trained_models/t5_30ep-qanom-baseline-13.03.22",
    # "trained_models/t5_qasrl-baseline-10.05.22"
):
    load_and_evaluate(model_dir,
                test_dataset = "qasrl",
                output_dir=None, 
                wandb_run_name=f"re-run eval {model_dir} eval - mistakes",
                do_eval_on_dev=False,
                evaluation_protocol="qanom",
                # constrain_generation=True,
                # limit_eval_data=0.05,
                batch_size=12,
                num_beams=5,
                )
