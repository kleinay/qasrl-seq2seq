from notebook import *

#%%

# Prediction-Only experiment
# model_dir = "/home/nlp/kleinay/tmp/t5-tst-summarization/qasrl/qasrl_baseline"
# model_dir = "/home/nlp/kleinay/tmp/t5-tst-summarization/joint_qanom/qanom_joint"
# model_dir = "/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/custom_debug"
# model_dir = "trained_models/t5_qanom-joint-23.03.22"
# model_dir = "trained_models/t5_30ep-qanom-baseline-13.03.22"
# model_dir = "trained_models/t5_qasrl-baseline-10.05.22"

# model_dir = "/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/qanom_baseline"

# for model_dir in ["joint_qanom_short-prefix", "qanom_long"]:
#   load_and_evaluate("/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/" + model_dir,
# for method in (  'permutate_sample_num_of_qas',): # 'permutate_sample_fixed',
            #    'all_shuffled', 'all_by_answer_ordering' ):
    # model_dir = f"/home/nlp/kleinay/tmp/t5-tst-summarization/joint_qanom/linearization/{method}"
    # model_dir = f"/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/qanom/linearization/{method}"
src_domains = ["wikinews", "wikipedia", "TQA"]
dst_domains = ["wikinews", "wikipedia"]
for src_domain in src_domains:
    for dst_domain in dst_domains:
        model_dir = f"/home/nlp/kleinay/tmp/t5-tst-summarization/qasrl/qasrl/cross_domain/best-small/{src_domain}"
        load_and_evaluate(model_dir,
                        test_dataset = "qasrl",
                        test_domains = dst_domain,
                        output_dir=None, 
                        wandb_run_name=f"{now()} CD small: evaluate small qasrl-{src_domain} on qasrl {dst_domain} test",
                        do_eval_on_dev=False,
                        evaluation_protocol="qanom",
                        constrain_generation=False,
                        # limit_eval_data=0.05,
                        batch_size=12,
                        num_beams=5,
                        )
    
# for model_dir in (
#     # "trained_models/t5_qanom-joint-23.03.22",
#     # "trained_models/t5_qasrl-baseline-10.05.22",
#     # "trained_models/t5_30ep-qanom-baseline-13.03.22",
#     # "/home/nlp/kleinay/tmp/t5-tst-summarization/qasrl/full-data-linearization/all_by_role_ordering",
#     # "/home/nlp/kleinay/tmp/t5-tst-summarization/qanom/qanom/linearization/permutate_sample_num_of_qas",
#     "/home/nlp/kleinay/tmp/t5-tst-summarization/joint_qanom/linearization/permutate_sample_num_of_qas",
#                   ):
#     load_and_evaluate(model_dir,
#                     test_dataset = "qasrl",
#                     output_dir=None, 
#                     wandb_run_name=f"evaluate qanom-joint-permutate-num-qas on qasrl test",
#                     # wandb_run_name=f"debug evaluate",
#                     do_eval_on_dev=False,
#                     evaluation_protocol="qanom",
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
    
# for method in (
#         'permutate_sample_num_of_qas',
#         'all_by_answer_ordering',
#         'all_shuffled',
#         'permutate_sample_fixed',
#         'permutate_all',
#     ):
#     model_dir = f"/home/nlp/kleinay/tmp/t5-tst-summarization/qasrl/qasrl/linearization/{method}"
#     load_and_evaluate(model_dir,
#                 test_dataset = "qasrl",
#                 output_dir=None, 
#                 wandb_run_name=f"re-run eval {model_dir} eval - qa-position",
#                 do_eval_on_dev=False,
#                 evaluation_protocol="qanom",
#                 # constrain_generation=True,
#                 # limit_eval_data=0.05,
#                 batch_size=12,
#                 num_beams=5,
#                 )
