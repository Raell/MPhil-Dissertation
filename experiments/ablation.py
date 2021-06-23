import sys
sys.path.append('..')

from models.base_model import BaseModel
from experiments import run_eval


# Run ablation on source only and labelled target only experiments
run_eval(
    model_class=BaseModel,
    model_params={
        "use_KL": False
    },
    filename="standard_eval/NN_source_only.json",
    labelled_prop_list=[0],
    num_exps=1,
    src_domains=1,
    source_only=True
)

run_eval(
    model_class=BaseModel,
    model_params={
        "use_KL": True,
        "risk_lambda": 1e-3
    },
    filename="standard_eval/NN_KL_source_only.json",
    labelled_prop_list=[0],
    num_exps=1,
    src_domains=1,
    source_only=True
)


run_eval(
    model_class=BaseModel,
    model_params={
        "use_KL": False
    },
    filename="standard_eval/NN_label_target.json",
    labelled_prop_list=[0.1, 0.3],
    num_exps=1,
    src_domains=1,
    labeled_target_only=True
)

run_eval(
    model_class=BaseModel,
    model_params={
        "use_KL": True,
        "risk_lambda": 1e-3
    },
    filename="standard_eval/NN_KL_label_target.json",
    labelled_prop_list=[0.1, 0.3],
    num_exps=1,
    src_domains=1,
    source_only=True,
    labeled_target_only=True
)
