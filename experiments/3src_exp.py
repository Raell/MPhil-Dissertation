import sys
sys.path.append('..')

from models.DANN import DANNModel
from models.base_model import BaseModel
from experiments import run_eval

# Run experiments on 3 source domain combinations
run_eval(
    model_class=BaseModel,
    model_params={
        "use_KL": False,
    },
    filename="standard_eval/NN_3src.json",
    labelled_prop_list=[0.1, 0.3],
    num_exps=3,
    src_domains=3,
    standard_eval=True
)

run_eval(
    model_class=BaseModel,
    model_params={
        "use_KL": True,
        "risk_lambda": 1e-3,
        "two_step_training": False
    },
    filename="standard_eval/NN_KL_3src.json",
    labelled_prop_list=[0.1, 0.3],
    num_exps=3,
    src_domains=3,
    standard_eval=True
)

run_eval(
    model_class=DANNModel,
    model_params={
        "use_KL": True,
        "risk_lambda": 1e-3,
        "rep_lambda": 1e-1
    },
    filename="standard_eval/DANN_KL_3src.json",
    labelled_prop_list=[0.1, 0.3],
    num_exps=3,
    src_domains=3,
    standard_eval=True
)

run_eval(
    model_class=DANNModel,
    model_params={
        "use_KL": True,
        "risk_lambda": 1e-3,
        "rep_lambda": 1e-1,
        "two_step_training": False
    },
    filename="standard_eval/DANN_KL_3src.json",
    labelled_prop_list=[0.1, 0.3],
    num_exps=3,
    src_domains=3,
    standard_eval=True
)
