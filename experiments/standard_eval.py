from experiments.experiments import run_eval
from models.DANN import DANNModel
from models.base_model import BaseModel


# Run experiments for evaluation using standard evaluation
run_eval(
    model_class=BaseModel,
    model_params={
        "use_KL": False
    },
    filename="standard_eval/NN.json",
    labelled_prop_list=[0.1, 0.3],
    num_exps=3,
    src_domains=1
)

run_eval(
    model_class=BaseModel,
    model_params={
        "use_KL": True,
        "risk_lambda": 1e-3
    },
    filename="standard_eval/NN_KL.json",
    labelled_prop_list=[0.1, 0.3],
    num_exps=3,
    src_domains=1
)

run_eval(
    model_class=DANNModel,
    model_params={
        "use_KL": False,
        "rep_lambda": 1e-1
    },
    filename="standard_eval/DANN.json",
    labelled_prop_list=[0.1, 0.3],
    num_exps=3,
    src_domains=1
)

run_eval(
    model_class=DANNModel,
    model_params={
        "use_KL": True,
        "risk_lambda": 1e-3,
        "rep_lambda": 1e-1
    },
    filename="standard_eval/DANN_KL.json",
    labelled_prop_list=[0.1, 0.3],
    num_exps=3,
    src_domains=1
)