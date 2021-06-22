from models.DANN import DANNModel
from models.base_model import BaseModel
from experiments.experiments import run_eval

# Run experiments for evaluation using reverse validation
run_eval(
    model_class=BaseModel,
    model_params={
        "use_KL": False
    },
    filename="TCV_eval/NN.json",
    labelled_prop_list=[0.1, 0.3],
    num_exps=3,
    src_domains=1,
    standard_eval=False
)

run_eval(
    model_class=BaseModel,
    model_params={
        "use_KL": True,
        "risk_lambda": 1e-3
    },
    filename="TCV_eval/NN_KL.json",
    labelled_prop_list=[0.1, 0.3],
    num_exps=3,
    src_domains=1,
    standard_eval=False
)

run_eval(
    model_class=DANNModel,
    model_params={
        "use_KL": False,
        "rep_lambda": 1e-1
    },
    filename="TCV_eval/DANN.json",
    labelled_prop_list=[0.1, 0.3],
    num_exps=3,
    src_domains=1,
    standard_eval=False
)

run_eval(
    model_class=DANNModel,
    model_params={
        "use_KL": True,
        "risk_lambda": 1e-3,
        "rep_lambda": 1e-1
    },
    filename="TCV_eval/DANN_KL.json",
    labelled_prop_list=[0.1, 0.3],
    num_exps=3,
    src_domains=1,
    standard_eval=False
)