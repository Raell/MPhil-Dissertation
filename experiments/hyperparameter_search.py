from experiments.experiments import hyperparam_search
from models.DANN import DANNModel
from models.base_model import BaseModel

# Run learning rate search
hyperparam_search(
    BaseModel,
    model_params={
        "use_KL": False,
    },
    param_search_dict={
        "lr": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    },
    filename=f"hyperparam_search/lr_search.json"
)

# Run risk lambda search
hyperparam_search(
    BaseModel,
    model_params={
        "use_KL": True,
    },
    param_search_dict={
        "risk_lambda": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    },
    filename=f"hyperparam_search/NN_KL_search.json"
)

# Run rep lambda search
hyperparam_search(
    DANNModel,
    model_params={
        "use_KL": False
    },
    param_search_dict={
        "rep_lambda": [1e-3, 1e-2, 1e-1, 1, 10]
    },
    filename=f"hyperparam_search/DANN_search.json"
)

# Run risk and rep lambda cross search
hyperparam_search(
    DANNModel,
    model_params={
        "use_KL": True,
    },
    param_search_dict={
        "risk_lambda": [1e-4, 1e-3, 1e-2],
        "rep_lambda": [1e-2, 1e-1, 1]
    },
    filename=f"DANN_KL_search.json"
)