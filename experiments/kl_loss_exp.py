import sys
sys.path.append('..')

from models.base_model import BaseModel
from experiments import run_eval

# Run experiment for evaluating kl loss on base model
run_eval(
    model_class=BaseModel,
    model_params={
        "use_KL": False,
        "risk_lambda": 1e-3
    },
    filename="kl_study/NN.json",
    labelled_prop_list=[0.1, 0.3],
    num_exps=1,
    src_domains=1,
    kl_eval=True
)
