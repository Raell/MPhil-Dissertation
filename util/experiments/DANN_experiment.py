import os
import json
import sys

sys.path.append('../..')

from models.DANN import DANNModel
from util.experiments.experiment import run_experiment

data_folder = "~/Datasets/Experiment"
src_folder = os.path.join(data_folder, "Real World")
target_folder = os.path.join(data_folder, "Product")

labelled_prop = [0.01, 0.05, 0.1, 0.15, 0.25, 0.4, 0.7, 1]
risk_lambda = [100, 100, 100, 100, 10, 10, 10, 10, 10, 10]

num_experiments = 5
exp_history = {}

for i, (p, risk) in enumerate(zip(labelled_prop, risk_lambda)):
    print("DANN Semi-Supervised")
    print(f"{int(p * 100)}% labelled target")
    exp_history[f"{int(p * 100)}% labels"] = run_experiment(
        src_folder,
        target_folder,
        model_class=DANNModel,
        model_params={
            "use_KL": False,
            "risk_lambda": risk,
            "lr": 1e-5
        },
        num_experiments=num_experiments,
        patience=5,
        target_labels=p,
        verbose=False,
        min_epochs=20,
        max_epochs=100
    )
    print()

with open('DANN.json', 'w') as fp:
    json.dump(exp_history, fp)