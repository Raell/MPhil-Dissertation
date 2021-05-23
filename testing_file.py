import argparse
import itertools
import json
import os
import re
import shutil
from collections import defaultdict
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from pytorch_lightning.callbacks import ProgressBar, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import LightningLoggerBase
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

import logging
logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False

from models.DANN import DANNModel
from models.base_model import SingleDomainModel
import pytorch_lightning as pl

from util.data import DataGenerator, Dataset

DATA_FOLDER = "~/Datasets/Experiment"
DOMAINS = ["Art", "Clipart", "Product", "Real World"]
DOMAIN_FOLDERS = {d: os.path.join(DATA_FOLDER, d) for d in DOMAINS}


class MetricsLogger(LightningLoggerBase):

    def __init__(self):
        super().__init__()
        self.metrics = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list)
        }

    @property
    def experiment(self) -> Any:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for k, v in metrics.items():
            match = re.match("^(train|val|test)_(.*)$", k)
            if match:
                self.metrics[match.group(1)][match.group(2)].append(v)

    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs):
        pass

    @property
    def name(self) -> str:
        return "MetricsLogger"

    @property
    def version(self) -> Union[int, str]:
        pass


class LitProgressBar(ProgressBar):

    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar


class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)



def train_model(model, train, test, epochs=None, eval=True):
    logger = MetricsLogger()
    pbar = LitProgressBar()

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        callbacks=[
            pbar
        ],
        max_epochs=epochs,
        checkpoint_callback=False
    )
    trainer.fit(model, train_dataloader=train, val_dataloaders=test)
    if eval:
        trainer.test(model, test)
        return logger.metrics
    else:
        return trainer.predict(model, test)


class Results:
    def __init__(self, filename):
        self.filename = filename

        if os.path.isfile(filename):
            with open(filename) as json_file:
                self.metrics = defaultdict(dict, json.load(json_file))
                self.resume = True
        else:
            self.metrics = defaultdict(dict)
            self.resume = False

    def update(self, new_data):
        self.metrics = new_data

    def save(self):
        with open(self.filename, "w") as fp:
            json.dump(self.metrics, fp, indent="\t")


def run_experiments(
        model_class,
        model_params,
        train,
        test,
        epochs=None,
        return_preds=False
):
    model = model_class(**model_params)
    results = train_model(model, train, test, epochs=epochs, eval=not return_preds)
    return results


def get_cross_product(d):

    pairs = d.items()
    keys = [k for k, _ in pairs]
    vals = [v for _, v in pairs]
    param_list = itertools.product(*vals)

    cross = [{k: v for k, v in zip(keys, p)} for p in param_list]
    return cross


def hyperparam_search(
    model_class,
    param_search_dict,
    filename,
    num_epochs=30,
    use_KL=False
):
    results = Results(filename)

    metrics = defaultdict(dict, results.metrics)

    for src, target in itertools.permutations(DOMAINS, 2):
        src_folder = DOMAIN_FOLDERS[src]
        target_folder = DOMAIN_FOLDERS[target]

        data = DataGenerator(
            source_domain=src_folder,
            target_domain=target_folder
        )

        model_params = {
            "use_KL": use_KL,
            "lr": 1e-5,
            "classes": len(data.classes)
        }

        is_DANN = model_class == DANNModel

        param_list = get_cross_product(param_search_dict)

        for params in param_list:
            header = ""
            for k, v in params.items():
                model_params[k] = v
                header += f"{k} {v}, "

            header = header[:-2]

            print(f"{src} -> {target}, {header}")

            if header in metrics and f"{src} -> {target}" in metrics[header]:
                continue

            src_train, src_test, tar_nlabel = data.get_TCV()

            train_label_loader = DataLoader(
                dataset=src_train,
                batch_size=64,
                shuffle=True,
                # num_workers=8
            )

            train_loader = train_label_loader

            if is_DANN or use_KL:
                train_nlabel_loader = DataLoader(
                    dataset=tar_nlabel,
                    batch_size=64,
                    shuffle=True,
                    # num_workers=8
                )

                train_loader = [train_label_loader, train_nlabel_loader]

            test_loader = DataLoader(
                dataset=tar_nlabel,
                batch_size=64,
                # num_workers=8
            )

            preds = run_experiments(
                model_class,
                model_params,
                train_loader,
                test_loader,
                num_epochs,
                return_preds=True
            )

            if is_DANN:
                preds = [p[0] for p in preds]

            preds = np.concatenate(preds, axis=0)
            if use_KL:
                preds = np.sum(preds, axis=1)

            preds = torch.Tensor([(1, pred) for pred in np.argmax(preds, axis=1)]).long()

            train = ZipDataset(tar_nlabel, preds)

            train_label_loader = DataLoader(
                dataset=train,
                batch_size=64,
                shuffle=True,
                # num_workers=8
            )

            train_loader = train_label_loader

            if is_DANN or use_KL:
                train_nlabel_loader = DataLoader(
                    dataset=Dataset(src_train, labels=False),
                    batch_size=64,
                    shuffle=True,
                    # num_workers=8
                )
                train_loader = [train_label_loader, train_nlabel_loader]

            test_loader = DataLoader(
                dataset=src_test,
                batch_size=64,
                # num_workers=8
            )

            metrics[header][f"{src} -> {target}"] = run_experiments(
                model_class,
                model_params,
                train_loader,
                test_loader,
                num_epochs
            )

            results.update(metrics)
            results.save()


def run_eval(
    model_class,
    filename,
    labelled_prop_list=[0.1, 0.3],
    num_epochs=100,
    hyperparams_dict={},
    use_KL=False,
    num_exps=3,
    lr=1e-4
):
    results = Results(filename)

    overall_metrics = defaultdict(dict, results.metrics)

    for src, target in itertools.permutations(DOMAINS, 2):
        src_folder = DOMAIN_FOLDERS[src]
        target_folder = DOMAIN_FOLDERS[target]

        data = DataGenerator(
            source_domain=src_folder,
            target_domain=target_folder
        )

        model_params = {
            "use_KL": use_KL,
            "lr": lr,
            "classes": len(data.classes)
        }

        for k, v in hyperparams_dict.items():
            model_params[k] = v

        is_DANN = model_class == DANNModel
        metrics = defaultdict(dict, overall_metrics[f"{src} -> {target}"])

        for p in labelled_prop_list:
            header = f"{int(p * 100)}% labels"

            train_label, train_nlabel, test = data.get_datasets(p)

            train_label_loader = DataLoader(
                dataset=train_label,
                batch_size=64,
                shuffle=True,
                num_workers=8
            )

            train_loader = train_label_loader

            if is_DANN or use_KL:
                train_nlabel_loader = DataLoader(
                    dataset=train_nlabel,
                    batch_size=64,
                    shuffle=True,
                    num_workers=8
                )

                train_loader = [train_label_loader, train_nlabel_loader]

            test_loader = DataLoader(
                dataset=test,
                batch_size=64,
                num_workers=8
            )

            for i in range(num_exps):
                print(f"{src} -> {target}, {header}, Experiment {i+1}")

                if header in metrics and f"Experiment {i+1}" in metrics[header]:
                    continue

                metrics[header][f"Experiment {i+1}"] = run_experiments(
                    model_class,
                    model_params,
                    train_loader,
                    test_loader,
                    num_epochs
                )

                overall_metrics[f"{src} -> {target}"] = metrics

                results.update(overall_metrics)
                results.save()


# data_folder = "~/Datasets/Experiment"
# src_folder = os.path.join(data_folder, "Real World")
# target_folder = os.path.join(data_folder, "Product")
# data = DataGenerator(
#     source_domain=src_folder,
#     target_domain=target_folder
# )

# run_eval(
#     DANNModel,
#     data,
#     "DANN.json"
# )

# hyperparam_search(
#     SingleDomainModel,
#     {
#         "risk_lambda": [1e-3, 1e-2, 1e-1]
#     },
#     # f"NN_KL_search.json",
#     "test.json",
#     use_KL=True,
#     num_epochs=1
# )

# hyperparam_search(
#     DANNModel,
#     {
#         "rep_lambda": [1e-3, 1e-2, 1e-1]
#     },
#     f"DANN_search.json",
#     use_KL=False
# )
#
# hyperparam_search(
#     DANNModel,
#     {
#         "risk_lambda": [1e-3, 1e-2, 1e-1],
#         "rep_lambda": [1e-3, 1e-2, 1e-1]
#     },
#     f"DANN_KL_search.json",
#     use_KL=True
# )

run_eval(
    SingleDomainModel,
    "NN_KL.json",
    num_epochs=100,
    num_exps=1,
    hyperparams_dict={
        "risk_lambda": 0.01
    },
    lr=1e-4
)

run_eval(
    SingleDomainModel,
    "NN.json",
    num_epochs=100,
    num_exps=3,
    lr=1e-4
)

# run_eval(
#     SingleDomainModel,
#     "higher_lr.json",
#     num_epochs=100,
#     num_exps=3,
#     lr=1e-4
# )
