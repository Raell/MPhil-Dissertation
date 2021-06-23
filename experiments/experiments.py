import argparse
import itertools
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, Optional, Union
import sys
sys.path.append('..')

import numpy as np
import torch
from pytorch_lightning.callbacks import ProgressBar, EarlyStopping
from pytorch_lightning.loggers import LightningLoggerBase
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False

from models.DANN import DANNModel
import pytorch_lightning as pl

from util.data import DataGenerator, Dataset

DATA_FOLDER = "~/Datasets/Experiment"
DOMAINS = ["Art", "Clipart", "Product", "Real World"]
DOMAIN_FOLDERS = {d: os.path.join(DATA_FOLDER, d) for d in DOMAINS}


class MetricsLogger(LightningLoggerBase):
    # Logger for extracting metrics from model
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
    # Removes progress bar update due to visual bug in validation
    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar


class ZipDataset(torch.utils.data.Dataset):
    # Class for merging data with labels into Dataset
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def train_model(model, train, val, test, min_epochs=30, max_epochs=100, eval=True, patience=3):
    # Function for training model and returning predictions or test results
    logger = MetricsLogger()
    pbar = LitProgressBar()

    early_stopping = EarlyStopping(
        monitor='val_acc',
        patience=patience
    )

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        callbacks=[
            pbar,
            early_stopping
        ],
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        checkpoint_callback=False
    )
    trainer.fit(model, train_dataloader=train, val_dataloaders=val)
    if eval:
        trainer.test(model, test)
        return logger.metrics
    else:
        return trainer.predict(model, test)


class Results:
    # Framework for saving results in json format
    def __init__(self, filename):
        self.filename = filename

        # Load existing file if available
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
        # Saves to json file
        with open(self.filename, "w") as fp:
            json.dump(self.metrics, fp, indent="\t")


def run_experiments(
        model_class,
        model_params,
        train,
        val,
        test,
        return_preds=False,
        patience=3
):
    # Train and evaluate models
    model = model_class(**model_params)
    results = train_model(model, train, val, test, eval=not return_preds, patience=patience)

    # Returns either predicted labels or test results
    if return_preds:
        if model_class == DANNModel:
            results = [p[0] for p in results]

        results = np.concatenate(results, axis=0)

        if model_params["use_KL"]:
            results = np.sum(results, axis=1)

    return results


def eval_KL(
        model_class,
        model_params,
        train,
        val,
        test,
        patience=3
):
    # Train and evaluate model for KL loss test

    # Base model training
    is_DANN = model_class == DANNModel
    if not is_DANN:
        train_set = train[0]
    params = dict(model_params)
    model = model_class(**params)
    _ = train_model(model, train_set, val, test, patience=patience)

    # Freeze encoder and train a new joint classifier
    encoder_params = model.encoder.state_dict()
    params["use_KL"] = True
    params["kl_eval"] = True
    kl_model = model_class(**params)
    kl_model.encoder.load_state_dict(encoder_params)
    for param in kl_model.encoder.parameters():
        param.requires_grad = False

    results = train_model(kl_model, train, val, test, patience=patience)
    return results


def run_eval_KL(
        model_class,
        model_params,
        data,
        results,
        target_labels=0.0,
        num_exps=1,
        main_header="",
        sub_header=""
):
    # Runs KL loss evaluation

    full_metrics = defaultdict(dict, results.metrics)
    metrics = defaultdict(dict, full_metrics[main_header])

    for i in range(num_exps):
        print(f"{main_header}, {sub_header}, Experiment {i + 1}")

        if (
                main_header in full_metrics and
                sub_header in full_metrics[main_header] and
                f"Experiment {i + 1}" in full_metrics[main_header][sub_header]
        ):
            continue

        # Load datasets
        datasets = data.get_datasets(target_labels=target_labels)
        src_train, src_test, tar_label, tar_nlabel, tar_test = datasets

        # Prepare dataloaders
        dataset = ConcatDataset([tar_label, src_train, src_test])
        train, val = split_data(dataset, 0.2)
        train_label_loader = prepare_loader(train, is_train=True)
        val_loader = prepare_loader(val, is_train=False)

        train_nlabel_loader = prepare_loader(tar_nlabel, is_train=True)
        train_loader = [train_label_loader, train_nlabel_loader]

        test_loader = prepare_loader(tar_test, is_train=False)

        # Run evaluation
        metrics[sub_header][f"Experiment {i + 1}"] = eval_KL(
            model_class,
            model_params,
            train_loader,
            val_loader,
            test_loader,
        )

        # Saves results
        full_metrics[main_header] = metrics
        results.update(full_metrics)
        results.save()


def get_cross_product(d):
    # Returns all possible cross-product of dictionary keys
    pairs = d.items()
    keys = [k for k, _ in pairs]
    vals = [v for _, v in pairs]
    param_list = itertools.product(*vals)

    cross = [{k: v for k, v in zip(keys, p)} for p in param_list]
    return cross


def prepare_loader(dataset, is_train=False):
    # Returns dataloader from dataset
    return DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=is_train
    )


def split_data(data, split, seed=None):
    # Splits dataset into 2 proportioned subsets
    split1, split2 = torch.utils.data.random_split(
        data,
        [
            round(len(data) * (1 - split) - 1e-5),
            round(len(data) * split + 1e-5)
        ],
        generator=None if seed is None else torch.Generator().manual_seed(seed)
    )
    return split1, split2


def run_standard_val(
        model_class,
        model_params,
        data,
        results,
        target_labels=0.0,
        num_exps=3,
        main_header="",
        sub_header="",
        labeled_target_only=False,
        source_only=False
):
    # Performs standard evaluation of model

    is_DANN = model_class == DANNModel
    use_KL = model_params["use_KL"]

    full_metrics = defaultdict(dict, results.metrics)
    metrics = defaultdict(dict, full_metrics[main_header])

    for i in range(num_exps):
        print(f"{main_header}, {sub_header}, Experiment {i + 1}")

        # Skips experiment if already recorded
        if (
                main_header in full_metrics and
                sub_header in full_metrics[main_header] and
                f"Experiment {i + 1}" in full_metrics[main_header][sub_header]
        ):
            continue

        # Load datasets
        datasets = data.get_datasets(target_labels=target_labels)
        src_train, src_test, tar_label, tar_nlabel, tar_test = datasets

        # Prepare dataloaders
        dataset = ConcatDataset([tar_label, src_train, src_test])
        if labeled_target_only:
            dataset = tar_label
        elif source_only:
            dataset = ConcatDataset([src_train, src_test])
        train, val = split_data(dataset, 0.2)
        train_label_loader = prepare_loader(train, is_train=True)
        val_loader = prepare_loader(val, is_train=False)

        if is_DANN or use_KL:
            if source_only:
                train_nlabel_loader = prepare_loader(Dataset(train, labels=False), is_train=True)
            else:
                train_nlabel_loader = prepare_loader(tar_nlabel, is_train=True)

            train_loader = [train_label_loader, train_nlabel_loader]

        else:
            train_loader = train_label_loader

        test_loader = prepare_loader(tar_test, is_train=False)

        # Run evaluation
        metrics[sub_header][f"Experiment {i + 1}"] = run_experiments(
            model_class,
            model_params,
            train_loader,
            val_loader,
            test_loader,
        )

        full_metrics[main_header] = metrics

        results.update(full_metrics)
        results.save()


def run_reverse_val(
        model_class,
        model_params,
        data,
        results,
        target_labels=0.0,
        eval=False,
        num_exps=3,
        main_header="",
        sub_header=""
):
    is_DANN = model_class == DANNModel
    use_KL = model_params["use_KL"]

    full_metrics = defaultdict(dict, results.metrics)
    metrics = defaultdict(dict, full_metrics[main_header])

    for i in range(num_exps):
        print(f"{main_header}, {sub_header}, Experiment {i + 1}")

        if (
                main_header in full_metrics and
                sub_header in full_metrics[main_header] and
                f"Experiment {i + 1}" in full_metrics[main_header][sub_header]
        ):
            continue

        # Load datasets
        datasets = data.get_datasets(target_labels=target_labels)
        src_train, src_test, tar_label, tar_nlabel, tar_test = datasets

        # Prepare dataloaders
        train_label_loader = prepare_loader(src_train, is_train=True)
        val_loader = prepare_loader(src_test, is_train=False)

        if is_DANN or use_KL:
            train_nlabel_loader = prepare_loader(tar_nlabel, is_train=True)
            train_loader = [train_label_loader, train_nlabel_loader]
        else:
            train_loader = train_label_loader

        test_loader = prepare_loader(tar_nlabel, is_train=False)

        # Run source training and return predictions
        preds = run_experiments(
            model_class,
            model_params,
            train_loader,
            val_loader,
            test_loader,
            return_preds=True
        )

        # Repeat process for reverse classifier using predicted labels
        preds = torch.Tensor([(data.domains - 1, pred) for pred in np.argmax(preds, axis=1)]).long()

        # Prepare dataloaders
        tar_nlabel = ZipDataset(tar_nlabel, preds)
        dataset = ConcatDataset([tar_label, tar_nlabel])

        train, val = split_data(dataset, 0.2)
        train_label_loader = prepare_loader(train, is_train=True)
        val_loader = prepare_loader(val, is_train=False)

        if is_DANN or use_KL:
            train_nlabel_loader = prepare_loader(Dataset(src_train, labels=False), is_train=True)
            train_loader = [train_label_loader, train_nlabel_loader]
        else:
            train_loader = train_label_loader

        if eval:
            test_loader = prepare_loader(tar_test, is_train=False)
        else:
            test_loader = prepare_loader(src_test, is_train=False)

        # Run target training and evaluate on source
        metrics[sub_header][f"Experiment {i + 1}"] = run_experiments(
            model_class,
            model_params,
            train_loader,
            val_loader,
            test_loader
        )

        # Saves results
        full_metrics[main_header] = metrics
        results.update(full_metrics)
        results.save()


def hyperparam_search(
        model_class,
        model_params,
        param_search_dict,
        filename,
        src_domains=1,
        num_exps=1
):
    # Runs hyperparameter search using reverse validation

    results = Results(filename)
    domains_combo = domain_combo(src_domains)

    for src, target in domains_combo:
        src_folder = [DOMAIN_FOLDERS[s] for s in src]
        target_folder = [DOMAIN_FOLDERS[target]]

        data = DataGenerator(
            source_domain=src_folder,
            target_domain=target_folder
        )

        model_params["classes"] = len(data.classes)
        model_params["domains"] = data.domains

        param_list = get_cross_product(param_search_dict)

        for params in param_list:
            main_header = ""
            for k, v in params.items():
                model_params[k] = v
                main_header += f"{k} {v}, "
            main_header = main_header[:-2]
            sub_header = f"{src} -> {target}"
            run_reverse_val(
                model_class,
                model_params,
                data,
                results,
                num_exps=num_exps,
                main_header=main_header,
                sub_header=sub_header
            )


def domain_combo(src_domains):
    # Loads all src/tar domain combinations with fixed number of src domains
    combo = []
    if src_domains == 1:
        for s in DOMAINS:
            for t in DOMAINS:
                if s == t:
                    continue
                combo.append(([s], t))

    elif src_domains == 2:
        for t in DOMAINS:
            remain = [d for d in DOMAINS if d != t]
            for src in itertools.combinations(remain, 2):
                combo.append((src, t))

    elif src_domains == 3:
        for t in DOMAINS:
            combo.append(([d for d in DOMAINS if d != t], t))
    return combo


def run_eval(
        model_class,
        model_params,
        filename,
        labelled_prop_list=[0.1, 0.3],
        num_exps=3,
        src_domains=1,
        standard_eval=True,
        labeled_target_only=False,
        source_only=False,
        kl_eval=False
):
    # Runs experimental setup

    results = Results(filename)
    domains_combo = domain_combo(src_domains)

    # Testing on all src/tar domain combinations
    for src, target in domains_combo:
        src_folder = [DOMAIN_FOLDERS[s] for s in src]
        target_folder = [DOMAIN_FOLDERS[target]]

        main_header = f"{src} -> {target}"

        data = DataGenerator(
            source_domain=src_folder,
            target_domain=target_folder
        )

        model_params["classes"] = len(data.classes)
        model_params["domains"] = data.domains

        for p in labelled_prop_list:
            sub_header = f"{int(p * 100)}% labels"

            # Runs KL loss evaluation
            if kl_eval:
                run_eval_KL(
                    model_class,
                    model_params,
                    data,
                    results,
                    target_labels=p,
                    main_header=main_header,
                    sub_header=sub_header
                )

            # Runs standard evaluation
            elif standard_eval:
                run_standard_val(
                    model_class,
                    model_params,
                    data,
                    results,
                    target_labels=p,
                    num_exps=num_exps,
                    main_header=main_header,
                    sub_header=sub_header,
                    labeled_target_only=labeled_target_only,
                    source_only=source_only
                )

            # Runs reverse validation evaluation
            else:
                run_reverse_val(
                    model_class,
                    model_params,
                    data,
                    results,
                    target_labels=p,
                    eval=True,
                    num_exps=num_exps,
                    main_header=main_header,
                    sub_header=sub_header
                )