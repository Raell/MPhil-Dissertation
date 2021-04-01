import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from models.builder import build_encoder, build_classifier
from util.kl_loss import KL_Loss


class SingleDomainModel(nn.Module):
    def __init__(
            self,
            classes=65,
            use_KL=True,
            risk_lambda=1,
            lr=1e-4
    ):
        super().__init__()
        self.classes = classes
        self.encoder = build_encoder()
        self.classifier = build_classifier(512, self.classes, use_KL)

        self.lr = lr

        self.use_KL = use_KL
        self.risk_lambda = risk_lambda

    def forward(self, inputs):
        features = self.encoder(inputs)
        classes = self.classifier(features)
        if self.use_KL:
            classes = torch.reshape(classes, (-1, 2, self.classes))
        return classes

    @staticmethod
    def _print_metrics(metrics, prefix="", precision=4, space=1):
        for k, v in metrics.items():
            print(prefix + k, round(v, precision), sep=": ", end=' ')
        for _ in range(space):
            print()

    def __train_encoder(self, imgs, labels):
        encoder_opt = optim.Adam(self.encoder.parameters(), lr=self.lr)
        encoder_opt.zero_grad()

        # Feedforward
        y_pred = self(imgs)
        y_class = torch.sum(y_pred, 1)

        # Calculate losses and metrics
        class_pred_loss = torch.nn.NLLLoss()(y_class, labels)
        kl_loss = KL_Loss(y_pred, self.classes) * self.risk_lambda
        encoder_loss = class_pred_loss + kl_loss

        # Backpropagation
        encoder_loss.backward()
        encoder_opt.step()

        return y_class, float(encoder_loss), float(kl_loss)

    def __train_classifier(self, imgs, labels):
        classifier_opt = optim.Adam(self.encoder.parameters(), lr=self.lr)
        classifier_opt.zero_grad()

        # Feedforward
        y_pred = self(imgs)
        y_joint = torch.reshape(y_pred, (-1, 2 * self.classes))

        # Calculate losses and metrics
        joint_pred_loss = torch.nn.NLLLoss()(y_joint, labels)
        classifier_loss = joint_pred_loss

        # Backpropagation
        classifier_loss.backward()
        classifier_opt.step()

        return y_pred, float(classifier_loss)

    def __full_training(self, imgs, labels):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        opt.zero_grad()

        # Feedforward
        y_pred = self(imgs)

        # Calculate losses and metrics
        class_pred_loss = torch.nn.NLLLoss()(y_pred, labels)
        loss = class_pred_loss

        # Backpropagation
        loss.backward()
        opt.step()

        return y_pred, float(loss)

    def __train_batch(self, imgs, labels):

        imgs = imgs.cuda()
        labels = labels.cuda()
        labels = labels.to(dtype=torch.int64)
        class_labels = labels[:, 1]
        joint_labels = labels[:, 0] * self.classes + labels[:, 1]

        if self.use_KL:
            # Two step training of encoder and classifier with KL-loss
            _, encoder_loss, kl_loss = self.__train_encoder(imgs, class_labels)
            y_pred, classifier_loss = self.__train_classifier(imgs, joint_labels)

            y_class = torch.sum(y_pred, 1)
            y_joint = torch.reshape(y_pred, (-1, 2 * self.classes))

            pred_class_count = torch.count_nonzero(torch.argmax(y_class, 1) == class_labels)
            pred_joint_count = torch.count_nonzero(torch.argmax(y_joint, 1) == joint_labels)

            loss_metrics = {
                "loss": encoder_loss + classifier_loss,
                "encoder_loss": encoder_loss,
                "kl_loss": kl_loss,
                "classifier_loss": classifier_loss
            }

            acc_metrics = {
                "joint_acc": float(pred_joint_count),
                "class_acc": float(pred_class_count)
            }

        else:
            y_pred, loss = self.__full_training(imgs, class_labels)

            pred_class_count = torch.count_nonzero(torch.argmax(y_pred, 1) == class_labels)

            loss_metrics = {
                "loss": loss
            }

            acc_metrics = {
                "class_acc": float(pred_class_count)
            }

        return loss_metrics, acc_metrics

    def train_model(self, min_epochs, max_epochs, train_loaders, val_loader=None, patience=None, verbose=True):

        # train_src_loader, train_tar_loader, _ = train_loaders
        train_loader, _ = train_loaders
        min_val_loss = np.inf
        best_model = None

        current_patience = patience

        history = {}

        # Perform training over number of epochs
        for i in range(max_epochs):
            self.train()

            if verbose:
                print(f"Epoch {i + 1}")

            loss_metrics = defaultdict(float)
            acc_metrics = defaultdict(float)

            # Prepare train data
            # Each epoch uses the same amount of src and target domain data
            # src_len = len(train_src_loader)
            # tar_len = len(train_tar_loader) if train_tar_loader is not None else src_len
            #
            # len_dataloader = min(src_len, tar_len)
            # src_iter = iter(train_src_loader)
            # tar_iter = iter(train_tar_loader) if train_tar_loader is not None else None

            len_dataloader = len(train_loader)
            train_iter = iter(train_loader)

            # Keep count of amount of train data
            train_size = 0

            for _ in range(len_dataloader):
                imgs, labels = train_iter.next()
                # src_imgs, src_labels = src_iter.next()
                #
                # if tar_iter is not None:
                #     tar_imgs, tar_labels = tar_iter.next()
                #     imgs = torch.cat((src_imgs, tar_imgs), 0)
                #     labels = torch.cat((src_labels, tar_labels), 0)
                # else:
                #     imgs = src_imgs
                #     labels = src_labels

                train_size += imgs.shape[0]
                batch_loss, batch_acc = self.__train_batch(imgs, labels)

                for k, v in batch_loss.items():
                    loss_metrics[k] += v
                for k, v in batch_acc.items():
                    acc_metrics[k] += v
            #
            # loss_metrics = {k: v / train_size for k, v in loss_metrics.items()}
            # acc_metrics = {k: v / train_size for k, v in acc_metrics.items()}

            if "train" in history:
                for k, v in loss_metrics.items():
                    history["train"]["loss_metrics"][k].append(v/train_size)
                for k, v in acc_metrics.items():
                    history["train"]["acc_metrics"][k].append(v/train_size)
            else:
                history["train"] = {
                    "loss_metrics": {
                        k: [v/train_size] for k, v in loss_metrics.items()
                    },
                    "acc_metrics": {
                        k: [v/train_size] for k, v in acc_metrics.items()
                    }
                }

            if verbose:
                self._print_metrics(loss_metrics, precision=6)
                self._print_metrics(acc_metrics, precision=4, space=2)

            # Run on validation set if provided
            if val_loader is not None:
                loss_metrics, acc_metrics = self.evaluate(val_loader, val=True, verbose=verbose)

                if "val" in history:
                    for k, v in loss_metrics.items():
                        history["val"]["loss_metrics"][k].append(v)
                    for k, v in acc_metrics.items():
                        history["val"]["acc_metrics"][k].append(v)
                else:
                    history["val"] = {
                        "loss_metrics": {
                            k: [v / train_size] for k, v in loss_metrics.items()
                        },
                        "acc_metrics": {
                            k: [v / train_size] for k, v in acc_metrics.items()
                        }
                    }

                # Early stopping
                if loss_metrics["classifier_loss"] < min_val_loss:
                    min_val_loss = loss_metrics["classifier_loss"]
                    best_model = copy.deepcopy(self.state_dict())
                    current_patience = patience
                else:
                    if i > min_epochs and current_patience is not None:
                        current_patience -= 1
                        if current_patience <= 0:
                            break

        print("Epochs used: ", i + 1)
        self.load_state_dict(best_model)
        return history

    def evaluate(self, loader, val=False, verbose=True):
        data_size = len(loader.dataset)
        loss_metrics = defaultdict(float)
        acc_metrics = defaultdict(float)

        for imgs, labels in loader:
            batch_loss, batch_acc = self._evaluate_batch(imgs, labels)

            for k, v in batch_loss.items():
                loss_metrics[k] += v / data_size
            for k, v in batch_acc.items():
                acc_metrics[k] += v / data_size

        prefix = "val_" if val else ""

        if verbose:
            self._print_metrics(loss_metrics, prefix=prefix, precision=6)
            self._print_metrics(acc_metrics, prefix=prefix, precision=4, space=2)

        return loss_metrics, acc_metrics

    def _evaluate_batch(self, imgs, labels):
        self.eval()

        with torch.no_grad():
            imgs = imgs.cuda()
            labels = labels.cuda()
            labels = labels.to(dtype=torch.int64)
            class_labels = labels[:, 1]

            # Feedforward
            y_pred = self(imgs)

            if self.use_KL:
                y_class = torch.sum(y_pred, 1)

                class_pred_loss = float(torch.nn.NLLLoss()(y_class, class_labels))
                kl_loss = float(KL_Loss(y_pred, self.classes) * self.risk_lambda)

                encoder_loss = class_pred_loss + kl_loss

                pred_class_count = torch.count_nonzero(torch.argmax(y_class, 1) == class_labels)

                loss_metrics = {
                    "loss": encoder_loss,
                    "kl_loss": kl_loss,
                    "classifier_loss": class_pred_loss
                }

                acc_metrics = {
                    "class_acc": float(pred_class_count)
                }

            else:
                pred_class_count = torch.count_nonzero(torch.argmax(y_pred, 1) == class_labels)
                loss = float(torch.nn.NLLLoss()(y_pred, class_labels))

                loss_metrics = {
                    "loss": loss,
                    "classifier_loss": loss
                }

                acc_metrics = {
                    "class_acc": float(pred_class_count)
                }

            return loss_metrics, acc_metrics
