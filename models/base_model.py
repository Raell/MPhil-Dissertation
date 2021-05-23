import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import torchmetrics
from models.builder import build_encoder, build_classifier
from util.kl_loss import KL_Loss
import pytorch_lightning as pl


class SingleDomainModel(pl.LightningModule):
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

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, inputs):
        features = self.encoder(inputs)
        classes = self.classifier(features)
        if self.use_KL:
            classes = torch.reshape(classes, (-1, 2, self.classes))
        return classes

    def configure_optimizers(self):
        if self.use_KL:
            enc_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
            cls_opt = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
            return [enc_opt, cls_opt], []
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

    def training_epoch_end(self, outputs) -> None:
        self.log("train_acc", self.train_acc.compute())

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if self.use_KL:
            (l_imgs, labels), nl_imgs = batch
            class_labels = labels[:, 1]
            joint_labels = labels[:, 0] * self.classes + labels[:, 1]

            # Feedforward
            y_pred = self(l_imgs)
            nl_y_pred = self(nl_imgs)

            # Two step training of encoder and classifier with KL-loss
            if optimizer_idx == 0:
                # Encoder training
                y_class = torch.sum(y_pred, 1)

                # Calculate losses and metrics
                class_pred_loss = torch.nn.CrossEntropyLoss()(y_class, class_labels)
                kl_loss = KL_Loss(y_pred, self.classes) * self.risk_lambda
                kl_loss += KL_Loss(nl_y_pred, self.classes) * self.risk_lambda
                encoder_loss = class_pred_loss + kl_loss

                self.train_acc(nn.functional.softmax(y_class, dim=1), class_labels)

                self.log_dict(
                    {
                        "train_encoder_loss": class_pred_loss.float(),
                        "train_KL_loss": kl_loss.float(),
                    },
                    on_step=False,
                    on_epoch=True
                )

                return encoder_loss

            if optimizer_idx == 1:
                # Classifier training
                y_joint = torch.reshape(y_pred, (-1, 2 * self.classes))

                # Calculate losses and metrics
                classifier_loss = torch.nn.CrossEntropyLoss()(y_joint, joint_labels)

                self.log_dict(
                    {
                        "train_classifier_loss": classifier_loss.float()
                    },
                    on_step=False,
                    on_epoch=True
                )

                return classifier_loss

        else:
            imgs, labels = batch
            class_labels = labels[:, 1]

            # Feedforward
            y_pred = self(imgs)

            # Calculate losses and metrics
            loss = torch.nn.CrossEntropyLoss()(y_pred, class_labels)

            self.train_acc(nn.functional.softmax(y_pred, dim=1), class_labels)

            self.log_dict(
                {
                    "train_loss": loss.float()
                },
                on_step=False,
                on_epoch=True
            )

            return loss

    def validation_epoch_end(self, outputs) -> None:
        self.log("val_acc", self.val_acc.compute())

    def validation_step(self, batch, batch_idx):
        if self.use_KL:
            (l_imgs, labels), nl_imgs = batch
            class_labels = labels[:, 1]

            # Feedforward
            y_pred = self(l_imgs)
            nl_y_pred = self(nl_imgs)

            y_class = torch.sum(y_pred, 1)

            class_pred_loss = torch.nn.CrossEntropyLoss()(y_class, class_labels)
            kl_loss = KL_Loss(y_pred, self.classes) * self.risk_lambda
            kl_loss += KL_Loss(nl_y_pred, self.classes) * self.risk_lambda

            loss = class_pred_loss + kl_loss

            self.val_acc(nn.functional.softmax(y_class, dim=1), class_labels)

            self.log_dict(
                {
                    "val_loss": loss.float(),
                    "val_KL_loss": kl_loss.float(),
                },
                on_step=False,
                on_epoch=True
            )

            return loss

        else:
            imgs, labels = batch
            class_labels = labels[:, 1]

            # Feedforward
            y_pred = self(imgs)

            loss = torch.nn.CrossEntropyLoss()(y_pred, class_labels)

            self.val_acc(nn.functional.softmax(y_pred, dim=1), class_labels)

            self.log_dict(
                {
                    "val_loss": loss.float()
                },
                on_step=False,
                on_epoch=True
            )

            return loss

    def test_epoch_end(self, outputs) -> None:
        self.log("test_acc", self.test_acc.compute())

    def test_step(self, batch, batch_idx):
        if self.use_KL:
            (l_imgs, labels), nl_imgs = batch
            class_labels = labels[:, 1]

            # Feedforward
            y_pred = self(l_imgs)
            nl_y_pred = self(nl_imgs)

            y_class = torch.sum(y_pred, 1)

            class_pred_loss = torch.nn.CrossEntropyLoss()(y_class, class_labels)
            kl_loss = KL_Loss(y_pred, self.classes) * self.risk_lambda
            kl_loss += KL_Loss(nl_y_pred, self.classes) * self.risk_lambda

            loss = class_pred_loss + kl_loss

            self.test_acc(nn.functional.softmax(y_class, dim=1), class_labels)

            self.log_dict(
                {
                    "test_loss": loss.float(),
                    "test_KL_loss": kl_loss.float(),
                },
                on_step=False,
                on_epoch=True
            )

            return loss

        else:
            imgs, labels = batch
            class_labels = labels[:, 1]

            # Feedforward
            y_pred = self(imgs)

            loss = torch.nn.CrossEntropyLoss()(y_pred, class_labels)

            self.test_acc(nn.functional.softmax(y_pred, dim=1), class_labels)

            self.log_dict(
                {
                    "test_loss": loss.float()
                },
                on_step=False,
                on_epoch=True
            )

            return loss
