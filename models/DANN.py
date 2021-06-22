from itertools import chain

import torch
import torchmetrics
from torch import nn

from models.base_model import BaseModel
from models.builder import build_discriminator
from util.kl_loss import KL_Loss
from util.reverse_layer import ReverseLayer


class DANNModel(BaseModel):
    def __init__(
        self,
        classes=65,
        domains=2,
        use_KL=True,
        two_step_training=True,
        risk_lambda=0.1,
        rep_lambda=0.1,
        lr=1e-4
    ):
        super().__init__(
            classes=classes,
            domains=domains,
            use_KL=use_KL,
            risk_lambda=risk_lambda,
            lr=lr,
            two_step_training=two_step_training
        )
        self.discriminator = build_discriminator(512, domains)
        self.rep_lambda = rep_lambda

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, inputs):
        features = self.encoder(inputs)
        classes = self.classifier(features)
        if self.use_KL:
            classes = torch.reshape(classes, (-1, self.domains, self.classes))
        domains = self.discriminator(ReverseLayer.apply(features, self.rep_lambda))
        # domains = torch.squeeze(domains)
        return classes, domains

    def configure_optimizers(self):
        if self.use_KL:
            enc_dis_opt = torch.optim.Adam(chain(self.encoder.parameters(), self.discriminator.parameters()), lr=self.lr)
            cls_opt = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
            return [enc_dis_opt, cls_opt], []
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

    def training_epoch_end(self, outputs) -> None:
        self.log("train_acc", self.train_acc.compute())

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        (l_imgs, labels), nl_imgs = batch
        class_labels = labels[:, 1]
        joint_labels = labels[:, 0] * self.classes + labels[:, 1]
        domain_labels = labels[:, 0]

        # Feedforward
        y_pred, l_domain_pred = self(l_imgs)
        nl_pred, nl_domain_pred = self(nl_imgs)

        if self.use_KL:
            if self.two_step_training:
                # Two step training of encoder/discriminator and classifier with KL-loss
                if optimizer_idx == 0:
                    # Encoder/discriminator training
                    y_class = torch.sum(y_pred, 1)

                    # Calculate losses and metrics
                    class_pred_loss = torch.nn.NLLLoss()(y_class, class_labels)
                    kl_loss = KL_Loss(y_pred, self.classes) * self.risk_lambda
                    kl_loss += KL_Loss(nl_pred, self.classes) * self.risk_lambda
                    dis_loss = torch.nn.CrossEntropyLoss()(l_domain_pred, domain_labels)
                    dis_loss += torch.nn.CrossEntropyLoss()(
                        nl_domain_pred,
                        (torch.ones(nl_domain_pred.shape[0]) * (self.domains - 1)).type_as(nl_domain_pred).long()
                    )
                    loss = class_pred_loss + kl_loss + dis_loss

                    self.train_acc(nn.functional.softmax(y_class, dim=1), class_labels)

                    self.log_dict(
                        {
                            "train_encoder_loss": class_pred_loss.float(),
                            "train_KL_loss": kl_loss.float(),
                            "train_discriminator_loss": dis_loss.float(),
                        },
                        on_step=False,
                        on_epoch=True
                    )

                    return loss

                if optimizer_idx == 1:
                    # Classifier training
                    y_joint = torch.reshape(y_pred, (-1, self.domains * self.classes))

                    # Calculate losses and metrics
                    classifier_loss = torch.nn.NLLLoss()(y_joint, joint_labels)

                    self.log_dict(
                        {
                            "train_classifier_loss": classifier_loss.float()
                        },
                        on_step=False,
                        on_epoch=True
                    )

                    return classifier_loss

            else:
                # Single step training
                y_class = torch.sum(y_pred, 1)

                # Calculate losses and metrics
                class_pred_loss = torch.nn.NLLLoss()(y_class, class_labels)
                kl_loss = KL_Loss(y_pred, self.classes) * self.risk_lambda
                kl_loss += KL_Loss(nl_pred, self.classes) * self.risk_lambda
                dis_loss = torch.nn.CrossEntropyLoss()(l_domain_pred, domain_labels)
                dis_loss += torch.nn.CrossEntropyLoss()(
                    nl_domain_pred,
                    (torch.ones(nl_domain_pred.shape[0]) * (self.domains - 1)).type_as(nl_domain_pred).long()
                )
                loss = class_pred_loss + kl_loss + dis_loss

                self.train_acc(nn.functional.softmax(y_class, dim=1), class_labels)

                self.log_dict(
                    {
                        "train_encoder_loss": class_pred_loss.float(),
                        "train_KL_loss": kl_loss.float(),
                        "train_discriminator_loss": dis_loss.float(),
                    },
                    on_step=False,
                    on_epoch=True
                )

                return loss

        else:
            # Calculate losses and metrics
            class_pred_loss = torch.nn.NLLLoss()(y_pred, class_labels)
            dis_loss = torch.nn.CrossEntropyLoss()(l_domain_pred, domain_labels)
            dis_loss += torch.nn.CrossEntropyLoss()(
                nl_domain_pred,
                (torch.ones(nl_domain_pred.shape[0]) * (self.domains - 1)).type_as(nl_domain_pred).long()
            )
            loss = class_pred_loss + dis_loss

            self.train_acc(nn.functional.softmax(y_pred, dim=1), class_labels)

            self.log_dict(
                {
                    "train_loss": class_pred_loss.float(),
                    "train_discriminator_loss": dis_loss.float()
                },
                on_step=False,
                on_epoch=True
            )

            return loss

    def validation_epoch_end(self, outputs) -> None:
        self.log("val_acc", self.val_acc.compute())

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        class_labels = labels[:, 1]
        domain_labels = labels[:, 0]

        # Feedforward
        y_pred, domain_pred = self(imgs)

        if self.use_KL:
            y_class = torch.sum(y_pred, 1)

            # Calculate losses and metrics
            class_pred_loss = torch.nn.NLLLoss()(y_class, class_labels)
            kl_loss = KL_Loss(y_pred, self.classes) * self.risk_lambda
            dis_loss = torch.nn.CrossEntropyLoss()(domain_pred, domain_labels)
            loss = class_pred_loss + kl_loss + dis_loss

            self.val_acc(nn.functional.softmax(y_class, dim=1), class_labels)

            self.log_dict(
                {
                    "val_loss": class_pred_loss.float(),
                    "val_KL_loss": kl_loss.float(),
                    "val_discriminator_loss": dis_loss.float()
                },
                on_step=False,
                on_epoch=True
            )

            return loss

        else:
            class_pred_loss = torch.nn.NLLLoss()(y_pred, class_labels)
            dis_loss = torch.nn.CrossEntropyLoss()(domain_pred, domain_labels)
            loss = class_pred_loss + dis_loss

            self.val_acc(nn.functional.softmax(y_pred, dim=1), class_labels)

            self.log_dict(
                {
                    "val_loss": class_pred_loss.float(),
                    "val_discriminator_loss": dis_loss.float()
                },
                on_step=False,
                on_epoch=True
            )

            return loss

    def test_epoch_end(self, outputs) -> None:
        self.log("test_acc", self.test_acc.compute())

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        class_labels = labels[:, 1]
        domain_labels = labels[:, 0]

        # Feedforward
        y_pred, domain_pred = self(imgs)

        if self.use_KL:
            y_class = torch.sum(y_pred, 1)

            # Calculate losses and metrics
            class_pred_loss = torch.nn.NLLLoss()(y_class, class_labels)
            kl_loss = KL_Loss(y_pred, self.classes) * self.risk_lambda
            dis_loss = torch.nn.CrossEntropyLoss()(domain_pred, domain_labels)
            loss = class_pred_loss + kl_loss + dis_loss

            self.test_acc(nn.functional.softmax(y_class, dim=1), class_labels)

            self.log_dict(
                {
                    "test_loss": class_pred_loss.float(),
                    "test_KL_loss": kl_loss.float(),
                    "test_discriminator_loss": dis_loss.float()
                },
                on_step=False,
                on_epoch=True
            )

            return loss

        else:
            class_pred_loss = torch.nn.NLLLoss()(y_pred, class_labels)
            dis_loss = torch.nn.CrossEntropyLoss()(domain_pred, domain_labels)
            loss = class_pred_loss + dis_loss

            self.test_acc(nn.functional.softmax(y_pred, dim=1), class_labels)

            self.log_dict(
                {
                    "test_loss": class_pred_loss.float(),
                    "test_discriminator_loss": dis_loss.float()
                },
                on_step=False,
                on_epoch=True
            )

            return loss
