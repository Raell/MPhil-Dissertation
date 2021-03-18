import torch
import torch.optim as optim
import numpy as np
from collections import defaultdict
import copy
from models.base_model import SingleDomainModel
from models.builder import build_discriminator
from util.kl_loss import KL_Loss
from util.reverse_layer import ReverseLayer
from itertools import chain

class DANNModel(SingleDomainModel):
    def __init__(
        self,
        classes=65,
        use_KL=True,
        risk_lambda=1,
        rep_lambda=1,
        lr = 1e-4
    ):
        super().__init__(
            classes=classes,
            use_KL=use_KL,
            risk_lambda=risk_lambda,
            lr=lr
        )
        self.discriminator = build_discriminator(512)
        self.rep_lambda = rep_lambda

    def forward(self, inputs):
        features = self.encoder(inputs)
        classes = self.classifier(features)
        if self.use_KL:
            classes = torch.reshape(classes, (-1, 2, self.classes))
        domains = self.discriminator(ReverseLayer.apply(features, self.rep_lambda))
        domains = torch.squeeze(domains)
        return classes, domains

    def __train_encoder_discriminator(self, imgs, domain_labels, y_labels=None):
        opt = optim.Adam(chain(self.encoder.parameters(), self.discriminator.parameters()), lr=self.lr)
        opt.zero_grad()

        # Feedforward
        y_pred, domain_pred = self(imgs)
        y_class = torch.sum(y_pred, 1)[:y_labels.shape[0]]

        # Calculate losses and metrics
        class_pred_loss = torch.nn.NLLLoss()(y_class, y_labels) if y_labels is not None else 0
        kl_loss = KL_Loss(y_pred, self.classes) * self.risk_lambda if y_labels is not None else 0
        dis_loss = torch.nn.BCEWithLogitsLoss()(domain_pred, domain_labels)

        loss = class_pred_loss + kl_loss + dis_loss

        # Backpropagation
        loss.backward()
        opt.step()

        return domain_pred, float(loss), float(kl_loss), float(dis_loss)

    def __train_classifier(self, imgs, labels=None):
        classifier_opt = optim.Adam(self.encoder.parameters(), lr=self.lr)
        classifier_opt.zero_grad()

        # Feedforward
        y_pred, _ = self(imgs[:labels.shape[0]])
        y_joint = torch.reshape(y_pred, (-1, 2 * self.classes))

        # Calculate losses and metrics
        joint_pred_loss = torch.nn.NLLLoss()(y_joint, labels)
        classifier_loss = joint_pred_loss

        # Backpropagation
        classifier_loss.backward()
        classifier_opt.step()

        return y_pred, float(classifier_loss)

    def __full_training(self, imgs, domain_labels, y_labels=None):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        opt.zero_grad()

        # Feedforward
        y_pred, domain_pred = self(imgs)
        y_pred = y_pred[:y_labels.shape[0]]

        # Calculate losses and metrics
        class_pred_loss = torch.nn.NLLLoss()(y_pred, y_labels) if y_labels is not None else 0
        dis_loss = torch.nn.BCEWithLogitsLoss()(domain_pred, domain_labels) if y_labels is not None else 0
        loss = class_pred_loss + dis_loss

        # Backpropagation
        loss.backward()
        opt.step()

        return y_pred, domain_pred, float(loss), float(dis_loss)

    def __train_batch(self, imgs, labels=None):
        imgs = imgs.cuda()

        if labels is not None:
            labels = labels.cuda()
            labels = labels.to(dtype=torch.int64)
            class_labels = labels[:, 1]
            joint_labels = labels[:, 0] * self.classes + labels[:, 1]
            domain_labels = torch.cat(
                (
                    labels[:, 0],
                    torch.ones(
                        imgs.shape[0] - labels.shape[0]).cuda()
                 ), 0)
        else:
            joint_labels = None
            class_labels = None
            domain_labels = torch.ones(imgs.shape[0])

        if self.use_KL:
            # Two step training of encoder/discriminator and classifier with KL-loss
            domain_pred, encoder_loss, kl_loss, dis_loss = self.__train_encoder_discriminator(imgs, domain_labels, class_labels)
            pred_class_count = 0
            pred_joint_count = 0

            if labels is not None:
                y_pred, classifier_loss = self.__train_classifier(imgs, joint_labels)

                y_class = torch.sum(y_pred, 1)
                y_joint = torch.reshape(y_pred, (-1, 2 * self.classes))

                pred_class_count = torch.count_nonzero(torch.argmax(y_class, 1) == class_labels)
                pred_joint_count = torch.count_nonzero(torch.argmax(y_joint, 1) == joint_labels)

            pred_dis_count = torch.count_nonzero(torch.round(domain_pred) == domain_labels)

            loss_metrics = {
                "loss": encoder_loss + classifier_loss,
                "encoder_loss": encoder_loss,
                "kl_loss": kl_loss,
                "discriminator_loss": dis_loss,
                "classifier_loss": classifier_loss
            }

            acc_metrics = {
                "joint_acc": float(pred_joint_count),
                "class_acc": float(pred_class_count),
                "dis_acc": float(pred_dis_count)
            }

        else:
            y_pred, domain_pred, loss, dis_loss = self.__full_training(imgs, domain_labels, class_labels)

            pred_class_count = torch.count_nonzero(torch.argmax(y_pred, 1) == class_labels)
            pred_dis_count = torch.count_nonzero(torch.round(domain_pred) == domain_labels)

            loss_metrics = {
                "loss": loss,
                "discriminator_loss": dis_loss
            }

            acc_metrics = {
                "class_acc": float(pred_class_count),
                "dis_acc": float(pred_dis_count)
            }

        return loss_metrics, acc_metrics

    def train_model(self, min_epochs, max_epochs, train_loaders, val_loader=None, patience=None, verbose=True):
        train_label_loader, train_nlabel_loader = train_loaders
        # train_src_loader, train_tar_label_loader, train_tar_nlabel_loader = train_loaders
        min_val_loss = np.inf
        best_model = None

        current_patience = patience

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
            # tar_label_len = len(train_tar_label_loader) if train_tar_label_loader is not None else 0
            # tar_nlabel_len = len(train_tar_nlabel_loader) if train_tar_nlabel_loader is not None else 0
            #
            # len_dataloader = min(src_len, tar_label_len + tar_nlabel_len)
            # src_iter = iter(train_src_loader)
            # tar_label_iter = iter(train_tar_label_loader) if train_tar_label_loader is not None else None
            # tar_nlabel_iter = iter(train_tar_nlabel_loader) if train_tar_nlabel_loader is not None else None

            label_len = len(train_label_loader)
            nlabel_len = len(train_nlabel_loader) if train_nlabel_loader is not None else 0
            len_dataloader = max(label_len, nlabel_len)

            label_iter = iter(train_label_loader)
            nlabel_iter = iter(train_nlabel_loader)

            # Keep count of amount of train data
            train_size = 0
            labelled_size = 0

            for j in range(len_dataloader):
                # src_imgs, src_labels = src_iter.next()
                #
                # if j < tar_label_len:
                #     tar_imgs, tar_labels = tar_label_iter.next()
                #     imgs = torch.cat((src_imgs, tar_imgs), 0)
                #     labels = torch.cat((src_labels, tar_labels), 0)
                # else:
                #     tar_imgs = tar_nlabel_iter.next()
                #     imgs = torch.cat((src_imgs, tar_imgs), 0)
                #     labels = src_labels

                if j < label_len and j < nlabel_len:
                    label_imgs, labels = label_iter.next()
                    nlabel_imgs = nlabel_iter.next()
                    imgs = torch.cat((label_imgs, nlabel_imgs), 0)
                elif j < label_len:
                    imgs, labels = label_iter.next()
                else:
                    imgs = nlabel_iter.next()
                    labels = None


                train_size += imgs.shape[0]
                if labels is not None:
                    labelled_size += labels.shape[0]

                batch_loss, batch_acc = self.__train_batch(imgs, labels)

                for k, v in batch_loss.items():
                    loss_metrics[k] += v
                for k, v in batch_acc.items():
                    acc_metrics[k] += v

            loss_metrics = {k: v / train_size for k, v in loss_metrics.items()}
            for k, v in acc_metrics.items():
                if k == "dis_acc":
                    acc_metrics[k] = v / train_size
                else:
                    acc_metrics[k] = v / labelled_size

            if verbose:
                self._print_metrics(loss_metrics, precision=6)
                self._print_metrics(acc_metrics, precision=4, space=2)

            # Run on validation set if provided
            if val_loader is not None:
                loss_metrics, acc_metrics = self.evaluate(val_loader, val=True, verbose=verbose)

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

    def _evaluate_batch(self, imgs, labels):
        self.eval()

        with torch.no_grad():

            imgs = imgs.cuda()
            labels = labels.cuda()
            class_labels = labels[:, 1].to(dtype=torch.int64)
            domain_labels = labels[:, 0]

            # Feedforward
            y_pred, domain_pred = self(imgs)

            if self.use_KL:
                y_class = torch.sum(y_pred, 1)

                class_pred_loss = float(torch.nn.NLLLoss()(y_class, class_labels))
                kl_loss = float(KL_Loss(y_pred, self.classes) * self.risk_lambda)
                dis_loss = float(torch.nn.BCEWithLogitsLoss()(domain_pred, domain_labels))

                encoder_loss = class_pred_loss + kl_loss + dis_loss

                pred_class_count = torch.count_nonzero(torch.argmax(y_class, 1) == class_labels)
                pred_dis_count = torch.count_nonzero(torch.round(domain_pred) == domain_labels)

                loss_metrics = {
                    "loss": encoder_loss,
                    "kl_loss": kl_loss,
                    "classifier_loss": class_pred_loss,
                    "discriminator_loss": dis_loss,
                }

                acc_metrics = {
                    "class_acc": float(pred_class_count),
                    "dis_acc": float(pred_dis_count)
                }

            else:
                pred_class_count = torch.count_nonzero(torch.argmax(y_pred, 1) == class_labels)
                pred_dis_count = torch.count_nonzero(torch.round(domain_pred) == domain_labels)

                class_pred_loss = float(torch.nn.NLLLoss()(y_pred, class_labels))
                dis_loss = float(torch.nn.BCEWithLogitsLoss()(domain_pred, domain_labels))

                loss = class_pred_loss + dis_loss

                loss_metrics = {
                    "loss": loss,
                    "classifier_loss": class_pred_loss,
                    "discriminator_loss": dis_loss
                }

                acc_metrics = {
                    "class_acc": float(pred_class_count),
                    "dis_acc": float(pred_dis_count)
                }

            return loss_metrics, acc_metrics
