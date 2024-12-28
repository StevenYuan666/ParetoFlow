import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader

from offline_moo.off_moo_baselines.data import spearman_correlation, tkwargs
from offline_moo.off_moo_baselines.util.pcgrad import PCGrad


def get_trainer(train_mode):
    if train_mode.lower() == "gradnorm":
        trainer = End2EndGradNormTrainer
    elif train_mode.lower() == "pcgrad":
        trainer = End2EndPcGradTrainer
    else:
        trainer = End2EndBaseTrainer
    return trainer


class End2EndBaseTrainer:

    def __init__(self, forward_model, config):

        self.config = config

        if config["data_pruning"] and not isinstance(
            config["data_preserved_ratio"], float
        ):
            config["data_preserved_ratio"] = 0.2

        self.forward_lr = config["forward_lr"]
        self.forward_lr_decay = config["forward_lr_decay"]
        self.n_epochs = config["n_epochs"]

        self.use_wandb = config["use_wandb"]
        self.forward_model = forward_model

        self.forward_opt = Adam(forward_model.parameters(), lr=config["forward_lr"])
        self.train_criterion = lambda yhat, y: (
            torch.sum(torch.mean((yhat - y) ** 2, dim=1))
            * (1 / config["data_preserved_ratio"])
            if config["data_pruning"]
            else torch.sum(torch.mean((yhat - y) ** 2, dim=1))
        )
        self.mse_criterion = nn.MSELoss()

    def _evaluate_performance(
        self, statistics, epoch, train_loader, val_loader, test_loader
    ):
        self.forward_model.eval()
        with torch.no_grad():
            y_all = torch.zeros((0, self.n_obj)).to(**tkwargs)
            outputs_all = torch.zeros((0, self.n_obj)).to(**tkwargs)
            for (
                batch_x,
                batch_y,
            ) in train_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)

                y_all = torch.cat((y_all, batch_y), dim=0)
                outputs = self.forward_model(batch_x)
                outputs_all = torch.cat((outputs_all, outputs), dim=0)

            train_mse = self.mse_criterion(outputs_all, y_all)
            train_corr = spearman_correlation(outputs_all, y_all)

            statistics["train/mse"] = train_mse.item()
            for i in range(self.n_obj):
                statistics[f"train/rank_corr_{i + 1}"] = train_corr[i].item()

            print(
                "Epoch [{}/{}], MSE: {:}".format(
                    epoch + 1, self.n_epochs, train_mse.item()
                )
            )

        with torch.no_grad():
            y_all = torch.zeros((0, self.n_obj)).to(**tkwargs)
            outputs_all = torch.zeros((0, self.n_obj)).to(**tkwargs)

            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)

                y_all = torch.cat((y_all, batch_y), dim=0)
                outputs = self.forward_model(batch_x)
                outputs_all = torch.cat((outputs_all, outputs))

            val_mse = self.mse_criterion(outputs_all, y_all)
            val_corr = spearman_correlation(outputs_all, y_all)

            statistics["valid/mse"] = val_mse.item()
            for i in range(self.n_obj):
                statistics[f"valid/rank_corr_{i + 1}"] = val_corr[i].item()

            print("Valid MSE: {:}".format(val_mse.item()))

            if len(test_loader) != 0:
                y_all = torch.zeros((0, self.n_obj)).to(**tkwargs)
                outputs_all = torch.zeros((0, self.n_obj)).to(**tkwargs)

                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(**tkwargs)
                    batch_y = batch_y.to(**tkwargs)

                    y_all = torch.cat((y_all, batch_y), dim=0)
                    outputs = self.forward_model(batch_x)
                    outputs_all = torch.cat((outputs_all, outputs))

                test_mse = self.mse_criterion(outputs_all, y_all)
                test_corr = spearman_correlation(outputs_all, y_all)

                statistics["test/mse"] = test_mse.item()
                for i in range(self.n_obj):
                    statistics[f"test/rank_corr_{i + 1}"] = test_corr[i].item()

                print("Test MSE: {:}".format(test_mse.item()))

            if val_mse.item() < self.min_mse:
                print("🌸 New best epoch! 🌸")
                self.min_mse = val_mse.item()
                self.forward_model.save(val_mse=self.min_mse)
        return statistics

    def launch(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        retrain_model: bool = True,
    ):

        def update_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        if not retrain_model and os.path.exists(self.forward_model.save_path):
            self.forward_model.load()
            return

        assert train_loader is not None
        assert val_loader is not None

        self.n_obj = None
        self.min_mse = float("inf")
        statistics = {}

        for epoch in range(self.n_epochs):
            self.forward_model.train()

            losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)
                if self.n_obj is None:
                    self.n_obj = batch_y.shape[1]

                self.forward_opt.zero_grad()
                outputs = self.forward_model(batch_x)
                loss = self.train_criterion(outputs, batch_y)
                losses.append(loss.item() / batch_x.size(0))
                loss.backward()
                self.forward_opt.step()

            statistics["train/loss/mean"] = np.array(losses).mean()
            statistics["train/loss/std"] = np.array(losses).std()
            statistics["train/loss/max"] = np.array(losses).max()

            self._evaluate_performance(
                statistics, epoch, train_loader, val_loader, test_loader
            )

            statistics["train/lr"] = self.forward_lr
            self.forward_lr *= self.forward_lr_decay
            update_lr(self.forward_opt, self.forward_lr)

            if self.use_wandb:
                statistics["train_epoch"] = epoch
                wandb.log(statistics)


class End2EndGradNormTrainer(End2EndBaseTrainer):
    def __init__(self, forward_model, config):
        super(End2EndGradNormTrainer, self).__init__(forward_model, config)
        self.alpha = config["gradient_alpha"]
        self.weight_lr = config["weight_lr"]
        self.weight_lr_decay = config["weight_lr_decay"]

        self.norm_layer = forward_model.layers[-1]

    def launch(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        retrain_model: bool = True,
    ):

        def update_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        if not retrain_model and os.path.exists(self.forward_model.save_path):
            self.forward_model.load()
            return

        assert train_loader is not None
        assert val_loader is not None

        self.n_obj = None
        iters = 0
        self.min_mse = float("inf")
        statistics = {}

        for epoch in range(self.n_epochs):
            gradnorm_losses = []
            weighted_losses = []

            self.forward_model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)
                if self.n_obj is None:
                    self.n_obj = batch_y.shape[1]

                self.forward_opt.zero_grad()
                outputs = self.forward_model(batch_x)

                loss = []
                for i in range(batch_y.shape[1]):
                    loss.append(
                        self.mse_criterion(batch_y[:, i].float(), outputs[:, i].float())
                    )

                loss = torch.stack(loss).float()

                if iters == 0:
                    weights = torch.ones_like(loss).float()
                    weights = torch.nn.Parameter(weights)

                    T = weights.sum().detach()
                    self.weight_opt = torch.optim.Adam([weights], lr=self.weight_lr)

                    l0 = loss.detach()

                weighted_loss = weights @ loss

                weighted_loss = weighted_loss.float()
                self.forward_opt.zero_grad()
                weighted_loss.backward(retain_graph=True)

                gw = []
                for i in range(len(loss)):
                    dl = torch.autograd.grad(
                        weights[i] * loss[i],
                        self.norm_layer.parameters(),
                        retain_graph=True,
                        create_graph=True,
                    )[0]
                    gw.append(torch.norm(dl))
                gw = torch.stack(gw)

                loss_ratio = loss.detach() / l0
                rt = loss_ratio / loss_ratio.mean()

                gw_avg = gw.mean().detach()

                constant = (gw_avg * rt**self.alpha).detach()
                gradnorm_loss = torch.abs(gw - constant).sum()
                if self.config["data_pruning"]:
                    gradnorm_loss = gradnorm_loss * (
                        1 / self.config["data_preserved_ratio"]
                    )

                gradnorm_losses.append(gradnorm_loss.item() / batch_x.size(0))
                weighted_losses.append(weighted_loss.item() / batch_x.size(0))

                self.weight_opt.zero_grad()
                gradnorm_loss.backward()

                self.forward_opt.step()
                self.weight_opt.step()

                weights = (weights / weights.sum() * T).detach()
                weights = torch.nn.Parameter(weights)
                self.weight_opt = torch.optim.Adam(
                    [weights], lr=self.config["weight_lr"]
                )

                iters += 1

            statistics["train/loss/mean"] = np.array(gradnorm_losses).mean()
            statistics["train/loss/std"] = np.array(gradnorm_losses).std()
            statistics["train/loss/max"] = np.array(gradnorm_losses).max()

            statistics["train/weight_loss/mean"] = np.array(weighted_losses).mean()
            statistics["train/weight_loss/std"] = np.array(weighted_losses).std()
            statistics["train/weight_loss/max"] = np.array(weighted_losses).max()

            self._evaluate_performance(
                statistics, epoch, train_loader, val_loader, test_loader
            )

            statistics["train/lr"] = self.forward_lr
            self.forward_lr *= self.forward_lr_decay
            update_lr(self.forward_opt, self.forward_lr)

            statistics["train/weight_lr"] = self.weight_lr
            self.weight_lr *= self.weight_lr_decay
            update_lr(self.weight_opt, self.weight_lr)

            if self.use_wandb:
                statistics["train_epoch"] = epoch
                wandb.log(statistics)


class End2EndPcGradTrainer(End2EndBaseTrainer):

    def __init__(self, forward_model, config):
        super(End2EndPcGradTrainer, self).__init__(forward_model, config)
        self.forward_opt = PCGrad(self.forward_opt)

    def launch(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        retrain_model: bool = True,
    ):

        if not retrain_model and os.path.exists(self.forward_model.save_path):
            self.forward_model.load()
            return

        assert train_loader is not None
        assert val_loader is not None

        self.n_obj = None
        self.min_mse = float("inf")
        statistics = {}

        for epoch in range(self.n_epochs):
            self.forward_model.train()

            losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)
                if self.n_obj is None:
                    self.n_obj = batch_y.shape[1]

                self.forward_opt.zero_grad()
                outputs = self.forward_model(batch_x)
                loss = []
                for i in range(batch_y.shape[1]):
                    loss.append(
                        self.mse_criterion(batch_y[:, i].float(), outputs[:, i].float())
                        * (
                            1 / self.config["data_preserved_ratio"]
                            if self.config["data_pruning"]
                            else 1
                        )
                    )
                assert len(loss) == self.n_obj

                losses.append(
                    np.array([single_loss.item() for single_loss in loss]).mean()
                    / batch_x.size(0)
                )
                self.forward_opt.pc_backward(loss)
                self.forward_opt.step()

            statistics["train/loss/mean"] = np.array(losses).mean()
            statistics["train/loss/std"] = np.array(losses).std()
            statistics["train/loss/max"] = np.array(losses).max()

            self._evaluate_performance(
                statistics, epoch, train_loader, val_loader, test_loader
            )

            if self.use_wandb:
                statistics["train_epoch"] = epoch
                wandb.log(statistics)
