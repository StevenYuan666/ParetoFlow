"""
This file contains the implementation of the MultipleModels and SingleModel classes.
Referring to the implementation of MultipleModels in
https://github.com/lamda-bbo/offline-moo/blob/main/off_moo_baselines/multiple_models/nets.py
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from paretoflow.predictor_utils import compute_pcc, spearman_correlation

tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}


class SingleModelBaseTrainer(nn.Module):
    """
    The base trainer for the single model.
    """

    def __init__(
        self,
        model: nn.Module,
        which_obj: int,
        proxy_lr: float = 1e-3,
        proxy_lr_decay: float = 0.98,
        n_epochs: int = 200,
    ):
        """
        Initialize the base trainer.
        :param model: nn.Module: the model
        :param which_obj: int: the learning objective
        :param proxy_lr: float: the learning rate
        :param proxy_lr_decay: float: the learning rate decay
        :param n_epochs: int: the number of epochs
        """
        super(SingleModelBaseTrainer, self).__init__()

        self.forward_lr = proxy_lr
        self.forward_lr_decay = proxy_lr_decay
        self.n_epochs = n_epochs

        self.model = model

        self.which_obj = which_obj

        self.forward_opt = Adam(model.parameters(), lr=self.forward_lr)
        self.train_criterion = lambda yhat, y: (
            torch.sum(torch.mean((yhat - y) ** 2, dim=1))
        )
        self.mse_criterion = nn.MSELoss()

    def _evaluate_performance(
        self,
        statistics: dict,
        epoch: int,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> dict:
        self.model.eval()
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
                outputs = self.model(batch_x)
                outputs_all = torch.cat((outputs_all, outputs), dim=0)

            train_mse = self.mse_criterion(outputs_all, y_all)
            train_corr = spearman_correlation(outputs_all, y_all)
            train_pcc = compute_pcc(outputs_all, y_all)

            statistics[f"model_{self.which_obj}/train/mse"] = train_mse.item()
            for i in range(self.n_obj):
                statistics[f"model_{self.which_obj}/train/rank_corr_{i + 1}"] = (
                    train_corr[i].item()
                )

            print(
                "Epoch [{}/{}], MSE: {:}, PCC: {:}".format(
                    epoch + 1, self.n_epochs, train_mse.item(), train_pcc.item()
                )
            )

        with torch.no_grad():
            y_all = torch.zeros((0, self.n_obj)).to(**tkwargs)
            outputs_all = torch.zeros((0, self.n_obj)).to(**tkwargs)

            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)

                y_all = torch.cat((y_all, batch_y), dim=0)
                outputs = self.model(batch_x)
                outputs_all = torch.cat((outputs_all, outputs))

            val_mse = self.mse_criterion(outputs_all, y_all)
            val_corr = spearman_correlation(outputs_all, y_all)
            val_pcc = compute_pcc(outputs_all, y_all)

            statistics[f"model_{self.which_obj}/valid/mse"] = val_mse.item()
            for i in range(self.n_obj):
                statistics[f"model_{self.which_obj}/valid/rank_corr_{i + 1}"] = (
                    val_corr[i].item()
                )

            print(
                "Valid MSE: {:}, Valid PCC: {:}".format(val_mse.item(), val_pcc.item())
            )

            if val_pcc.item() > self.min_pcc:
                print("🌸 New best epoch! 🌸")
                self.min_pcc = val_pcc.item()
                self.model.save(val_pcc=self.min_pcc)
        return statistics

    def launch(
        self,
        train_loader: torch.utils.data.DataLoader = None,
        val_loader: torch.utils.data.DataLoader = None,
        retrain_model: bool = True,
    ):

        def update_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        if not retrain_model and os.path.exists(self.model.save_path):
            self.model.load()
            return

        assert train_loader is not None
        assert val_loader is not None

        self.n_obj = None
        self.min_pcc = -1.0
        statistics = {}

        for epoch in range(self.n_epochs):
            self.model.train()
            with tqdm(
                total=len(train_loader),
                desc=f"Training {epoch + 1}/{self.n_epochs}",
                unit="batch",
            ) as pbar:
                losses = []
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(**tkwargs)
                    batch_y = batch_y.to(**tkwargs)
                    if self.n_obj is None:
                        self.n_obj = batch_y.shape[1]

                    self.forward_opt.zero_grad()
                    outputs = self.model(batch_x)
                    loss = self.train_criterion(outputs, batch_y)
                    losses.append(loss.item() / batch_x.size(0))
                    loss.backward()
                    self.forward_opt.step()
                    pbar.set_postfix({"loss": np.array(losses).mean()})
                    pbar.update(1)

            statistics[f"model_{self.which_obj}/train/loss/mean"] = np.array(
                losses
            ).mean()
            statistics[f"model_{self.which_obj}/train/loss/std"] = np.array(
                losses
            ).std()
            statistics[f"model_{self.which_obj}/train/loss/max"] = np.array(
                losses
            ).max()

            self._evaluate_performance(statistics, epoch, train_loader, val_loader)

            statistics[f"model_{self.which_obj}/train/lr"] = self.forward_lr
            self.forward_lr *= self.forward_lr_decay
            update_lr(self.forward_opt, self.forward_lr)


class MultipleModels(nn.Module):
    """
    The multiple models class.
    """

    def __init__(
        self,
        n_dim: int,
        n_obj: int,
        hidden_size: list = [2048, 2048],
        train_mode: str = "Vanilla",
        save_dir: str = None,
        save_prefix: str = None,
    ):
        """
        Initialize the multiple models.
        :param n_dim: int: the number of dimensions
        :param n_obj: int: the number of objectives
        :param hidden_size: list: the hidden size
        :param train_mode: str: the training mode
        :param save_dir: str: the save directory
        :param save_prefix: str: the save prefix
        """
        super(MultipleModels, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj

        self.obj2model = {}
        self.hidden_size = hidden_size
        self.train_mode = train_mode

        self.save_dir = save_dir
        self.save_prefix = save_prefix
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        for obj in range(self.n_obj):
            self.create_models(obj)

    def create_models(self, learning_obj: int):
        """
        Create the models.
        :param learning_obj: int: the learning objective
        """
        model = SingleModel
        new_model = model(
            self.n_dim,
            self.hidden_size,
            which_obj=learning_obj,
            save_dir=self.save_dir,
            save_prefix=self.save_prefix,
        )
        self.obj2model[learning_obj] = new_model

    def set_kwargs(self, device: torch.device = None, dtype: torch.dtype = None):
        """
        Set the keyword arguments.
        :param device: torch.device: the device
        :param dtype: torch.dtype: the data type
        """
        for model in self.obj2model.values():
            model.set_kwargs(device=device, dtype=dtype)
            model.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, forward_objs: list = None) -> torch.Tensor:
        """
        Forward pass.
        :param x: torch.Tensor: the input tensor
        :param forward_objs: list: the forward objects
        :return: torch.Tensor: the output tensor
        """
        if forward_objs is None:
            forward_objs = list(self.obj2model.keys())
        x = [self.obj2model[obj](x) for obj in forward_objs]
        x = torch.cat(x, dim=1)
        return x

    def save(self):
        """
        Save the model.
        """
        for model in self.obj2model.values():
            model.save()

    def load(self):
        """
        Load the model.
        """
        for model in self.obj2model.values():
            model.load()


class SingleModel(nn.Module):
    """
    The single model class.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: list,
        which_obj: int,
        save_dir: str = None,
        save_prefix: str = None,
    ):
        """
        Initialize the single model.
        :param input_size: int: the input size
        :param hidden_size: list: the hidden size
        :param which_obj: int: the learning objective
        :param save_dir: str: the save directory
        :param save_prefix: str: the save prefix
        """
        super(SingleModel, self).__init__()
        self.n_dim = input_size
        self.n_obj = 1
        self.which_obj = which_obj
        self.activate_functions = [nn.LeakyReLU(), nn.LeakyReLU()]

        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0]))
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
        layers.append(nn.Linear(hidden_size[len(hidden_size) - 1], 1))

        self.layers = nn.Sequential(*layers)
        self.hidden_size = hidden_size

        self.save_path = os.path.join(save_dir, f"{save_prefix}-{which_obj}.pt")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param x: torch.Tensor: the input tensor
        :return: torch.Tensor: the output tensor
        """
        for i in range(len(self.hidden_size)):
            x = self.layers[i](x)
            x = self.activate_functions[i](x)

        x = self.layers[len(self.hidden_size)](x)
        out = x

        return out

    def set_kwargs(self, device: torch.device = None, dtype: torch.dtype = None):
        """
        Set the keyword arguments.
        :param device: torch.device: the device
        :param dtype: torch.dtype: the data type
        """
        self.to(device=device, dtype=dtype)

    def check_model_path_exist(self, save_path: str = None) -> bool:
        """
        Check if the model path exists.
        :param save_path: str: the save path
        :return: bool: whether the model path exists
        """
        assert (
            self.save_path is not None or save_path is not None
        ), "save path should be specified"
        if save_path is None:
            save_path = self.save_path
        return os.path.exists(save_path)

    def save(self, val_pcc: float = None, save_path: str = None):
        """
        Save the model.
        :param val_pcc: float: the validation PCC
        :param save_path: str: the save path
        """
        assert (
            self.save_path is not None or save_path is not None
        ), "save path should be specified"
        if save_path is None:
            save_path = self.save_path

        self = self.to("cpu")
        checkpoint = {
            "model_state_dict": self.state_dict(),
        }
        if val_pcc is not None:
            checkpoint["valid_pcc"] = val_pcc

        torch.save(checkpoint, save_path)
        self = self.to(**tkwargs)

    def load(self, save_path: str = None):
        """
        Load the model.
        :param save_path: str: the save path
        """
        assert (
            self.save_path is not None or save_path is not None
        ), "save path should be specified"
        if save_path is None:
            save_path = self.save_path

        checkpoint = torch.load(save_path)
        self.load_state_dict(checkpoint["model_state_dict"])
        valid_pcc = checkpoint["valid_pcc"]
        print(
            f"Successfully load trained model from {save_path} "
            f"with valid PCC = {valid_pcc}"
        )
