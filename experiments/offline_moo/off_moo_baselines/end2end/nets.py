import os

import torch
import torch.nn as nn

activate_functions = [nn.LeakyReLU(), nn.LeakyReLU()]


class End2EndModel(nn.Module):
    def __init__(self, n_dim, n_obj, hidden_size, save_path=None):
        super(End2EndModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj

        layers = []
        layers.append(nn.Linear(n_dim, hidden_size[0]))
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
        layers.append(nn.Linear(hidden_size[len(hidden_size) - 1], n_obj))

        self.layers = nn.Sequential(*layers)
        self.hidden_size = hidden_size

        self.save_path = save_path

    def forward(self, x):

        for i in range(len(self.hidden_size)):
            x = self.layers[i](x)
            x = activate_functions[i](x)

        x = self.layers[len(self.hidden_size)](x)
        out = x

        return out

    def set_kwargs(self, device=None, dtype=None):
        self.to(device=device, dtype=dtype)

    def check_model_path_exist(self, save_path=None):
        assert (
            self.save_path is not None or save_path is not None
        ), "save path should be specified"
        if save_path is None:
            save_path = self.save_path
        return os.path.exists(save_path)

    def save(self, val_mse=None, save_path=None):
        assert (
            self.save_path is not None or save_path is not None
        ), "save path should be specified"
        if save_path is None:
            save_path = self.save_path

        from offline_moo.off_moo_baselines.data import tkwargs

        self = self.to("cpu")
        checkpoint = {
            "model_state_dict": self.state_dict(),
        }
        if val_mse is not None:
            checkpoint["valid_mse"] = val_mse

        torch.save(checkpoint, save_path)
        self = self.to(**tkwargs)

    def load(self, save_path=None):
        assert (
            self.save_path is not None or save_path is not None
        ), "save path should be specified"
        if save_path is None:
            save_path = self.save_path

        checkpoint = torch.load(save_path)
        self.load_state_dict(checkpoint["model_state_dict"])
        valid_mse = checkpoint["valid_mse"]
        print(
            f"Successfully load trained model from {save_path} "
            f"with valid MSE = {valid_mse}"
        )
