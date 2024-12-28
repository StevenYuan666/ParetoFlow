import os

import torch
import torch.nn as nn

activation_functions = [nn.LeakyReLU(), nn.LeakyReLU()]


class MultiHeadModel(nn.Module):
    def __init__(self, n_dim, n_obj, hidden_size, save_path=None):
        super(MultiHeadModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.activation_functions = [nn.LeakyReLU(), nn.LeakyReLU()]
        self.hidden_size = hidden_size

        self.save_path = save_path
        self.feature_extractor = FeatureExtractor(n_dim, [hidden_size], hidden_size)
        self.obj2head = {}
        for obj in range(self.n_obj):
            self.create_head(obj)

    def create_head(self, learning_obj):
        new_head = Head(
            input_size=self.hidden_size, hidden_size=[], which_obj=learning_obj
        )
        self.obj2head[learning_obj] = new_head

    def set_kwargs(self, device=None, dtype=None):
        self.feature_extractor.to(device=device, dtype=dtype)
        for head in self.obj2head.values():
            head.to(device=device, dtype=dtype)

    def forward(self, x, forward_objs):
        h = self.feature_extractor(x)
        q = [self.obj2head[obj](h) for obj in forward_objs]
        q = torch.cat(q, dim=1)
        return q

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

        self.set_kwargs(device="cpu")
        checkpoint = {
            "feature_extractor": self.feature_extractor.state_dict(),
        }
        for obj, head in self.obj2head.items():
            checkpoint[f"head_{obj}"] = head.state_dict()

        if val_mse is not None:
            checkpoint["valid_mse"] = val_mse

        torch.save(checkpoint, save_path)
        self.set_kwargs(**tkwargs)

    def load(self, save_path=None):
        assert (
            self.save_path is not None or save_path is not None
        ), "save path should be specified"
        if save_path is None:
            save_path = self.save_path

        checkpoint = torch.load(save_path)
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        for obj, head in self.obj2head.items():
            head.load_state_dict(checkpoint[f"head_{obj}"])
        valid_mse = checkpoint["valid_mse"]
        print(
            f"Successfully load trained model from {save_path} "
            f"with valid MSE = {valid_mse}"
        )


class FeatureExtractor(nn.Module):
    def __init__(self, n_dim, hidden_size, output_size):
        super(FeatureExtractor, self).__init__()
        self.n_dim = n_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_functions = [nn.LeakyReLU(), nn.LeakyReLU()]

        layers = []
        layers.append(nn.Linear(n_dim, hidden_size[0]))
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
        layers.append(nn.Linear(hidden_size[len(hidden_size) - 1], output_size))

        self.layers = nn.Sequential(*layers)
        self.hidden_size = hidden_size

    def forward(self, x):
        """
        BUG: It may raise dtype Runtime Error while conducting Multi-Head + MOBO, I suggest using the 'try' part.
        """
        for i in range(len(self.hidden_size) + 1):
            x = self.layers[i](x)
            x = self.activation_functions[i](x)

        return x


class Head(nn.Module):
    def __init__(self, input_size, hidden_size, which_obj):
        super(Head, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1
        self.which_obj = which_obj
        self.activation_functions = [nn.LeakyReLU(), nn.LeakyReLU()]

        layers = []
        if hidden_size != []:
            layers.append(nn.Linear(input_size, hidden_size[0]))
            for i in range(len(hidden_size) - 1):
                layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            layers.append(nn.Linear(hidden_size[len(hidden_size) - 1], 1))
        else:
            layers.append(nn.Linear(input_size, 1))

        self.layers = nn.Sequential(*layers)
        self.hidden_size = hidden_size

    def forward(self, x):
        """
        BUG: It may raise dtype Runtime Error while conducting Multi-Head + MOBO, we suggest using the 'try' part
        """
        x = x
        for i in range(len(self.hidden_size) + 1):
            x = self.layers[i](x)
            x = self.activation_functions[i](x)

        return x
