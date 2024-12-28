import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

activate_functions = [nn.LeakyReLU(), nn.LeakyReLU()]


def get_model(model_type: str):
    model_type = model_type.lower()
    type2model = {
        "vallina": SingleModel,
        "iom": InvariantObjectiveModel,
        "com": ConservativeObjectiveModel,
        "roma": RoMAModel,
        "trimentoring": TriMentoringModel,
        "ict": ICTModel,
    }
    assert model_type in type2model.keys(), f"model {model_type} not found"
    return type2model[model_type]


class MultipleModels(nn.Module):
    def __init__(
        self, n_dim, n_obj, hidden_size, train_mode, save_dir=None, save_prefix=None
    ):
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

    def create_models(self, learning_obj):
        model = get_model(self.train_mode)
        new_model = model(
            self.n_dim,
            self.hidden_size,
            which_obj=learning_obj,
            save_dir=self.save_dir,
            save_prefix=self.save_prefix,
        )
        self.obj2model[learning_obj] = new_model

    def set_kwargs(self, device=None, dtype=None):
        for model in self.obj2model.values():
            model.set_kwargs(device=device, dtype=dtype)
            model.to(device=device, dtype=dtype)

    def forward(self, x, forward_objs=None):
        if forward_objs is None:
            forward_objs = list(self.obj2model.keys())
        x = [self.obj2model[obj](x) for obj in forward_objs]
        x = torch.cat(x, dim=1)
        return x


class SingleModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, which_obj, save_dir=None, save_prefix=None
    ):
        super(SingleModel, self).__init__()
        self.n_dim = input_size
        self.n_obj = 1
        self.which_obj = which_obj
        self.activate_functions = activate_functions

        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0]))
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
        layers.append(nn.Linear(hidden_size[len(hidden_size) - 1], 1))

        self.layers = nn.Sequential(*layers)
        self.hidden_size = hidden_size

        self.save_path = os.path.join(save_dir, f"{save_prefix}-{which_obj}.pt")

    def forward(self, x):
        for i in range(len(self.hidden_size)):
            x = self.layers[i](x)
            x = self.activate_functions[i](x)

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


class ConservativeObjectiveModel(SingleModel):
    pass


class InvariantObjectiveModel(nn.Module):
    def __init__(self, n_dim, hidden_size, which_obj, save_dir=None, save_prefix=None):
        super(InvariantObjectiveModel, self).__init__()
        self.n_dim = n_dim

        self.which_obj = which_obj

        self.save_dir = save_dir
        self.save_prefix = save_prefix
        self.save_path = os.path.join(self.save_dir, f"{save_prefix}-{which_obj}.pt")
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        self.discriminator_model = IOMDiscriminatorModel(
            input_size=128, hidden_size=512, final_tanh=False
        )
        self.forward_model = IOMForwardModel(
            input_size=128,
            activations=["relu", "relu"],
            hidden_size=2048,
            final_tanh=False,
        )
        self.representation_model = IOMRepModel(
            input_size=n_dim,
            activations=["relu", "relu"],
            output_shape=128,
            hidden_size=2048,
            final_tanh=False,
        )

    def set_kwargs(self, device=None, dtype=None):
        self.discriminator_model.to(device=device, dtype=dtype)
        self.forward_model.to(device=device, dtype=dtype)
        self.representation_model.to(device=device, dtype=dtype)

    def forward(self, x):
        rep_x = self.representation_model(x)
        rep_x = rep_x / (rep_x.norm(dim=-1, keepdim=True) + 1e-6)
        d_pos_rep = self.forward_model(rep_x)
        return d_pos_rep

    def train(self, mode: bool = True):
        self.representation_model.train(mode)
        self.discriminator_model.train(mode)
        self.forward_model.train(mode)
        return super().train(mode)

    def eval(self):
        self.representation_model.eval()
        self.discriminator_model.eval()
        self.forward_model.eval()
        return super().eval()

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

        checkpoint = {}
        self.set_kwargs(device="cpu")
        checkpoint["representation"] = self.representation_model.state_dict()
        checkpoint["forward"] = self.forward_model.state_dict()
        checkpoint["discriminator"] = self.discriminator_model.state_dict()

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
        self.representation_model.load_state_dict(checkpoint["representation"])
        self.forward_model.load_state_dict(checkpoint["forward"])
        self.discriminator_model.load_state_dict(checkpoint["discriminator"])

        valid_mse = checkpoint["valid_mse"]
        print(
            f"Successfully load trained model from {save_path} "
            f"with valid MSE = {valid_mse}"
        )


class IOMForwardModel(nn.Module):
    def __init__(
        self,
        input_size,
        activations=("relu", "relu"),
        hidden_size=1024,
        final_tanh=False,
    ):
        super(IOMForwardModel, self).__init__()
        self.layers = nn.Sequential()
        activation_functions = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        for act in activations:
            act_fn = (
                activation_functions.get(act, act) if isinstance(act, str) else act()
            )
            self.layers.append(
                nn.Linear(
                    input_size if len(self.layers) == 0 else hidden_size, hidden_size
                )
            )
            self.layers.append(act_fn)
        self.layers.append(nn.Linear(hidden_size, 1))
        if final_tanh:
            self.layers.append(nn.Tanh())

    def forward(self, x):
        return self.layers(x)


class IOMDiscriminatorModel(nn.Module):
    def __init__(self, input_size, hidden_size=1024, final_tanh=False):
        super(IOMDiscriminatorModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() if not final_tanh else nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)


class IOMRepModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_shape,
        activations=("relu", "relu"),
        hidden_size=2048,
        final_tanh=False,
    ):
        super(IOMRepModel, self).__init__()
        self.layers = nn.Sequential()
        activation_functions = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        for act in activations:
            act_fn = (
                activation_functions.get(act, act) if isinstance(act, str) else act()
            )
            self.layers.append(
                nn.Linear(
                    input_size if len(self.layers) == 0 else hidden_size, hidden_size
                )
            )
            self.layers.append(act_fn)
        self.layers.append(nn.Linear(hidden_size, np.prod(output_shape)))
        if final_tanh:
            self.layers.append(nn.Tanh())

    def forward(self, x):
        return self.layers(x)


class RoMADoubleHeadModel(nn.Module):
    def __init__(self, input_size, which_obj, hidden_size=64):
        super(RoMADoubleHeadModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.which_obj = which_obj

        self.max_logstd = nn.Parameter(
            torch.full((1, 1), np.log(0.2).astype(np.float32)), requires_grad=True
        )
        self.min_logstd = nn.Parameter(
            torch.full((1, 1), np.log(0.1).astype(np.float32)), requires_grad=True
        )

        self.dense_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, 2),
        )

        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.float()
        pred = self.dense_layers(x)

        mean, logstd = torch.chunk(pred, 2, dim=-1)

        logstd = self.max_logstd - self.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + self.softplus(logstd - self.min_logstd)

        return mean, logstd

    def get_params(self, x):
        mean, logstd = self(x)

        scale = torch.exp(logstd)
        return {"loc": mean, "scale": scale}

    def get_distribution(self, x):
        params = self.get_params(x)
        return Normal(**params)


class RoMAModel(nn.Module):
    def __init__(self, n_dim, hidden_size, which_obj, save_dir=None, save_prefix=None):
        super(RoMAModel, self).__init__()
        self.n_dim = n_dim

        self.which_obj = which_obj

        self.save_dir = save_dir
        self.save_prefix = save_prefix
        self.save_path = os.path.join(self.save_dir, f"{save_prefix}-{which_obj}.pt")
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        self.forward_model = RoMADoubleHeadModel(
            input_size=n_dim, which_obj=which_obj, hidden_size=64
        )

    def set_kwargs(self, device=None, dtype=None):
        self.forward_model.to(device=device, dtype=dtype)

    def forward(self, x):
        return self.forward_model.get_distribution(x).mean

    def train(self, mode: bool = True):
        self.forward_model.train(mode)
        return super().train(mode)

    def eval(self):
        self.forward_model.eval()
        return super().eval()

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

        checkpoint = {}
        self.set_kwargs(device="cpu")
        checkpoint["forward"] = self.forward_model.state_dict()

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
        self.forward_model.load_state_dict(checkpoint["forward"])

        valid_mse = checkpoint["valid_mse"]
        print(
            f"Successfully load trained model from {save_path} "
            f"with valid MSE = {valid_mse}"
        )


class TriMentoringBaseModel(SingleModel):
    def __init__(
        self, seed, input_size, hidden_size, which_obj, save_dir=None, save_prefix=None
    ):
        super(TriMentoringBaseModel, self).__init__(
            input_size, hidden_size, which_obj, save_dir, save_prefix
        )
        self.seed = seed
        self.activate_functions = [F.relu, F.relu]


class TriMentoringModel(nn.Module):
    def __init__(
        self,
        n_dim,
        hidden_size,
        which_obj,
        train_seeds=[2024, 2025, 2026],
        save_dir=None,
        save_prefix=None,
    ):
        super(TriMentoringModel, self).__init__()
        self.n_dim = n_dim
        self.which_obj = which_obj

        self.save_dir = save_dir
        self.save_prefix = save_prefix
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, f"{save_prefix}-{which_obj}.pt")

        from utils import now_seed, set_seed

        initial_seed = now_seed
        self.models = []
        self.train_seeds = train_seeds
        for seed in train_seeds:
            set_seed(seed)
            self.models.append(
                TriMentoringBaseModel(
                    seed=seed,
                    input_size=n_dim,
                    hidden_size=hidden_size,
                    which_obj=which_obj,
                    save_dir=save_dir,
                    save_prefix=save_prefix,
                )
            )
        set_seed(initial_seed)

    def set_kwargs(self, device=None, dtype=None):
        for model in self.models:
            model.to(device=device, dtype=dtype)

    def forward(self, x):
        y = torch.cat([model(x) for model in self.models], dim=1)
        return y.mean(dim=1).reshape(-1, 1)

    def train(self, mode: bool = True):
        for model in self.models:
            model.train(mode)
        return super().train(mode)

    def eval(self):
        for model in self.models:
            model.eval()
        return super().eval()

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

        checkpoint = {}
        self.set_kwargs(device="cpu")
        for seed, model in zip(self.train_seeds, self.models):
            checkpoint[seed] = model.state_dict()

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
        for seed, model in zip(self.train_seeds, self.models):
            model.load_state_dict(checkpoint[seed])

        valid_mse = checkpoint["valid_mse"]
        print(
            f"Successfully load trained model from {save_path} "
            f"with valid MSE = {valid_mse}"
        )


class ICTBaseModel(TriMentoringBaseModel):
    pass


class ICTModel(TriMentoringModel):
    def __init__(
        self,
        n_dim,
        hidden_size,
        which_obj,
        train_seeds=[2024, 2025, 2026],
        save_dir=None,
        save_prefix=None,
    ):
        super().__init__(
            n_dim, hidden_size, which_obj, train_seeds, save_dir, save_prefix
        )
        from utils import now_seed, set_seed

        initial_seed = now_seed
        self.models = []
        self.train_seeds = train_seeds
        for seed in train_seeds:
            set_seed(seed)
            self.models.append(
                TriMentoringBaseModel(
                    seed=seed,
                    input_size=n_dim,
                    hidden_size=hidden_size,
                    which_obj=which_obj,
                    save_dir=save_dir,
                    save_prefix=save_prefix,
                )
            )
        set_seed(initial_seed)
