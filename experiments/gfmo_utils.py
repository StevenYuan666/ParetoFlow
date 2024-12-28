import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm

from gfmo_reference_directions import UniformReferenceDirectionFactory
from offline_moo.off_moo_baselines.data import spearman_correlation

tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}

# get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_reference_directions(name, *args, **kwargs):
    REF = {
        "uniform": UniformReferenceDirectionFactory,
        "das-dennis": UniformReferenceDirectionFactory,
    }

    if name not in REF:
        raise Exception("Reference directions factory not found.")

    return REF[name](*args, **kwargs)()


def evaluation(test_loader, name=None, model_best=None, epoch=None):
    # EVALUATION
    if model_best is None:
        # load best performing model
        model_best = torch.load(name + ".model")

    model_best.eval()
    loss = 0.0
    N = 0.0
    # use tqdm for progress bar
    with tqdm(total=len(test_loader), desc="Validation", unit="batch") as pbar:
        for indx_batch, test_batch in enumerate(test_loader):
            test_batch = test_batch.float()
            test_batch = test_batch.to(device)
            loss_t = -model_best.log_prob(test_batch, reduction="sum")
            loss = loss + loss_t.item()
            N = N + test_batch.shape[0]
            pbar.update(1)

    loss = loss / N

    if epoch is None:
        print(f"FINAL LOSS: nll={loss}")
    else:
        print(f"Epoch: {epoch}, val nll={loss}")

    return loss


def training(
    name, max_patience, num_epochs, model, optimizer, training_loader, val_loader
):
    nll_val = []
    best_nll = float("inf")
    patience = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        # use tqdm for progress bar
        epoch_loss = 0
        with tqdm(
            total=len(training_loader),
            desc=f"Training {e + 1}/{num_epochs}",
            unit="batch",
        ) as pbar:
            for indx_batch, batch in enumerate(training_loader):
                batch = batch.float()
                batch = batch.to(device)
                loss = model(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss = epoch_loss + loss.item()
                pbar.set_postfix({"loss": epoch_loss / (indx_batch + 1)})
                pbar.update(1)
        print(f"Epoch: {e}, train nll={epoch_loss / len(training_loader)}")

        # Validation
        loss_val = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting

        if e == 0:
            print("saved!")
            torch.save(model, name + ".model")
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                print("saved!")
                torch.save(model, name + ".model")
                best_nll = loss_val
                patience = 0
            else:
                patience = patience + 1

        if patience > max_patience:
            print(f"Early stopping at epoch {e + 1}!")
            break

    nll_val = np.asarray(nll_val)

    return nll_val


class DesignDataset(Dataset):
    def __init__(self, designs):
        self.X = designs

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


# We don't use permutation tasks in our experiments
SyntheticFunctionDict = {
    "zdt1": "ZDT1-Exact-v0",
    "zdt2": "ZDT2-Exact-v0",
    "zdt3": "ZDT3-Exact-v0",
    "zdt4": "ZDT4-Exact-v0",
    "zdt6": "ZDT6-Exact-v0",
    "omnitest": "OmniTest-Exact-v0",
    "vlmop1": "VLMOP1-Exact-v0",
    "vlmop2": "VLMOP2-Exact-v0",
    "vlmop3": "VLMOP3-Exact-v0",
    "dtlz1": "DTLZ1-Exact-v0",
    "dtlz2": "DTLZ2-Exact-v0",
    "dtlz3": "DTLZ3-Exact-v0",
    "dtlz4": "DTLZ4-Exact-v0",
    "dtlz5": "DTLZ5-Exact-v0",
    "dtlz6": "DTLZ6-Exact-v0",
    "dtlz7": "DTLZ7-Exact-v0",
}

MONASSequenceDict = {
    "c10mop1": "C10MOP1-Exact-v0",
    "c10mop2": "C10MOP2-Exact-v0",
    "c10mop3": "C10MOP3-Exact-v0",
    "c10mop4": "C10MOP4-Exact-v0",
    "c10mop5": "C10MOP5-Exact-v0",
    "c10mop6": "C10MOP6-Exact-v0",
    "c10mop7": "C10MOP7-Exact-v0",
    "c10mop8": "C10MOP8-Exact-v0",
    "c10mop9": "C10MOP9-Exact-v0",
    "in1kmop1": "IN1KMOP1-Exact-v0",
    "in1kmop2": "IN1KMOP2-Exact-v0",
    "in1kmop3": "IN1KMOP3-Exact-v0",
    "in1kmop4": "IN1KMOP4-Exact-v0",
    "in1kmop5": "IN1KMOP5-Exact-v0",
    "in1kmop6": "IN1KMOP6-Exact-v0",
    "in1kmop7": "IN1KMOP7-Exact-v0",
    "in1kmop8": "IN1KMOP8-Exact-v0",
    "in1kmop9": "IN1KMOP9-Exact-v0",
}

MONASLogitsDict = {
    "nb201_test": "NASBench201Test-Exact-v0",
}

MOCOContinuousDict = {"portfolio": "Portfolio-Exact-v0"}

MORLDict = {
    "mo_swimmer_v2": "MOSwimmerV2-Exact-v0",
    "mo_hopper_v2": "MOHopperV2-Exact-v0",
}

ScientificDesignContinuousDict = {
    "molecule": "Molecule-Exact-v0",
}

ScientificDesignSequenceDict = {
    "regex": "Regex-Exact-v0",
    "zinc": "ZINC-Exact-v0",
    "rfp": "RFP-Exact-v0",
}

RESuiteDict = {
    "re21": "RE21-Exact-v0",
    "re22": "RE22-Exact-v0",
    "re23": "RE23-Exact-v0",
    "re24": "RE24-Exact-v0",
    "re25": "RE25-Exact-v0",
    "re31": "RE31-Exact-v0",
    "re32": "RE32-Exact-v0",
    "re33": "RE33-Exact-v0",
    "re34": "RE34-Exact-v0",
    "re35": "RE35-Exact-v0",
    "re36": "RE36-Exact-v0",
    "re37": "RE37-Exact-v0",
    "re41": "RE41-Exact-v0",
    "re42": "RE42-Exact-v0",
    "re61": "RE61-Exact-v0",
}

SyntheticFunction = list(SyntheticFunctionDict.values())
MONASSequence = list(MONASSequenceDict.values())
MONASLogits = list(MONASLogitsDict.values())
MOCOContinuous = list(MOCOContinuousDict.values())
MORL = list(MORLDict.values())
ScientificDesignContinuous = list(ScientificDesignContinuousDict.values())
ScientificDesignSequence = list(ScientificDesignSequenceDict.values())
RESuite = list(RESuiteDict.values())

MONAS = MONASSequence + MONASLogits
MOCO = MOCOContinuous
ScientificDesign = ScientificDesignContinuous + ScientificDesignSequence

ALLTASKS = SyntheticFunction + MONAS + MOCO + MORL + ScientificDesign + RESuite
ALLTASKSDICT = {
    **SyntheticFunctionDict,
    **MONASSequenceDict,
    **MONASLogitsDict,
    **MOCOContinuousDict,
    **MORLDict,
    **ScientificDesignContinuousDict,
    **ScientificDesignSequenceDict,
    **RESuiteDict,
}

CONTINUOUSTASKS = (
    SyntheticFunction
    + MONASLogits
    + MOCOContinuous
    + MORL
    + ScientificDesignContinuous
    + RESuite
)
SEQUENCETASKS = MONASSequence + ScientificDesignSequence

# Get all keys in the dictionary
all_task_names = list(ALLTASKSDICT.keys())


class SingleModelBaseTrainer(nn.Module):

    def __init__(self, model, which_obj, args):
        super(SingleModelBaseTrainer, self).__init__()
        self.args = args

        self.forward_lr = args.proxies_lr
        self.forward_lr_decay = args.proxies_lr_decay
        self.n_epochs = args.proxies_epochs

        self.model = model

        self.which_obj = which_obj

        self.forward_opt = Adam(model.parameters(), lr=args.proxies_lr)
        self.train_criterion = lambda yhat, y: (
            torch.sum(torch.mean((yhat - y) ** 2, dim=1))
        )
        self.mse_criterion = nn.MSELoss()

    def _evaluate_performance(self, statistics, epoch, train_loader, val_loader):
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
            train_pcc = self.compute_pcc(outputs_all, y_all)

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
            val_pcc = self.compute_pcc(outputs_all, y_all)

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
        train_loader=None,
        val_loader=None,
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

    def compute_pcc(self, valid_preds, valid_labels):
        vx = valid_preds - torch.mean(valid_preds)
        vy = valid_labels - torch.mean(valid_labels)
        pcc = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx**2) + 1e-12) * torch.sqrt(torch.sum(vy**2) + 1e-12)
        )
        return pcc


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
        model = SingleModel
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


activate_functions = [nn.LeakyReLU(), nn.LeakyReLU()]


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

    def save(self, val_pcc=None, save_path=None):
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

    def load(self, save_path=None):
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


def get_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.9,
    batch_size: int = 32,
):

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(**tkwargs)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).to(**tkwargs)

    tensor_dataset = TensorDataset(X, y)
    lengths = [
        int(val_ratio * len(tensor_dataset)),
        len(tensor_dataset) - int(val_ratio * len(tensor_dataset)),
    ]
    train_dataset, val_dataset = random_split(tensor_dataset, lengths)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 4, shuffle=False, drop_last=False
    )

    return train_loader, val_loader
