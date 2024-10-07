"""
This module contains utility functions for the flow matching models.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def evaluation_fm(
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    name: str = None,
    model_best: torch.nn.Module = None,
    epoch: int = None,
) -> float:
    """
    Evaluate the model on the test set.
    :param test_loader: torch.utils.data.DataLoader: the test data loader
    :param device: torch.device: the device to run the model
    :param name: str: the name of the model
    :param model_best: torch.nn.Module: the best model
    :param epoch: int: the epoch number
    :return: float: the negative log-likelihood
    """
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


def training_fm(
    name: str,
    max_patience: int,
    num_epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Train the model.
    :param name: str: the name of the model
    :param max_patience: int: the maximum patience
    :param num_epochs: int: the number of epochs
    :param model: torch.nn.Module: the model
    :param optimizer: torch.optim.Optimizer: the optimizer
    :param training_loader: torch.utils.data.DataLoader: the training data loader
    :param val_loader: torch.utils.data.DataLoader: the validation data loader
    :param device: torch.device: the device to run the model
    :return: np.ndarray: the negative log-likelihood
    """
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
        loss_val = evaluation_fm(val_loader, model_best=model, epoch=e, device=device)
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
    """
    Dataset class for the designs.
    """

    def __init__(self, designs: np.ndarray):
        """
        Initialize the dataset.
        :param designs: np.ndarray: the designs
        """
        self.X = designs

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        :return: int: the length of the dataset
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.X[idx]
