import os
from copy import deepcopy
from typing import Optional

import higher
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from offline_moo.off_moo_baselines.data import spearman_correlation, tkwargs


def get_trainer(train_mode):
    if train_mode.lower() == "com":
        trainer = ConservativeObjectiveTrainer
    elif train_mode.lower() == "iom":
        trainer = InvariantObjectiveTrainer
    elif train_mode.lower() == "roma":
        trainer = RoMATrainer
    elif train_mode.lower() == "trimentoring":
        trainer = TriMentoringTrainer
    elif train_mode.lower() == "ict":
        trainer = ICTTrainer
    else:
        trainer = SingleModelBaseTrainer
    return trainer


class SingleModelBaseTrainer(nn.Module):

    def __init__(self, model, config):
        super(SingleModelBaseTrainer, self).__init__()
        self.config = config

        if config["data_pruning"] and not isinstance(
            config["data_preserved_ratio"], float
        ):
            config["data_preserved_ratio"] = 0.2

        self.forward_lr = config["forward_lr"]
        self.forward_lr_decay = config["forward_lr_decay"]
        self.n_epochs = config["n_epochs"]

        self.use_wandb = config["use_wandb"]
        self.model = model

        self.which_obj = config["which_obj"]

        ################## TODO: to be fixed ################
        try:
            self.forward_opt = Adam(model.parameters(), lr=config["forward_lr"])
        except:
            pass
        #####################################################
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

            statistics[f"model_{self.which_obj}/train/mse"] = train_mse.item()
            for i in range(self.n_obj):
                statistics[f"model_{self.which_obj}/train/rank_corr_{i + 1}"] = (
                    train_corr[i].item()
                )

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
                outputs = self.model(batch_x)
                outputs_all = torch.cat((outputs_all, outputs))

            val_mse = self.mse_criterion(outputs_all, y_all)
            val_corr = spearman_correlation(outputs_all, y_all)

            statistics[f"model_{self.which_obj}/valid/mse"] = val_mse.item()
            for i in range(self.n_obj):
                statistics[f"model_{self.which_obj}/valid/rank_corr_{i + 1}"] = (
                    val_corr[i].item()
                )

            print("Valid MSE: {:}".format(val_mse.item()))

            if len(test_loader) != 0:
                y_all = torch.zeros((0, self.n_obj)).to(**tkwargs)
                outputs_all = torch.zeros((0, self.n_obj)).to(**tkwargs)

                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(**tkwargs)
                    batch_y = batch_y.to(**tkwargs)

                    y_all = torch.cat((y_all, batch_y), dim=0)
                    outputs = self.model(batch_x)
                    outputs_all = torch.cat((outputs_all, outputs))

                test_mse = self.mse_criterion(outputs_all, y_all)
                test_corr = spearman_correlation(outputs_all, y_all)

                statistics[f"model_{self.which_obj}/test/mse"] = test_mse.item()
                for i in range(self.n_obj):
                    statistics[f"model_{self.which_obj}/test/rank_corr_{i + 1}"] = (
                        test_corr[i].item()
                    )

                print("Test MSE: {:}".format(test_mse.item()))

            if val_mse.item() < self.min_mse:
                print("🌸 New best epoch! 🌸")
                self.min_mse = val_mse.item()
                self.model.save(val_mse=self.min_mse)
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

        if not retrain_model and os.path.exists(self.model.save_path):
            self.model.load()
            return

        assert train_loader is not None
        assert val_loader is not None

        self.n_obj = None
        self.min_mse = float("inf")
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

            self._evaluate_performance(
                statistics, epoch, train_loader, val_loader, test_loader
            )

            statistics[f"model_{self.which_obj}/train/lr"] = self.forward_lr
            self.forward_lr *= self.forward_lr_decay
            update_lr(self.forward_opt, self.forward_lr)

            if self.use_wandb:
                statistics[f"model_{self.which_obj}/train_epoch"] = epoch
                wandb.log(statistics)


class ConservativeObjectiveTrainer(SingleModelBaseTrainer):
    def __init__(self, model, config):
        super(ConservativeObjectiveTrainer, self).__init__(model, config)

        alpha = torch.tensor(config["alpha"])
        self.log_alpha = torch.nn.Parameter(torch.log(alpha))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config["alpha_lr"])

        self.overestimation_limit = config["overestimation_limit"]
        self.particle_lr = config["particle_lr"] * np.sqrt(
            np.prod(config["input_shape"])
        )
        self.particle_gradient_steps = config["particle_gradient_steps"]
        self.entropy_coefficient = config["entropy_coefficient"]
        self.noise_std = config["noise_std"]

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def obtain_x_neg(self, x, steps, **kwargs):

        # gradient ascent on the conservatism
        def gradient_step(xt):

            # shuffle the designs for calculating entropy
            indices = torch.randperm(xt.size(0))
            shuffled_xt = xt[indices]

            # entropy using the gaussian kernel
            entropy = torch.mean((xt - shuffled_xt) ** 2)

            # the predicted score according to the forward model
            score = self.model(xt, **kwargs)

            # the conservatism of the current set of particles
            losses = self.entropy_coefficient * entropy + score

            # calculate gradients for each element separately
            grads = torch.autograd.grad(
                outputs=losses, inputs=xt, grad_outputs=torch.ones_like(losses)
            )

            with torch.no_grad():
                xt.data = xt.data - self.particle_lr * grads[0].detach()
                xt.detach_()
                if xt.grad is not None:
                    xt.grad.zero_()
            return xt.detach()

        xt = torch.tensor(x, requires_grad=True).to(**tkwargs)

        for _ in range(steps):
            xt = gradient_step(xt)
            xt.requires_grad = True
        return xt

    def train_step(self, x, y, statistics):
        # corrupt the inputs with noise
        x = x + self.noise_std * torch.randn_like(x).to(**tkwargs)
        x, y = Variable(x, requires_grad=True), Variable(y, requires_grad=False)

        # calculate the prediction error and accuracy of the model
        d_pos = self.model(x)
        mse = F.mse_loss(d_pos.to(**tkwargs), y.to(**tkwargs))

        # calculate negative samples starting from the dataset
        x_neg = self.obtain_x_neg(x, self.particle_gradient_steps)
        # calculate the prediction error and accuracy of the model
        d_neg = self.model(x_neg)
        overestimation = d_pos[:, 0] - d_neg[:, 0]
        statistics[f"model_{self.which_obj}/train/overestimation/mean"] = (
            overestimation.mean()
        )
        statistics[f"model_{self.which_obj}/train/overestimation/std"] = (
            overestimation.std()
        )
        statistics[f"model_{self.which_obj}/train/overestimation/max"] = (
            overestimation.max()
        )

        # build a lagrangian for dual descent
        alpha_loss = (
            self.alpha * self.overestimation_limit - self.alpha * overestimation
        )
        statistics[f"model_{self.which_obj}/train/alpha"] = self.alpha

        # loss that combines maximum likelihood with a constraint
        model_loss = mse + self.alpha * overestimation
        if self.config["data_pruning"]:
            model_loss = model_loss * (1 / self.config["data_preserved_ratio"])
        total_loss = model_loss.mean()
        alpha_loss = alpha_loss.mean()

        # calculate gradients using the model
        alpha_grads = torch.autograd.grad(
            alpha_loss, self.log_alpha, retain_graph=True
        )[0]
        model_grads = torch.autograd.grad(total_loss, self.model.parameters())

        # take gradient steps on the model
        with torch.no_grad():
            self.log_alpha.grad = alpha_grads
            self.alpha_opt.step()
            self.alpha_opt.zero_grad()

            for param, grad in zip(self.model.parameters(), model_grads):
                param.grad = grad
            self.forward_opt.step()
            self.forward_opt.zero_grad()

        return statistics

    def launch(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        retrain_model: bool = True,
    ):

        if not retrain_model and os.path.exists(self.model.save_path):
            self.model.load()
            return

        assert train_loader is not None
        assert val_loader is not None

        self.n_obj = None
        iters = 0
        self.min_mse = float("inf")
        statistics = {}

        for epoch in range(self.n_epochs):

            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)
                if self.n_obj is None:
                    self.n_obj = batch_y.shape[1]

                statistics = self.train_step(batch_x, batch_y, statistics)

            self._evaluate_performance(
                statistics, epoch, train_loader, val_loader, test_loader
            )

            if self.use_wandb:
                statistics[f"model_{self.which_obj}/train_epoch"] = epoch
                wandb.log(statistics)


class InvariantObjectiveTrainer(SingleModelBaseTrainer):

    def __init__(self, model, config):
        nn.Module.__init__(self)
        self.config = config

        if config["data_pruning"] and not isinstance(
            config["data_preserved_ratio"], float
        ):
            config["data_preserved_ratio"] = 0.2

        self.forward_lr = config["forward_lr"]
        self.forward_lr_decay = config["forward_lr_decay"]
        self.n_epochs = config["n_epochs"]

        self.use_wandb = config["use_wandb"]
        self.model = model
        self.forward_model = model.forward_model
        self.rep_model = model.representation_model
        self.discriminator_model = model.discriminator_model

        self.which_obj = config["which_obj"]

        self.forward_opt = Adam(
            self.forward_model.parameters(), lr=config["forward_lr"]
        )
        self.rep_opt = Adam(self.rep_model.parameters(), lr=config["rep_lr"])
        self.discriminator_opt = Adam(
            self.discriminator_model.parameters(),
            lr=config["discriminator_lr"],
            betas=eval(config["discriminator_betas"]),
        )

        # Initialize the alpha parameter and its optimizer
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(config["alpha"]).float()))
        self.alpha_opt = Adam([self.log_alpha], lr=config["alpha_lr"])

        # Set other attributes
        self.mmd_param = config["mmd_param"]
        self.overestimation_limit = config["overestimation_limit"]
        self.particle_lr = config["particle_lr"]
        self.particle_gradient_steps = config["particle_gradient_steps"]
        self.entropy_coefficient = config["entropy_coefficient"]
        self.noise_std = config["noise_std"]

        # Variables for the particles (equivalent to tf.Variable)
        g = config["best_x"]
        self.register_parameter("g", nn.Parameter(g.clone().detach()))
        self.register_parameter("g0", nn.Parameter(g.clone().detach()))

        self.mse_criterion = nn.MSELoss()

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def optimize(self, x, steps, **kwargs):
        # gradient ascent on the conservatism
        for _ in range(steps):
            x.requires_grad_(True)

            # shuffle the designs for calculating entropy
            indices = torch.randperm(x.size(0))
            shuffled_xt = x[indices]

            # entropy using the gaussian kernel
            entropy = torch.mean((x - shuffled_xt) ** 2)

            # the predicted score according to the forward model
            with torch.no_grad():  # rep_model is used in inference mode, no need to track gradients
                xt_rep = self.rep_model(x)
                xt_rep = xt_rep / (
                    torch.sqrt(torch.sum(xt_rep**2, dim=-1, keepdim=True) + 1e-6) + 1e-6
                )
            score = self.forward_model(xt_rep, **kwargs)

            # the conservatism of the current set of particles
            losses = (
                self.entropy_coefficient * entropy + score
            )  # negative because we need gradient ascent

            grads = torch.autograd.grad(
                outputs=losses, inputs=x, grad_outputs=torch.ones_like(losses)
            )

            # update the particles to maximize the conservatism
            with torch.no_grad():  # updates should not be tracked by autograd
                x.data = x.data - self.particle_lr * grads[0].detach()
                x.detach_()  # stop tracking history
                if x.grad is not None:
                    x.grad.zero_()  # reset gradients for the next iteration

        return x

    def train_step(self, x, y, statistics):
        # Corrupt the inputs with noise
        x = x + self.noise_std * torch.randn_like(x)

        # Forward pass and compute gradients for rep_model and forward_model
        rep_x = self.rep_model(x)
        rep_x = rep_x / (rep_x.norm(dim=-1, keepdim=True) + 1e-6)

        d_pos_rep = self.forward_model(rep_x)
        mse = F.mse_loss(y, d_pos_rep)
        statistics["train/mse_L2"] = mse.item()
        mse_l1 = F.l1_loss(y, d_pos_rep)
        statistics["train/mse_L1"] = mse_l1.item()

        # Evaluate how correct the rank of the model predictions are
        rank_corr = spearman_correlation(y[:, 0], d_pos_rep[:, 0])
        statistics["train/rank_corr"] = rank_corr

        # Calculate negative samples starting from the dataset
        x_neg = self.optimize(self.g, 1)
        self.g.data = x_neg.data

        statistics["train/distance_from_start"] = (
            (self.g - self.g0).norm().mean().item()
        )
        x_neg = x_neg[: x.shape[0]]

        # Calculate the prediction error and accuracy of the model
        rep_x_neg = self.rep_model(x_neg)
        rep_x_neg = rep_x_neg / (rep_x_neg.norm(dim=-1, keepdim=True) + 1e-6)

        d_neg_rep = self.forward_model(rep_x_neg)
        overestimation = d_neg_rep[:, 0] - d_pos_rep[:, 0]
        statistics["train/overestimation"] = overestimation.mean().item()
        statistics["train/prediction"] = d_neg_rep.mean().item()

        # Build a Lagrangian for dual descent
        alpha_loss = (
            self.alpha * self.overestimation_limit - self.alpha * overestimation.mean()
        )
        statistics["train/alpha"] = self.alpha.item()

        mmd = F.mse_loss(rep_x.mean(dim=0), rep_x_neg.mean(dim=0))
        statistics["train/mmd"] = mmd.item()

        mmd_before_rep = F.mse_loss(x.mean(dim=0), x_neg.mean(dim=0))
        statistics["train/distance_before_rep"] = mmd_before_rep.item()

        # GAN loss
        valid = torch.ones(rep_x.shape[0], 1, device=x.device)
        fake = torch.zeros(rep_x.shape[0], 1, device=x.device)

        # Discriminator predictions
        dis_rep_x = self.discriminator_model(rep_x.detach()).view(rep_x.shape[0])
        dis_rep_x_neg = self.discriminator_model(rep_x_neg).view(rep_x.shape[0])

        real_loss = F.mse_loss(dis_rep_x, valid.squeeze())
        fake_loss = F.mse_loss(dis_rep_x_neg, fake.squeeze())
        d_loss = (real_loss + fake_loss) / 2
        statistics["train/d_loss"] = d_loss.item()
        statistics["train/real_loss"] = real_loss.item()
        statistics["train/fake_loss"] = fake_loss.item()

        statistics["train/square_dif_x_neg"] = (
            F.mse_loss(rep_x, rep_x_neg).mean().item()
        )

        # Accuracy real and fake
        truth_pos = (self.discriminator_model(rep_x) >= 0.5).float().mean().item()
        statistics["train/accuracy_real"] = truth_pos

        truth_neg = (self.discriminator_model(rep_x_neg) < 0.5).float().mean().item()
        statistics["train/accuracy_fake"] = truth_neg

        mmd_param = self.mmd_param

        model_loss1 = mse - d_loss * mmd_param
        total_loss1 = model_loss1.mean()
        statistics["train/loss1"] = total_loss1.item()

        model_loss2 = mse - d_loss * mmd_param
        total_loss2 = model_loss2.mean()
        # continue from above code
        statistics["train/loss2"] = total_loss2.item()

        # Backward pass and optimize
        self.rep_opt.zero_grad()
        self.forward_opt.zero_grad()
        self.discriminator_opt.zero_grad()

        # Backpropagation for different losses
        total_loss1.backward(retain_graph=True)
        self.forward_opt.step()

        total_loss2.backward(retain_graph=True)
        self.rep_opt.step()

        d_loss.backward(retain_graph=True)
        self.discriminator_opt.step()

        # Update alpha (dual variable)
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        return statistics

    def launch(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        retrain_model: bool = True,
    ):

        if not retrain_model and os.path.exists(self.model.save_path):
            self.model.load()
            return

        assert train_loader is not None
        assert val_loader is not None

        self.n_obj = None
        self.min_mse = float("inf")
        statistics = {}

        for epoch in range(self.n_epochs):

            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)
                if self.n_obj is None:
                    self.n_obj = batch_y.shape[1]

                statistics = self.train_step(batch_x, batch_y, statistics)

            self._evaluate_performance(
                statistics, epoch, train_loader, val_loader, test_loader
            )

            if self.use_wandb:
                statistics[f"model_{self.which_obj}/train_epoch"] = epoch
                wandb.log(statistics)


class RoMATrainer(SingleModelBaseTrainer):
    def __init__(self, model, config):
        super(RoMATrainer, self).__init__(model, config)
        self.forward_model = self.model.forward_model
        self.forward_opt = Adam(
            self.forward_model.parameters(), lr=config["forward_lr"]
        )

        self.is_discrete = "is_discrete" in config.keys() and config["is_discrete"]
        if not self.is_discrete:
            self.perturb_fn = lambda x: x + 0.2 * torch.randn_like(x)
        else:
            self.perturb_fn = lambda x: x

        self.init_sol_x = config["best_x"]
        self.sol_x = (
            config["best_x"].clone().detach().to(**tkwargs).requires_grad_(True)
        )
        self.sol_x_opt = Adam([self.sol_x], lr=config["sol_x_opt_lr"])

        from offline_moo.off_moo_baselines.multiple_models.nets import RoMAModel

        self.temp_model = RoMAModel(
            n_dim=self.model.n_dim,
            hidden_size=64,
            which_obj=self.model.which_obj,
            save_dir=model.save_dir,
            save_prefix=model.save_prefix,
        )
        self.temp_model.set_kwargs(**tkwargs)

        self.coef_stddev = config["coef_stddev"]
        self.sol_x_samples = self.sol_x.size(0)
        self.inner_lr = config["inner_lr"]
        self.prev_sol_x = config["best_x"].clone().detach().to(**tkwargs)
        self.steps_per_update = config["steps_per_update"]
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.region = config["region"]
        self.sol_y = (
            config["best_y"].clone().detach().to(**tkwargs).requires_grad_(True)
        )
        self.temp_x = config["best_x"].clone().detach().requires_grad_(True)

        self.alpha = config["alpha"]
        self.updates = config["updates"]
        self.warmup_epochs = config["warmup_epochs"]

    def get_sol_x(self):
        return self.sol_x

    @torch.enable_grad()
    def train_step(self, x, y, statistics):
        self.model.train()
        if self.perturb_fn is not None:
            x = self.perturb_fn(x)

        self.temp_model.forward_model.load_state_dict(self.forward_model.state_dict())

        x = self.perturb_fn(x)

        for _ in range(self.steps_per_update):
            d = self.forward_model.get_distribution(x)

            loss_total = d.log_prob(y).mean()
            if self.config["data_pruning"]:
                loss_total = loss_total * (1 / self.config["data_preserved_ratio"])
            self.forward_opt.zero_grad()
            loss_total.backward()

            for param, temp_param in zip(
                self.forward_model.parameters(),
                self.temp_model.forward_model.parameters(),
            ):
                if param.grad is not None and param.grad.size(0) > 1:
                    grad_norm = param.grad.norm(2)
                    param_norm = temp_param.data.norm(2)
                    normalized_grad = param.grad * (param_norm / grad_norm)
                    param.data -= (
                        self.inner_lr / self.steps_per_update
                    ) * normalized_grad

        perturbations = {
            name: param - temp_param
            for (name, param), temp_param in zip(
                self.model.named_parameters(), self.temp_model.parameters()
            )
        }

        # Outer optimization
        # Calculate the statistics for the unperturbed model
        d = self.model.forward_model.get_distribution(x)

        loss_nll = -d.log_prob(y).mean()

        # Take gradient steps on the model using the perturbations
        self.forward_opt.zero_grad()
        loss_nll.backward()
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Clip gradients to prevent exploding gradients
                param.grad = torch.clamp(param.grad, -1.0, 1.0)
        self.forward_opt.step()

        for param, (name, perturb) in zip(
            self.model.parameters(), perturbations.items()
        ):
            param.data -= perturbations[name]

        statistics[f"model_{self.which_obj}/train/loss/total"] = loss_total
        statistics[f"model_{self.which_obj}/train/loss/nll"] = loss_nll

        return statistics

    def init_step(self):
        self.temp_model.forward_model.load_state_dict(self.forward_model.state_dict())

    def valid_step(self, x, y, statistics):
        if self.perturb_fn is not None:
            x = self.perturb_fn(x)

        d = self.model.forward_model.get_distribution(x)
        loss_nll = -d.log_prob(y).mean()

        statistics[f"model_{self.which_obj}/valid/loss/nll"] = loss_nll

        return statistics

    @torch.enable_grad()
    def fix_step(self, statistics):
        self.temp_model.set_kwargs(**tkwargs)
        # weight perturbation (current, 1 step)
        for _ in range(self.steps_per_update * 5):
            self.temp_model.train()  # Set the model to training mode
            inp = self.sol_x.to(**tkwargs).requires_grad_(True)
            prev_inp = self.init_sol_x.to(**tkwargs).requires_grad_(True)
            sol_d = self.temp_model.forward_model.get_distribution(inp)
            prev_sol_d = self.temp_model.forward_model.get_distribution(prev_inp)
            loss_sol_x = sol_d.mean - self.coef_stddev * torch.log(sol_d.stddev)

            # Backpropagate the loss
            self.forward_opt.zero_grad()  # Clear gradients
            loss_sol_x.sum().backward(retain_graph=True)  # Compute gradients

            sol_x_grad = self.sol_x.grad
            loss_pessimism_gradnorm = sol_x_grad.view(self.sol_x_samples, -1).norm(
                dim=1
            )
            loss_pessimism_con = -sol_d.log_prob(self.sol_y)
            loss_total = (
                loss_pessimism_gradnorm.mean() + self.alpha * loss_pessimism_con.mean()
            )
            if self.config["data_pruning"]:
                loss_total = loss_total * (1 / self.config["data_preserved_ratio"])

            # Compute gradients with respect to the total loss
            self.forward_opt.zero_grad()  # Clear gradients
            loss_total.backward()  # Compute gradients

            # Update the weights
            for temp_param, origin_param in zip(
                self.temp_model.forward_model.parameters(), self.model.parameters()
            ):
                if (
                    temp_param.grad is not None and temp_param.grad.numel() > 1
                ):  # Equivalent to tf.shape(grad)[0] > 1
                    grad = temp_param.grad * (
                        origin_param.norm() / (temp_param.grad.norm() + 1e-6)
                    )
                    temp_param.data -= self.inner_lr / self.steps_per_update / 5 * grad

        # Calculate pessimism loss for the statistics
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No need to track gradients for this part
            inp = self.sol_x
            prev_inp = self.prev_sol_x
            sol_d = self.temp_model.forward_model.get_distribution(inp)
            prev_sol_d = self.temp_model.forward_model.get_distribution(prev_inp)

            loss_pessimism = (
                sol_d.mean
                - self.coef_stddev * torch.log(sol_d.stddev)
                - prev_sol_d.mean
                - self.coef_stddev * torch.log(prev_sol_d.stddev)
            ) ** 2
            loss_total = loss_pessimism.mean()

        statistics[f"model_{self.which_obj}/fix/loss"] = loss_total.item()

        return statistics

    def update_step(self, statistics):
        self.prev_sol_x.data.copy_(self.sol_x.data)

        # Set the model to evaluation mode
        self.temp_model.eval()
        self.model.eval()

        inp = self.sol_x.to(**tkwargs)
        prev_inp = self.init_sol_x.to(**tkwargs)

        d = self.temp_model.forward_model.get_distribution(inp)
        prev_sol_d = self.forward_model.get_distribution(prev_inp)

        loss_pessimism = (
            d.mean
            - self.coef_stddev * torch.log(d.stddev)
            - prev_sol_d.mean
            - self.coef_stddev * torch.log(prev_sol_d.stddev)
        ) ** 2

        loss = (
            -(d.mean - self.coef_stddev * torch.log(d.stddev))
            + (1.0 / (2.0 * self.region)) * loss_pessimism
        ) * (-1)
        # Times -1 since it is a minimization problem as offline MBO is maxmization

        if self.config["data_pruning"]:
            loss = loss * (1 / self.config["data_preserved_ratio"])

        # Manually zero the gradients after updating weights
        if self.sol_x.grad is not None:
            self.sol_x.grad.detach_()
            self.sol_x.grad.zero_()

        # Backward pass to compute the gradient
        loss.sum().backward(retain_graph=True)
        self.sol_x_opt.step()

        # Get new distribution after the update
        new_d = self.temp_model.forward_model.get_distribution(self.sol_x)
        self.sol_y.data.copy_(new_d.mean)

        # Compute the travelled distance and update statistics
        travelled = torch.norm(self.sol_x - self.init_sol_x) / self.sol_x.size(
            0
        )  # Assuming sol_x is 2D: [samples, features]

        statistics[f"model_{self.which_obj}/update/travelled"] = (
            travelled.item()
        )  # Convert to Python scalar with .item()
        statistics[f"model_{self.which_obj}/update/loss"] = loss.sum().item()

        return statistics

    def launch(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        retrain_model: bool = True,
    ):

        if not retrain_model and os.path.exists(self.model.save_path):
            self.model.load()
            return

        assert train_loader is not None
        assert val_loader is not None

        self.n_obj = None
        self.min_mse = float("inf")
        statistics = {}

        self.n_epochs = self.warmup_epochs
        for epoch in range(self.n_epochs):

            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)
                if self.n_obj is None:
                    self.n_obj = batch_y.shape[1]

                statistics = self.train_step(batch_x, batch_y, statistics)

            self.model.eval()
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)
                statistics = self.valid_step(batch_x, batch_y, statistics)

            self._evaluate_performance(
                statistics, epoch, train_loader, val_loader, test_loader
            )

            if self.use_wandb:
                statistics[f"model_{self.which_obj}/train_epoch"] = epoch
                wandb.log(statistics)

        update = 0
        self.n_epochs = self.updates
        self.min_mse = float("inf")

        while update < self.updates:
            self.init_step()
            statistics = {}
            statistics = self.fix_step(statistics)
            statistics = self.update_step(statistics)

            # self._evaluate_performance(statistics, update,
            #                             train_loader,
            #                             val_loader,
            #                             test_loader)

            update += 1
            # if self.use_wandb:
            #     statistics[f"model_{self.which_obj}/update_epoch"] = update
            #     wandb.log(statistics)
            if update + 1 > self.updates:
                break


class TriMentoringTrainer(SingleModelBaseTrainer):
    def __init__(self, model, config):
        super(TriMentoringTrainer, self).__init__(model, config)
        self.soft_label = config["soft_label"]
        self.majority_voting = config["majority_voting"]
        self.Tmax = config["Tmax"]
        self.ft_lr = config["ft_lr"]
        self.topk = config["topk"]
        self.interval = config["interval"]
        self.K = config["K"]
        self.method = config["method"]

        from utils import now_seed

        self.initial_seed = now_seed

    def train_single_model(
        self,
        model,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
    ):
        from utils import set_seed

        from offline_moo.off_moo_baselines.multiple_models.external.tri_mentoring_utils import (
            adjust_learning_rate,
        )

        set_seed(model.seed)

        lr = self.config["forward_lr"]
        forward_opt = Adam(model.parameters(), lr=lr)
        statistics = {}
        min_mse = float("inf")
        best_state_dict = None
        self.n_obj = None
        for epoch in range(self.n_epochs):
            adjust_learning_rate(forward_opt, lr, epoch, self.n_epochs)

            model.train()
            losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)
                if self.n_obj is None:
                    self.n_obj = batch_y.shape[1]

                forward_opt.zero_grad()
                outputs = model(batch_x)
                loss = self.train_criterion(outputs, batch_y)
                losses.append(loss.item() / batch_x.size(0))
                loss.backward()
                forward_opt.step()

            statistics[f"model_{self.which_obj}_{model.seed}/train/loss/mean"] = (
                np.array(losses).mean()
            )
            statistics[f"model_{self.which_obj}_{model.seed}/train/loss/std"] = (
                np.array(losses).std()
            )
            statistics[f"model_{self.which_obj}_{model.seed}/train/loss/max"] = (
                np.array(losses).max()
            )

            model.eval()
            with torch.no_grad():
                y_all = torch.zeros((0, 1)).to(**tkwargs)
                outputs_all = torch.zeros((0, 1)).to(**tkwargs)
                for (
                    batch_x,
                    batch_y,
                ) in train_loader:
                    batch_x = batch_x.to(**tkwargs)
                    batch_y = batch_y.to(**tkwargs)

                    y_all = torch.cat((y_all, batch_y), dim=0)
                    outputs = model(batch_x)
                    outputs_all = torch.cat((outputs_all, outputs), dim=0)

                train_mse = self.mse_criterion(outputs_all, y_all)
                train_corr = spearman_correlation(outputs_all, y_all)

                statistics[f"model_{self.which_obj}_{model.seed}/train/mse"] = (
                    train_mse.item()
                )
                for i in range(self.n_obj):
                    statistics[
                        f"model_{self.which_obj}_{model.seed}/train/rank_corr_{i + 1}"
                    ] = train_corr[i].item()

                print(
                    "Epoch [{}/{}], MSE: {:}".format(
                        epoch + 1, self.n_epochs, train_mse.item()
                    )
                )

            with torch.no_grad():
                y_all = torch.zeros((0, 1)).to(**tkwargs)
                outputs_all = torch.zeros((0, 1)).to(**tkwargs)

                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(**tkwargs)
                    batch_y = batch_y.to(**tkwargs)

                    y_all = torch.cat((y_all, batch_y), dim=0)
                    outputs = model(batch_x)
                    outputs_all = torch.cat((outputs_all, outputs))

                val_mse = self.mse_criterion(outputs_all, y_all)
                val_corr = spearman_correlation(outputs_all, y_all)

                statistics[f"model_{self.which_obj}_{model.seed}/valid/mse"] = (
                    val_mse.item()
                )
                for i in range(self.n_obj):
                    statistics[
                        f"model_{self.which_obj}_{model.seed}/valid/rank_corr_{i + 1}"
                    ] = val_corr[i].item()

                print("Valid MSE: {:}".format(val_mse.item()))

                if len(test_loader) != 0:
                    y_all = torch.zeros((0, 1)).to(**tkwargs)
                    outputs_all = torch.zeros((0, 1)).to(**tkwargs)

                    for batch_x, batch_y in test_loader:
                        batch_x = batch_x.to(**tkwargs)
                        batch_y = batch_y.to(**tkwargs)

                        y_all = torch.cat((y_all, batch_y), dim=0)
                        outputs = model(batch_x)
                        outputs_all = torch.cat((outputs_all, outputs))

                    test_mse = self.mse_criterion(outputs_all, y_all)
                    test_corr = spearman_correlation(outputs_all, y_all)

                    statistics[f"model_{self.which_obj}_{model.seed}/test/mse"] = (
                        test_mse.item()
                    )
                    for i in range(self.n_obj):
                        statistics[
                            f"model_{self.which_obj}_{model.seed}/test/rank_corr_{i + 1}"
                        ] = test_corr[i].item()

                    print("Test MSE: {:}".format(test_mse.item()))

                if val_mse.item() < min_mse:
                    print("🌸 New best epoch! 🌸")
                    min_mse = val_mse.item()
                    best_state_dict = model.state_dict()

            statistics[f"model_{self.which_obj}_{model.seed}/train/lr"] = (
                forward_opt.param_groups[0]["lr"]
            )

            if self.use_wandb:
                statistics[f"model_{self.which_obj}_{model.seed}/train_epoch"] = epoch
                wandb.log(statistics)

        model.load_state_dict(best_state_dict)
        set_seed(self.initial_seed)

    def launch(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        retrain_model: bool = True,
    ):

        if not retrain_model and os.path.exists(self.model.save_path):
            self.model.load()
            return

        assert train_loader is not None
        assert val_loader is not None

        self.n_obj = None
        self.min_mse = float("inf")
        statistics = {}

        from offline_moo.off_moo_baselines.multiple_models.external.tri_mentoring_utils import (
            adjust_proxy,
        )

        for model in self.model.models:
            self.train_single_model(model, train_loader, val_loader, test_loader)

        best_x = self.config["best_x"]
        best_y = self.config["best_y"]
        indexs = torch.argsort(best_y.squeeze())
        index = indexs[: self.topk]
        x_init = deepcopy(best_x[index])

        models = self.model.models
        proxy1, proxy2, proxy3 = models[0], models[1], models[2]

        self.n_epochs = x_init.shape[0]

        for x_i in range(x_init.shape[0]):
            candidate = x_init[x_i : x_i + 1]
            candidate.requires_grad = True
            candidate_opt = Adam([candidate], lr=self.ft_lr)

            for i in range(1, self.Tmax + 1):
                adjust_proxy(
                    proxy1,
                    proxy2,
                    proxy3,
                    candidate.data,
                    x0=self.config["x"],
                    y0=self.config["y"],
                    K=self.K,
                    majority_voting=self.majority_voting,
                    soft_label=self.soft_label,
                )
                loss = (
                    1.0
                    / 3.0
                    * (proxy1(candidate) + proxy2(candidate) + proxy3(candidate))
                )
                # Reverse since minimization

                candidate_opt.zero_grad()
                loss.backward()
                candidate_opt.step()

            # self._evaluate_performance(statistics, x_i,
            #                            train_loader,
            #                            val_loader,
            #                            test_loader)

            # if self.use_wandb:
            #     statistics[f"model_{self.which_obj}/finetune_epoch"] = x_i
            #     wandb.log(statistics)


class ICTTrainer(TriMentoringTrainer):
    def __init__(self, model, config):
        super(TriMentoringTrainer, self).__init__(model, config)
        self.reweight_mode = config["reweight_mode"]
        self.topk = config["topk"]
        self.ft_lr = config["ft_lr"]
        self.alpha = config["alpha"]
        self.wd = config["wd"]
        self.Tmax = config["Tmax"]
        self.K = config["K"]
        self.noise_coefficient = config["noise_coefficient"]
        self.mu = config["mu"]
        self.std = config["std"]
        self.num_coteaching = config["num_coteaching"]
        self.clamp_norm = config["clamp_norm"]
        self.clamp_min = config["clamp_min"]
        self.clamp_max = config["clamp_max"]
        self.beta = config["beta"]

        from utils import now_seed

        self.initial_seed = now_seed

    # Unpacked Co-teaching Loss function
    def loss_coteaching(self, y_1, y_2, t, num_remember):
        # ind, noise_or_not
        loss_1 = F.mse_loss(y_1, t, reduction="none").view(128)
        ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.mse_loss(y_2, t, reduction="none").view(128)
        ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
        loss_2_sorted = loss_2[ind_2_sorted]

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        # exchange
        loss_1_update = F.mse_loss(y_1[ind_2_update], t[ind_2_update], reduction="none")
        loss_2_update = F.mse_loss(y_2[ind_1_update], t[ind_1_update], reduction="none")

        return loss_1_update, loss_2_update

    def launch(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        retrain_model: bool = True,
    ):

        if not retrain_model and os.path.exists(self.model.save_path):
            self.model.load()
            return

        assert train_loader is not None
        assert val_loader is not None

        self.n_obj = None
        self.min_mse = float("inf")
        statistics = {}

        for model in self.model.models:
            self.train_single_model(model, train_loader, val_loader, test_loader)

        best_x = self.config["best_x"]
        best_y = self.config["best_y"]
        indexs = torch.argsort(best_y.squeeze())

        # get top k candidates
        if self.reweight_mode == "top128":
            index_val = indexs[-self.topk :]
        elif self.reweight_mode == "half":
            index_val = indexs[-(len(indexs) // 2) :]
        else:
            index_val = indexs

        x_val = deepcopy(best_x[index_val])
        label_val = deepcopy(best_y[index_val])
        x_init = deepcopy(best_x[indexs[:1]])

        models = self.model.models
        f1, f2, f3 = models[0], models[1], models[2]

        candidate = x_init[0]  # i.e., x_0
        candidate.requires_grad = True
        candidate_opt = Adam([candidate], lr=self.ft_lr)
        optimizer1 = torch.optim.Adam(
            f1.parameters(), lr=self.alpha, weight_decay=self.wd
        )
        optimizer2 = torch.optim.Adam(
            f2.parameters(), lr=self.alpha, weight_decay=self.wd
        )
        optimizer3 = torch.optim.Adam(
            f3.parameters(), lr=self.alpha, weight_decay=self.wd
        )
        for i in range(1, self.Tmax + 1):
            loss = 1.0 / 3.0 * (f1(candidate) + f2(candidate) + f3(candidate))
            # Reverse since minimization
            candidate_opt.zero_grad()
            loss.backward()
            candidate_opt.step()
            x_train = []
            y1_label = []
            y2_label = []
            y3_label = []
            # sample K points around current candidate
            for k in range(self.K):
                temp_x = candidate.data + self.noise_coefficient * np.random.normal(
                    self.mu, self.std
                )  # add gaussian noise
                x_train.append(temp_x)
                temp_y1 = f1(temp_x)
                y1_label.append(temp_y1)

                temp_y2 = f2(temp_x)
                y2_label.append(temp_y2)

                temp_y3 = f3(temp_x)
                y3_label.append(temp_y3)

            x_train = torch.stack(x_train)
            y1_label = torch.Tensor(y1_label).to(**tkwargs)
            y1_label = torch.reshape(y1_label, (self.K, 1))
            y2_label = torch.Tensor(y2_label).to(**tkwargs)
            y2_label = torch.reshape(y2_label, (self.K, 1))
            y3_label = torch.Tensor(y3_label).to(**tkwargs)
            y3_label = torch.reshape(y3_label, (self.K, 1))

            # Round 1, use f3 to update f1 and f2
            weight_1 = torch.ones(self.num_coteaching).to(**tkwargs)
            weight_1.requires_grad = True
            weight_2 = torch.ones(self.num_coteaching).to(**tkwargs)
            weight_2.requires_grad = True

            with higher.innerloop_ctx(f1, optimizer1) as (model1, opt1):
                with higher.innerloop_ctx(f2, optimizer2) as (model2, opt2):
                    l1, l2 = self.loss_coteaching(
                        model1(x_train), model2(x_train), y3_label, self.num_coteaching
                    )

                    l1_t = weight_1 * l1
                    l1_t = torch.sum(l1_t) / self.num_coteaching
                    opt1.step(l1_t)

                    logit1 = model1(x_val)
                    loss1_v = F.mse_loss(logit1, label_val)
                    g1 = torch.autograd.grad(loss1_v, weight_1)[0].data
                    g1 = F.normalize(g1, p=self.clamp_norm, dim=0)
                    g1 = torch.clamp(g1, min=self.clamp_min, max=self.clamp_max)
                    weight_1 = weight_1 - self.beta * g1
                    weight_1 = torch.clamp(weight_1, min=0, max=2)

                    l2_t = weight_2 * l2
                    l2_t = torch.sum(l2_t) / self.num_coteaching
                    opt2.step(l2_t)

                    logit2 = model2(x_val)
                    loss2_v = F.mse_loss(logit2, label_val)
                    g2 = torch.autograd.grad(loss2_v, weight_2)[0].data
                    g2 = F.normalize(g2, p=self.clamp_norm, dim=0)
                    g2 = torch.clamp(g2, min=self.clamp_min, max=self.clamp_max)
                    weight_2 = weight_2 - self.beta * g2
                    weight_2 = torch.clamp(weight_2, min=0, max=2)

            loss1 = weight_1 * F.mse_loss(f1(x_train), y3_label, reduction="none")
            loss1 = torch.sum(loss1) / self.num_coteaching
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss2 = weight_2 * F.mse_loss(f2(x_train), y3_label, reduction="none")
            loss2 = torch.sum(loss2) / self.num_coteaching
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            # Round 2, use f2 to update f1 and f3
            weight_1 = torch.ones(self.num_coteaching).to(**tkwargs)
            weight_1.requires_grad = True
            weight_3 = torch.ones(self.num_coteaching).to(**tkwargs)
            weight_3.requires_grad = True

            with higher.innerloop_ctx(f1, optimizer1) as (model1, opt1):
                with higher.innerloop_ctx(f3, optimizer3) as (model3, opt3):
                    l1, l3 = self.loss_coteaching(
                        model1(x_train), model3(x_train), y2_label, self.num_coteaching
                    )

                    l1_t = weight_1 * l1
                    l1_t = torch.sum(l1_t) / self.num_coteaching
                    opt1.step(l1_t)

                    logit1 = model1(x_val)
                    loss1_v = F.mse_loss(logit1, label_val)
                    g1 = torch.autograd.grad(loss1_v, weight_1)[0].data
                    g1 = F.normalize(g1, p=self.clamp_norm, dim=0)
                    g1 = torch.clamp(g1, min=self.clamp_min, max=self.clamp_max)
                    weight_1 = weight_1 - self.beta * g1
                    weight_1 = torch.clamp(weight_1, min=0, max=2)

                    l3_t = weight_3 * l3
                    l3_t = torch.sum(l3_t) / self.num_coteaching
                    opt3.step(l3_t)

                    logit3 = model3(x_val)
                    loss3_v = F.mse_loss(logit3, label_val)
                    g3 = torch.autograd.grad(loss3_v, weight_3)[0].data
                    g3 = F.normalize(g3, p=self.clamp_norm, dim=0)
                    g3 = torch.clamp(g3, min=self.clamp_min, max=self.clamp_max)
                    weight_3 = weight_3 - self.beta * g3
                    weight_3 = torch.clamp(weight_3, min=0, max=2)

            loss1 = weight_1 * F.mse_loss(f1(x_train), y2_label, reduction="none")
            loss1 = torch.sum(loss1) / self.num_coteaching
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss3 = weight_3 * F.mse_loss(f3(x_train), y2_label, reduction="none")
            loss3 = torch.sum(loss3) / self.num_coteaching
            optimizer3.zero_grad()
            loss3.backward()
            optimizer3.step()

            # Round 3, use f1 to update f2 and f3
            weight_2 = torch.ones(self.num_coteaching).to(**tkwargs)
            weight_2.requires_grad = True
            weight_3 = torch.ones(self.num_coteaching).to(**tkwargs)
            weight_3.requires_grad = True
            with higher.innerloop_ctx(f2, optimizer2) as (model2, opt2):
                with higher.innerloop_ctx(f3, optimizer3) as (model3, opt3):
                    l2, l3 = self.loss_coteaching(
                        model2(x_train), model3(x_train), y1_label, self.num_coteaching
                    )

                    l2_t = weight_2 * l2
                    l2_t = torch.sum(l2_t) / self.num_coteaching
                    opt2.step(l2_t)

                    logit2 = model2(x_val)
                    loss2_v = F.mse_loss(logit2, label_val)
                    g2 = torch.autograd.grad(loss2_v, weight_2)[0].data
                    g2 = F.normalize(g2, p=self.clamp_norm, dim=0)
                    g2 = torch.clamp(g2, min=self.clamp_min, max=self.clamp_max)
                    weight_2 = weight_2 - self.beta * g2
                    weight_2 = torch.clamp(weight_2, min=0, max=2)

                    l3_t = weight_3 * l3
                    l3_t = torch.sum(l3_t) / self.num_coteaching
                    opt3.step(l3_t)

                    logit3 = model3(x_val)
                    loss3_v = F.mse_loss(logit3, label_val)
                    g3 = torch.autograd.grad(loss3_v, weight_3)[0].data
                    g3 = F.normalize(g3, p=self.clamp_norm, dim=0)
                    g3 = torch.clamp(g3, min=self.clamp_min, max=self.clamp_max)
                    weight_3 = weight_3 - self.beta * g3
                    weight_3 = torch.clamp(weight_3, min=0, max=2)

            loss2 = weight_2 * F.mse_loss(f2(x_train), y1_label, reduction="none")
            loss2 = torch.sum(loss2) / self.num_coteaching
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            loss3 = weight_3 * F.mse_loss(f3(x_train), y1_label, reduction="none")
            loss3 = torch.sum(loss3) / self.num_coteaching
            optimizer3.zero_grad()
            loss3.backward()
            optimizer3.step()
