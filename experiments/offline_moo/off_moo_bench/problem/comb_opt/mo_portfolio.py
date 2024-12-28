import os

import numpy as np
import pandas as pd
import torch
from pymoo.core.population import Population
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair

from offline_moo.off_moo_bench.problem.base import BaseProblem


class PortfolioBaseProblem(ElementwiseProblem):

    def __init__(self, mu, cov, df, risk_free_rate=0.02, **kwargs):
        super().__init__(n_var=len(df.columns), n_obj=2, xl=0.0, xu=1.0, **kwargs)
        self.mu = mu
        self.cov = cov
        self.risk_free_rate = risk_free_rate

    def _evaluate(self, x, out, *args, **kwargs):
        exp_return = x @ self.mu
        exp_risk = np.sqrt(x.T @ self.cov @ x)
        sharpe = (exp_return - self.risk_free_rate) / exp_risk

        out["F"] = [exp_risk, -exp_return]
        out["sharpe"] = sharpe


class MOPortfolio(BaseProblem):
    def __init__(self, n_obj=2, xl=0, xu=1, n_dim=20):
        super().__init__(
            name=self.__class__.__name__,
            problem_type="comb. opt",
            n_dim=n_dim,
            n_obj=n_obj,
            xl=0,
            xu=xu,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "portfolio_allocation.csv"
        )
        df = pd.read_csv(file, parse_dates=True, index_col="date")

        returns = df.pct_change().dropna(how="all")
        mu = (1 + returns).prod() ** (252 / returns.count()) - 1
        cov = returns.cov() * 252

        mu, cov = mu.to_numpy(), cov.to_numpy()
        self.problem_instance = PortfolioBaseProblem(mu, cov, df)

    def _evaluate(self, x, out, *args, **kwargs):
        # assert 0, x.shape
        y = np.zeros((0, self.n_obj))
        for x0 in x:
            y0 = self.problem_instance.evaluate(x0, out, args, kwargs)
            y = np.concatenate((y, y0.reshape(-1, self.n_obj)), axis=0)

        out["F"] = y

    def get_nadir_point(self):
        return np.array([0.29405603, -0.1361627])

    def get_ideal_point(self):
        return np.array([0.16387525, -0.32456617])


class PortfolioRepair(Repair):

    def _do(self, problem, pop_or_X, **kwargs):
        is_array = not isinstance(pop_or_X, Population)

        X = pop_or_X if is_array else pop_or_X.get("X")

        X[X < 1e-3] = 0
        res = X / X.sum(axis=1, keepdims=True)
        if is_array:
            return res
        else:
            pop_or_X.set("X", res)
            return pop_or_X
