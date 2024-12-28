import numpy as np
import torch

from offline_moo.off_moo_bench.problem.base import BaseProblem


class DTLZ(BaseProblem):
    def __init__(self, name, n_obj, n_dim, k=None, nadir_point=None, ideal_point=None):

        if n_dim:
            self.k = n_dim - n_obj + 1
        elif k:
            self.k = k
            n_dim = k + n_obj - 1
        else:
            raise Exception("Either provide number of variables or k!")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(
            name,
            problem_type="synthetic",
            n_obj=n_obj,
            n_dim=n_dim,
            xl=0,
            xu=1,
            nadir_point=nadir_point,
            ideal_point=ideal_point,
        )

    def g1(self, X_M):
        # term1 = torch.sum((X_M - 0.5) ** 2 - torch.cos(20 * torch.pi * (X_M - 0.5)), dim=1)
        # result = 100 * (self.k + term1)
        # return result
        return 100 * (
            self.k
            + torch.sum(
                (X_M - 0.5) ** 2 - torch.cos(20 * torch.pi * (X_M - 0.5)), dim=1
            )
        )

    def g2(self, X_M):
        return torch.sum((X_M - 0.5) ** 2, dim=1)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            _f = 1 + g
            _f *= torch.prod(
                torch.cos(torch.pow(X_[:, : X_.shape[1] - i], alpha) * torch.pi / 2.0),
                axis=1,
            )
            if i > 0:
                _f *= torch.sin(
                    torch.pow(X_[:, X_.shape[1] - i], alpha) * torch.pi / 2.0
                )

            f.append(_f)

        f = torch.stack(f)
        return f

    def get_ideal_point(self):
        if self.ideal_point is None:
            raise NotImplementedError
        return self.ideal_point

    def get_nadir_point(self):
        if self.nadir_point is None:
            raise NotImplementedError
        return self.nadir_point


class DTLZ1(DTLZ):
    def __init__(self, n_dim=7, n_obj=3, **kwargs):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=n_obj,
            nadir_point=[507.46516068, 502.09050332, 516.68985527],
            ideal_point=[5.26912179e-09, 1.94578236e-07, 1.76968514e-06],
        )

    def obj_func(self, X_, g):
        f = []

        for i in range(0, self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= torch.prod(X_[:, : X_.shape[1] - i], dim=1)
            if i > 0:
                _f *= 1 - X_[:, X_.shape[1] - i]
            f.append(_f)

        return torch.stack(f, dim=1)

    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.from_numpy(x).to(self.device)
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g).cpu().numpy()


class DTLZ2(DTLZ):
    def __init__(self, n_dim=10, n_obj=3, **kwargs):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=n_obj,
            nadir_point=[2.51925352, 2.53011955, 2.66682285],
            ideal_point=[8.14503680e-06, 8.10665237e-06, 3.76396814e-06],
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.from_numpy(x).to(self.device)
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = self.g2(X_M)
        out["F"] = self.obj_func(X_, g, alpha=1).cpu().numpy()


class DTLZ3(DTLZ):
    def __init__(self, n_dim=10, n_obj=3, **kwargs):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=n_obj,
            nadir_point=[1548.83497615, 1459.58638514, 1518.61926977],
            ideal_point=[0.00024734, 0.00057021, 0.00393799],
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.from_numpy(x).to(self.device)
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g, alpha=1).cpu().numpy()


class DTLZ4(DTLZ):
    def __init__(self, n_dim=10, n_obj=3, alpha=100, d=100, **kwargs):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=n_obj,
            nadir_point=[2.75747516, 2.57674881, 2.52558196],
            ideal_point=[0, 0, 0],
        )
        self.alpha = alpha
        self.d = d

    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.from_numpy(x).to(self.device)
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = self.g2(X_M)
        out["F"] = self.obj_func(X_, g, alpha=self.alpha).cpu().numpy()


class DTLZ5(DTLZ):
    def __init__(self, n_dim=10, n_obj=3, **kwargs):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=n_obj,
            nadir_point=[2.40756462, 2.36937324, 2.45008585],
            ideal_point=[5.11293872e-06, 8.82751291e-06, 8.15403672e-06],
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.from_numpy(x).to(self.device)
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = self.g2(X_M)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = torch.stack([x[:, 0], theta[:, 1:].flatten()], dim=1)

        out["F"] = self.obj_func(theta, g).cpu().numpy()


class DTLZ6(DTLZ):
    def __init__(self, n_dim=10, n_obj=3, **kwargs):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=n_obj,
            nadir_point=[8.90471718, 8.89404128, 8.88893072],
            ideal_point=[9.56700075e-05, 9.51044735e-05, 5.57043760e-05],
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.from_numpy(x).to(self.device)
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = torch.sum(torch.pow(X_M, 0.1), axis=1)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = torch.stack([x[:, 0], theta[:, 1:].flatten()], dim=1)

        out["F"] = self.obj_func(theta, g).cpu().numpy()


class DTLZ7(DTLZ):
    def __init__(self, n_dim=10, n_obj=3, **kwargs):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=n_obj,
            nadir_point=[0.9999954, 0.99999772, 30.79227037],
            ideal_point=[1.80047270e-08, 5.11191256e-07, 4.40902012e00],
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.from_numpy(x).to(self.device)
        f = []
        for i in range(0, self.n_obj - 1):
            f.append(x[:, i])
        f = torch.stack(f, dim=1)

        g = 1 + 9 / self.k * torch.sum(x[:, -self.k :], axis=1)
        h = self.n_obj - torch.sum(
            f / (1 + g[:, None]) * (1 + torch.sin(3 * torch.pi * f)), axis=1
        )

        out["F"] = torch.cat([f, ((1 + g) * h).reshape(-1, 1)], dim=1).cpu().numpy()
