import math

import numpy as np
import torch
from torch import optim

from offline_moo.off_moo_bench.problem.base import BaseProblem
from offline_moo.off_moo_bench.problem.utils import constraint


class SyntheticProblem(BaseProblem):
    def __init__(
        self,
        name,
        n_obj,
        n_dim,
        lbound,
        ubound,
        problem_type="synthetic",
        nadir_point=None,
        ideal_point=None,
    ):
        super().__init__(
            name,
            problem_type,
            n_obj,
            n_dim,
            requires_normalized_x=True,
            nadir_point=nadir_point,
            ideal_point=ideal_point,
        )
        self.lbound = lbound
        self.ubound = ubound
        self.n_iter_optim = 500
        self.func = list()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = 'cpu'

    def get_func(self):
        raise NotImplementedError

    def generate_x(self, size):
        return (
            (torch.rand(size, self.n_dim) * (self.ubound - self.lbound) + self.lbound)
            .cpu()
            .numpy()
        )

    def f(self, x, *args, **kwargs):
        assert x.shape[-1] == self.n_dim
        if self.func == []:
            self.func = self.get_func()
        x = torch.from_numpy(x).to(self.device)
        self.lbound = self.lbound.to(x.device)
        self.ubound = self.ubound.to(x.device)
        x = x * (self.ubound - self.lbound) + self.lbound
        # print(x, self.func)
        objs = None
        for f in self.func:
            res = f(x).reshape(-1, 1)
            # print('res:', res)
            if objs is None:
                objs = res
            else:
                objs = torch.cat((objs, res), dim=1)
            # print('objs:', objs)
        return objs.cpu().numpy()

    def _evaluate(self, x, out, *args, **kwargs):

        out["F"] = self.f(x, *args, **kwargs)
        # return objs.cpu().numpy()

    def get_nadir_point(self):
        if self.nadir_point is not None:
            return self.nadir_point

        x = self.generate_x(size=1, n_dim=self.n_dim)
        x.requires_grad = True

        nadir_point = list()
        n_repeat = 10

        for f in self.func:
            max_value = float("-inf")

            optimizers = [
                optim.Adam([x], lr=0.01),
                optim.SGD([x], lr=0.01),
            ]

            for i in range(n_repeat):
                for optimizer in optimizers:

                    for _ in range(self.n_iter_optim):
                        optimizer.zero_grad()
                        loss = -f(constraint(x, self.lbound, self.ubound))
                        loss.backward()

                        optimizer.step()

                    max_value = max(
                        f(constraint(x, self.lbound, self.ubound)).detach().item(),
                        max_value,
                    )

            nadir_point.append(max_value)

        return torch.tensor(nadir_point)

    def get_ideal_point(self):
        if self.ideal_point is not None:
            return self.ideal_point

        x = self.generate_x(size=1, n_dim=self.n_dim)
        x.requires_grad = True

        ideal_point = list()
        n_repeat = 10

        for f in self.func:
            min_value = float("inf")

            optimizers = [
                optim.Adam([x], lr=0.01),
                optim.SGD([x], lr=0.01),
            ]

            for i in range(n_repeat):
                for optimizer in optimizers:

                    for _ in range(self.n_iter_optim):
                        optimizer.zero_grad()
                        loss = f(constraint(x, self.lbound, self.ubound))
                        loss.backward()

                        optimizer.step()

                    min_value = min(
                        f(constraint(x, self.lbound, self.ubound)).detach().item(),
                        min_value,
                    )

            ideal_point.append(min_value)

        return torch.tensor(ideal_point)


class VLMOP1(SyntheticProblem):
    def __init__(self, n_dim=1):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=2,
            lbound=torch.tensor([-2.0]).float(),
            ubound=torch.tensor([4.0]).float(),
            nadir_point=[4, 4],
            ideal_point=[0, 0],
        )
        self.func = self.get_func()

    def get_func(self):
        return [lambda x: torch.pow(x[:, 0], 2), lambda x: torch.pow(x[:, 0] - 2, 2)]


class VLMOP2(SyntheticProblem):
    def __init__(self, n_dim=6):

        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=2,
            lbound=torch.tensor([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0]).float(),
            ubound=torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]).float(),
            nadir_point=[1, 1],
            ideal_point=[0, 0],
        )
        self.func = self.get_func()

    def get_func(self):

        def f1(x):
            n = x.shape[1]
            return 1 - torch.exp(-torch.sum((x - 1 / np.sqrt(n)) ** 2, axis=1))

        def f2(x):
            n = x.shape[1]
            return 1 - torch.exp(-torch.sum((x + 1 / np.sqrt(n)) ** 2, axis=1))

        return [f1, f2]

    def _calc_pareto_front(self, n_pareto_points=1000):
        n = self.n_dim

        x = np.linspace(-1 / np.sqrt(n), 1 / np.sqrt(n), n_pareto_points)
        x_all = np.column_stack([x] * n)

        return self.evaluate(x_all)


class VLMOP3(SyntheticProblem):
    def __init__(self, n_dim=2):

        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=3,
            lbound=torch.tensor([-3.0, -3.0]).float(),
            ubound=torch.tensor([3.0, 3.0]).float(),
            nadir_point=[10, 60, 1],
            ideal_point=[0, 15, 0.0526],
        )
        self.func = self.get_func()

    def get_func(self):

        return [
            lambda x: 0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2)
            + torch.sin(x[:, 0] ** 2 + x[:, 1] ** 2),
            lambda x: (3 * x[:, 0] - 2 * x[:, 1] + 4) ** 2 / 8
            + (x[:, 0] - x[:, 1] + 1) ** 2 / 27
            + 15,
            lambda x: 1 / (x[:, 0] ** 2 + x[:, 1] ** 2 + 1)
            - 1.1 * torch.exp(-x[:, 0] ** 2 - x[:, 1] ** 2),
        ]


class Kursawe(SyntheticProblem):
    def __init__(self, n_dim=3):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=2,
            lbound=torch.ones(n_dim).float() * (-5),
            ubound=torch.ones(n_dim).float() * 5,
            nadir_point=[-4.88525476, 24.75967103],
            ideal_point=[-19.83156516 - 11.52304144],
        )
        self.func = self.get_func()

    def get_func(self):
        def f1(x):
            l = []
            for i in range(2):
                l.append(
                    -10
                    * torch.exp(
                        -0.2
                        * torch.sqrt(torch.square(x[:, i]) + torch.square(x[:, i + 1]))
                    )
                )
            return torch.sum(torch.stack(l, dim=1), axis=1)

        f2 = lambda x: torch.sum(
            torch.pow(torch.abs(x), 0.8) + 5 * torch.sin(torch.pow(x, 3)), axis=1
        )

        return [f1, f2]


class OmniTest(SyntheticProblem):
    def __init__(self, n_dim=2):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=2,
            lbound=torch.zeros(n_dim).float(),
            ubound=torch.ones(n_dim).float() * 6,
            nadir_point=[2, 2],
            ideal_point=[-2, -2],
        )
        self.func = self.get_func()

    def get_func(self):
        return [
            lambda x: torch.sum(torch.sin(torch.pi * x), axis=1),
            lambda x: torch.sum(torch.cos(torch.pi * x), axis=1),
        ]

    def _calc_pareto_set(self, n_pareto_points=500):
        # The Omni-test problem has 3^D Pareto subsets
        num_ps = int(3**self.n_var)
        h = int(n_pareto_points / num_ps)
        PS = np.zeros((num_ps * h, self.n_var))

        candidates = np.array(
            [np.linspace(2 * m + 1, 2 * m + 3 / 2, h) for m in range(3)]
        )
        # generate combination indices
        candidates_indices = [[0, 1, 2] for _ in range(self.n_var)]
        a = np.meshgrid(*candidates_indices)
        combination_indices = np.array(a).T.reshape(-1, self.n_var)
        # generate 3^D combinations
        for i in range(num_ps):
            PS[i * h : i * h + h, :] = candidates[combination_indices[i]].T
        return PS


class SYMPARTRotated(SyntheticProblem):
    def __init__(self, n_dim=2):
        self.a = torch.tensor(1)
        self.b = torch.tensor(10)
        self.c = torch.tensor(10)
        self.w = torch.tensor(torch.pi / 4)

        self.IRM = torch.tensor(
            [
                [torch.cos(self.w), torch.sin(self.w)],
                [-torch.sin(self.w), torch.cos(self.w)],
            ]
        )

        r = torch.max(self.b, self.c)

        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=2,
            lbound=torch.ones(n_dim).float() * (-10 * r),
            ubound=torch.ones(n_dim).float() * (10 * r),
            nadir_point=None,
            ideal_point=None,
        )
        self.func = self.get_func()

    def get_func(self):
        def preprocess(x):
            x = torch.tensor(x, dtype=torch.float32)
            if self.w == 0:
                x1 = x[:, 0]
                x2 = x[:, 1]
            else:
                self.IRM = self.IRM.to(x.device)
                y = torch.tensor([torch.matmul(self.IRM, x_i) for x_i in x])
                x1 = y[:, 0]
                x2 = y[:, 1]

            a, b, c = self.a.to(x.device), self.b.to(x.device), self.c.to(x.device)
            t1_hat = torch.sign(x1) * torch.ceil(
                (torch.abs(x1) - a - c / 2) / (2 * a + c)
            )
            t2_hat = torch.sign(x2) * torch.ceil((torch.abs(x2) - b / 2) / b)
            one = torch.ones(len(x))
            t1 = torch.sign(t1_hat) * torch.min(
                torch.vstack((torch.abs(t1_hat), one)), axis=0
            )
            t2 = torch.sign(t2_hat) * torch.min(
                torch.vstack((torch.abs(t2_hat), one)), axis=0
            )

            p1 = x1 - t1 * c
            p2 = x2 - t2 * b

            return a, p1, p2

        def f1(x):
            a, p1, p2 = preprocess(x)
            return (p1 + a) ** 2 + p2**2

        def f2(x):
            a, p1, p2 = preprocess(x)
            return (p1 - a) ** 2 + p2**2

        return [f1, f2]


class ZDT1(SyntheticProblem):
    def __init__(self, n_dim=30):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=2,
            lbound=torch.zeros(n_dim).float(),
            ubound=torch.ones(n_dim).float(),
            nadir_point=[0.99999809, 7.8250663],
            ideal_point=[7.27727072e-07, 2.38306515e-01],
        )
        self.func = self.get_func()

    def get_func(self):
        f1 = lambda x: x[:, 0]

        def f2(x):
            g = 1 + 9.0 / (self.n_dim - 1) * torch.sum(x[:, 1:], axis=1)
            return g * (1 - torch.pow((x[:, 0] / g), 0.5))

        return [f1, f2]


class ZDT2(SyntheticProblem):
    def __init__(self, n_dim=30):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=2,
            lbound=torch.zeros(n_dim).float(),
            ubound=torch.ones(n_dim).float(),
            nadir_point=[0.99999706, 9.74316166],
            ideal_point=[3.21854609e-09, 8.08941746e-02],
        )
        self.func = self.get_func()

    def get_func(self):
        f1 = lambda x: x[:, 0]

        def f2(x):
            c = torch.sum(x[:, 1:], axis=1)
            g = 1.0 + 9.0 * c / (self.n_dim - 1)
            return g * (1 - torch.pow((x[:, 0] * 1.0 / g), 2))

        return [f1, f2]


class ZDT3(SyntheticProblem):
    def __init__(self, n_dim=30):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=2,
            lbound=torch.zeros(n_dim).float(),
            ubound=torch.ones(n_dim).float(),
            nadir_point=[0.99999954, 9.83643758],
            ideal_point=[3.85619704e-07, -7.69222414e-01],
        )
        self.func = self.get_func()

    def get_func(self):
        f1 = lambda x: x[:, 0]

        def f2(x):
            c = torch.sum(x[:, 1:], axis=1)
            g = 1.0 + 9.0 * c / (self.n_dim - 1)
            return g * (
                1
                - torch.pow(x[:, 0] * 1.0 / g, 0.5)
                - (x[:, 0] * 1.0 / g) * torch.sin(10 * torch.pi * x[:, 0])
            )

        return [f1, f2]


class ZDT4(SyntheticProblem):
    def __init__(self, n_dim=10):
        xl = torch.ones(n_dim).float() * (-5)
        xl[0] = 0.0
        xu = torch.ones(n_dim).float() * 5
        xu[0] = 1.0

        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=2,
            lbound=xl,
            ubound=xu,
            nadir_point=[0.99980978, 273.25533168],
            ideal_point=[0, 1.58235196],
        )
        self.func = self.get_func()

    def get_func(self):
        f1 = lambda x: x[:, 0]

        def f2(x):
            g = 1.0
            g += 10 * (self.n_dim - 1)
            for i in range(1, self.n_dim):
                g += x[:, i] * x[:, i] - 10.0 * torch.cos(4.0 * torch.pi * x[:, i])
            h = 1.0 - torch.sqrt(x[:, 0] / g)
            return g * h

        return [f1, f2]


class ZDT6(SyntheticProblem):
    def __init__(self, n_dim=10):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=2,
            lbound=torch.zeros(n_dim).float(),
            ubound=torch.ones(n_dim).float(),
            nadir_point=[1.0, 9.34365208],
            ideal_point=[0.28077532, 0.07205017],
        )
        self.func = self.get_func()

    def get_func(self):
        f1 = lambda x: 1 - torch.exp(-4 * x[:, 0]) * torch.pow(
            torch.sin(6 * torch.pi * x[:, 0]), 6
        )

        def f2(x):
            g = 1 + 9.0 * torch.pow(
                torch.sum(x[:, 1:], axis=1) / (self.n_dim - 1.0), 0.25
            )
            x0 = 1 - torch.exp(-4 * x[:, 0]) * torch.pow(
                torch.sin(6 * torch.pi * x[:, 0]), 6
            )
            return g * (1 - torch.pow(x0 / g, 2))

        return [f1, f2]


class RE21(SyntheticProblem):
    def __init__(self, n_dim=4):

        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma

        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=2,
            lbound=torch.tensor(
                [tmp_val, np.sqrt(2.0) * tmp_val, np.sqrt(2.0) * tmp_val, tmp_val]
            ).float(),
            ubound=torch.ones(n_dim).float() * 3 * tmp_val,
            nadir_point=[2.97123988e03, 4.85512338e-02],
            ideal_point=[1.23920758e03, 2.98607537e-03],
        )
        self.func = self.get_func()

    def get_func(self):

        F = 10.0
        E = 2.0 * 1e5
        L = 200.0

        return [
            lambda x: L
            * ((2 * x[:, 0]) + np.sqrt(2.0) * x[:, 1] + torch.sqrt(x[:, 2]) + x[:, 3]),
            lambda x: ((F * L) / E)
            * (
                (2.0 / x[:, 0])
                + (2.0 * np.sqrt(2.0) / x[:, 1])
                - (2.0 * np.sqrt(2.0) / x[:, 2])
                + (2.0 / x[:, 3])
            ),
        ]


class RE22(SyntheticProblem):
    def __init__(self, n_dim=3):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=2,
            lbound=torch.tensor([0.2, 0.0, 0.0]).float(),
            ubound=torch.tensor([15, 20, 40]).float(),
            nadir_point=[8.29079443e02, 2.40721725e06],
            ideal_point=[5.88238115, 0.0],
        )

        self.feasible_vals = torch.tensor(
            [
                0.20,
                0.31,
                0.40,
                0.44,
                0.60,
                0.62,
                0.79,
                0.80,
                0.88,
                0.93,
                1.0,
                1.20,
                1.24,
                1.32,
                1.40,
                1.55,
                1.58,
                1.60,
                1.76,
                1.80,
                1.86,
                2.0,
                2.17,
                2.20,
                2.37,
                2.40,
                2.48,
                2.60,
                2.64,
                2.79,
                2.80,
                3.0,
                3.08,
                3,
                10,
                3.16,
                3.41,
                3.52,
                3.60,
                3.72,
                3.95,
                3.96,
                4.0,
                4.03,
                4.20,
                4.34,
                4.40,
                4.65,
                4.74,
                4.80,
                4.84,
                5.0,
                5.28,
                5.40,
                5.53,
                5.72,
                6.0,
                6.16,
                6.32,
                6.60,
                7.11,
                7.20,
                7.80,
                7.90,
                8.0,
                8.40,
                8.69,
                9.0,
                9.48,
                10.27,
                11.0,
                11.06,
                11.85,
                12.0,
                13.0,
                14.0,
                15.0,
            ]
        ).to(self.device)

    def get_func(self):
        def preprocess(x):
            # idx = torch.abs(torch.asarray(self.feasible_vals) - x[:, 0]).argmin()
            x1 = torch.tensor(
                [
                    self.feasible_vals[
                        torch.abs(torch.asarray(self.feasible_vals) - x[i, 0]).argmin()
                    ]
                    for i in range(x.shape[0])
                ]
            ).to(self.device)
            x2 = x[:, 1]
            x3 = x[:, 2]
            return x1, x2, x3

        def f1(x):
            x1, x2, x3 = preprocess(x)
            return (29.4 * x1) + (0.6 * x2 * x3)

        def f2(x):
            x1, x2, x3 = preprocess(x)
            g1 = (x1 * x3) - 7.735 * ((x1 * x1) / x2) - 180.0
            g2 = 4.0 - (x3 / x2)

            g1 = g1.float()
            g2 = g2.float()

            g = torch.stack([g1, g2]).float()
            z = torch.zeros(g.shape).float().to(g.device)
            g = torch.where(g < 0, -g, z)

            return torch.sum(g, dim=0).float()

        return [f1, f2]


class RE23(SyntheticProblem):
    def __init__(self, n_dim=4):

        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=2,
            lbound=torch.tensor([1, 1, 10, 10]).float(),
            ubound=torch.tensor([100, 100, 200, 240]).float(),
            nadir_point=[713710.875, 1288669.78054],
            ideal_point=[15.9018007813, 0.0],
        )
        self.func = self.get_func()

    def get_func(self):
        def preprocess(x):
            x1 = 0.0625 * torch.round(x[:, 0])
            x2 = 0.0625 * torch.round(x[:, 1])
            x3 = x[:, 2]
            x4 = x[:, 3]
            return x1, x2, x3, x4

        # First original objective function
        def f1(x):
            x1, x2, x3, x4 = preprocess(x)
            obj = (
                (0.6224 * x1 * x3 * x4)
                + (1.7781 * x2 * x3 * x3)
                + (3.1661 * x1 * x1 * x4)
                + (19.84 * x1 * x1 * x3)
            )
            return obj.float()

        # Original constraint functions
        def f2(x):
            x1, x2, x3, x4 = preprocess(x)
            g1 = x1 - (0.0193 * x3)
            g2 = x2 - (0.00954 * x3)
            g3 = (
                (np.pi * x3 * x3 * x4)
                + ((4.0 / 3.0) * (np.pi * x3 * x3 * x3))
                - 1296000
            )

            g1 = g1.float()
            g2 = g2.float()
            g3 = g3.float()

            g = torch.stack([g1, g2, g3]).float()
            z = torch.zeros(g.shape).float().to(g.device)
            g = torch.where(g < 0, -g, z)

            return torch.sum(g, axis=0).float()

        return [f1, f2]


class RE24(SyntheticProblem):
    def __init__(self, n_dim=2):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=2,
            lbound=torch.tensor([0.5, 0.5]).float(),
            ubound=torch.tensor([4, 50]).float(),
            nadir_point=[5997.8316325, 43.67584229],
            ideal_point=[60.65314083, 0.0],
        )

    def get_func(self):
        f1 = lambda x: x[:, 0] + (120 * x[:, 1])

        def f2(x):
            E = 700000
            sigma_b_max = 700
            tau_max = 450
            delta_max = 1.5
            sigma_k = (E * x[:, 0] * x[:, 1]) / 100
            sigma_b = 4500 / (x[:, 0] * x[:, 1])
            tau = 1800 / x[:, 1]
            delta = (56.2 * 10000) / (E * x[:, 0] * x[:, 1] * x[:, 1])

            g1 = 1 - (sigma_b / sigma_b_max)
            g2 = 1 - (tau / tau_max)
            g3 = 1 - (delta / delta_max)
            g4 = 1 - (sigma_b / sigma_k)

            g1 = g1.float()
            g2 = g2.float()
            g3 = g3.float()
            g4 = g4.float()

            g = torch.stack([g1, g2, g3, g4]).float()
            z = torch.zeros(g.shape).to(g.device)
            g = torch.where(g < 0, -g, z)
            return torch.sum(g, axis=0).float()

        return [f1, f2]


class RE25(SyntheticProblem):
    def __init__(self, n_dim=3):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=2,
            lbound=torch.tensor([1, 0.6, 0.09]).float(),
            ubound=torch.tensor([70, 3, 0.5]).float(),
            nadir_point=[1.24795202e02, 1.00387350e07],
            ideal_point=[0.03759284, 0.0],
        )

        self.feasible_vals = torch.tensor(
            [
                0.009,
                0.0095,
                0.0104,
                0.0118,
                0.0128,
                0.0132,
                0.014,
                0.015,
                0.0162,
                0.0173,
                0.018,
                0.02,
                0.023,
                0.025,
                0.028,
                0.032,
                0.035,
                0.041,
                0.047,
                0.054,
                0.063,
                0.072,
                0.08,
                0.092,
                0.105,
                0.12,
                0.135,
                0.148,
                0.162,
                0.177,
                0.192,
                0.207,
                0.225,
                0.244,
                0.263,
                0.283,
                0.307,
                0.331,
                0.362,
                0.394,
                0.4375,
                0.5,
            ]
        ).to(self.device)

    def get_func(self):
        def preprocess(x):
            x1 = torch.round(x[:, 0])
            x2 = x[:, 1]
            # idx = torch.abs(torch.asarray(self.feasible_vals) - x[:, 2]).argmin()
            x3 = torch.tensor(
                [
                    self.feasible_vals[
                        torch.abs(torch.asarray(self.feasible_vals) - x[i, 2]).argmin()
                    ]
                    for i in range(x.shape[0])
                ]
            ).to(self.device)
            return x1, x2, x3

        def f1(x):
            x1, x2, x3 = preprocess(x)
            return (torch.pi * torch.pi * x2 * x3 * x3 * (x1 + 2)) / 4.0

        def f2(x):
            x1, x2, x3 = preprocess(x)

            # constraint functions
            Cf = ((4.0 * (x2 / x3) - 1) / (4.0 * (x2 / x3) - 4)) + (0.615 * x3 / x2)
            Fmax = 1000.0
            S = 189000.0
            G = 11.5 * 1e6
            K = (G * x3 * x3 * x3 * x3) / (8 * x1 * x2 * x2 * x2)
            lmax = 14.0
            lf = (Fmax / K) + 1.05 * (x1 + 2) * x3
            dmin = 0.2
            Dmax = 3
            Fp = 300.0
            sigmaP = Fp / K
            sigmaPM = 6
            sigmaW = 1.25

            g1 = -((8 * Cf * Fmax * x2) / (np.pi * x3 * x3 * x3)) + S
            g2 = -lf + lmax
            g3 = -3 + (x2 / x3)
            g4 = -sigmaP + sigmaPM
            g5 = -sigmaP - ((Fmax - Fp) / K) - 1.05 * (x1 + 2) * x3 + lf
            g6 = sigmaW - ((Fmax - Fp) / K)

            g1 = g1.float()
            g2 = g2.float()
            g3 = g3.float()
            g4 = g4.float()
            g5 = g5.float()
            g6 = g6.float()

            g = torch.stack([g1, g2, g3, g4, g5, g6]).float()
            z = torch.zeros(g.shape).to(g.device)
            g = torch.where(g < 0, -g, z)

            return torch.sum(g, dim=0).float()

        return [f1, f2]


class RE31(SyntheticProblem):
    def __init__(self, n_dim=3):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=3,
            lbound=torch.tensor([0.00001, 0.00001, 1.0]).float(),
            ubound=torch.tensor([100.0, 100.0, 3.0]).float(),
            nadir_point=[8.08852742e02, 6.89337582e06, 6.79345000e06],
            ideal_point=[0.02500048, 0.33336024, 0.0],
        )

    def get_func(self):
        f1 = lambda x: x[:, 0] * torch.sqrt(16.0 + (x[:, 2] * x[:, 2])) + x[
            :, 1
        ] * torch.sqrt(1.0 + x[:, 2] * x[:, 2])
        f2 = lambda x: (20.0 * torch.sqrt(16.0 + (x[:, 2] * x[:, 2]))) / (
            x[:, 0] * x[:, 2]
        )

        def f3(x):
            g1 = 0.1 - f1(x)
            g2 = 100000.0 - f2(x)
            g3 = 100000 - (
                (80.0 * torch.sqrt(1.0 + x[:, 2] * x[:, 2])) / (x[:, 2] * x[:, 1])
            )

            g1 = g1.float()
            g2 = g2.float()
            g3 = g3.float()

            g = torch.stack([g1, g2, g3]).float()
            z = torch.zeros(g.shape).to(g.device)
            g = torch.where(g < 0, -g, z)

            return torch.sum(g, dim=0).float()

        return [f1, f2, f3]


class RE32(SyntheticProblem):
    def __init__(self, n_dim=4):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=3,
            lbound=torch.tensor([0.125, 0.1, 0.1, 0.125]).float(),
            ubound=torch.tensor([5.0, 10.0, 10.0, 5.0]).float(),
            nadir_point=[2.90661885e02, 1.65524628e04, 3.88265024e08],
            ideal_point=[0.01366746, 0.00043994, 0.0],
        )

    def get_func(self):
        P = 6000
        L = 14
        E = 30 * 1e6

        # // deltaMax = 0.25
        G = 12 * 1e6
        tauMax = 13600
        sigmaMax = 30000

        f1 = lambda x: (1.10471 * x[:, 0] * x[:, 0] * x[:, 1]) + (
            0.04811 * x[:, 2] * x[:, 3]
        ) * (14.0 + x[:, 1])
        # Second original objective function
        f2 = lambda x: (4 * P * L * L * L) / (E * x[:, 3] * x[:, 2] * x[:, 2] * x[:, 2])

        def f3(x):
            x1 = x[:, 0]
            x2 = x[:, 1]
            x3 = x[:, 2]
            x4 = x[:, 3]

            # Constraint functions
            M = P * (L + (x2 / 2))
            tmpVar = ((x2 * x2) / 4.0) + torch.pow((x1 + x3) / 2.0, 2)
            R = torch.sqrt(tmpVar)
            tmpVar = ((x2 * x2) / 12.0) + torch.pow((x1 + x3) / 2.0, 2)
            J = 2 * math.sqrt(2) * x1 * x2 * tmpVar

            tauDashDash = (M * R) / J
            tauDash = P / (math.sqrt(2) * x1 * x2)
            tmpVar = (
                tauDash * tauDash
                + ((2 * tauDash * tauDashDash * x2) / (2 * R))
                + (tauDashDash * tauDashDash)
            )
            tau = torch.sqrt(tmpVar)
            sigma = (6 * P * L) / (x4 * x3 * x3)
            tmpVar = (
                4.013
                * E
                * torch.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0)
                / (L * L)
            )
            tmpVar2 = (x3 / (2 * L)) * math.sqrt(E / (4 * G))
            PC = tmpVar * (1 - tmpVar2)

            g1 = tauMax - tau
            g2 = sigmaMax - sigma
            g3 = x4 - x1
            g4 = PC - P

            g1 = g1.float()
            g2 = g2.float()
            g3 = g3.float()
            g4 = g4.float()

            g = torch.stack([g1, g2, g3, g4]).float()
            z = torch.zeros(g.shape).to(g.device)
            g = torch.where(g < 0, -g, z)
            return torch.sum(g, dim=0).float()

        return [f1, f2, f3]


class RE33(SyntheticProblem):
    def __init__(self, n_dim=4):

        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=3,
            lbound=torch.tensor([55, 75, 1000, 11]).float(),
            ubound=torch.tensor([80, 110, 3000, 20]).float(),
            nadir_point=[8.01164324, 8.83604223, 2343.29711914],
            ideal_point=[-0.721525, 1.13907203907, 0.0],
        )
        self.func = self.get_func()

    def get_func(self):

        # First original objective function
        def f1(x):
            x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
            return 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)

        # Second original objective function
        def f2(x):
            x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
            return ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (
                x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)
            )

        # Reformulated objective functions
        def f3(x):
            x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
            g1 = (x2 - x1) - 20.0
            g2 = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
            g3 = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / torch.pow(
                (x2 * x2 - x1 * x1), 2
            )
            g4 = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (
                x2 * x2 - x1 * x1
            ) - 900.0

            g1 = g1.float()
            g2 = g2.float()
            g3 = g3.float()
            g4 = g4.float()

            g = torch.stack([g1, g2, g3, g4]).float()
            z = torch.zeros(g.shape).float().to(g.device)
            g = torch.where(g < 0, -g, z)

            return torch.sum(g, axis=0).float()

        return [f1, f2, f3]


class RE34(SyntheticProblem):
    def __init__(self, n_dim=5):

        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=3,
            lbound=torch.ones(n_dim).float(),
            ubound=torch.ones(n_dim).float() * 3,
            nadir_point=[1.70251811e03, 1.16807224e01, 2.63918844e-01],
            ideal_point=[1.66171457e03, 6.14473949e00, 4.00202874e-02],
        )
        self.func = self.get_func()

    def get_func(self):

        def f1(x):
            x1, x2, x3, x4, x5 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
            return (
                1640.2823
                + (2.3573285 * x1)
                + (2.3220035 * x2)
                + (4.5688768 * x3)
                + (7.7213633 * x4)
                + (4.4559504 * x5)
            )

        def f2(x):
            x1, x2, x3, x4, x5 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
            return (
                6.5856
                + (1.15 * x1)
                - (1.0427 * x2)
                + (0.9738 * x3)
                + (0.8364 * x4)
                - (0.3695 * x1 * x4)
                + (0.0861 * x1 * x5)
                + (0.3628 * x2 * x4)
                - (0.1106 * x1 * x1)
                - (0.3437 * x3 * x3)
                + (0.1764 * x4 * x4)
            )

        def f3(x):
            x1, x2, x3, x4, x5 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
            return (
                -0.0551
                + (0.0181 * x1)
                + (0.1024 * x2)
                + (0.0421 * x3)
                - (0.0073 * x1 * x2)
                + (0.024 * x2 * x3)
                - (0.0118 * x2 * x4)
                - (0.0204 * x3 * x4)
                - (0.008 * x3 * x5)
                - (0.0241 * x2 * x2)
                + (0.0109 * x4 * x4)
            )

        return [f1, f2, f3]


class RE35(SyntheticProblem):
    def __init__(self, n_dim=7):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=3,
            lbound=torch.tensor([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0]).float(),
            ubound=torch.tensor([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]).float(),
            nadir_point=[7050.78959905, 1696.66697789, 397.83456421],
            ideal_point=[2364.91273887, 694.31805919, 0.0],
        )

    def get_func(self):
        f1 = (
            lambda x: 0.7854
            * x[:, 0]
            * (x[:, 1] * x[:, 1])
            * (
                ((10.0 * torch.round(x[:, 2]) * torch.round(x[:, 2])) / 3.0)
                + (14.933 * torch.round(x[:, 2]))
                - 43.0934
            )
            - 1.508 * x[:, 0] * (x[:, 5] * x[:, 5] + x[:, 6] * x[:, 6])
            + 7.477 * (x[:, 5] * x[:, 5] * x[:, 5] + x[:, 6] * x[:, 6] * x[:, 6])
            + 0.7854 * (x[:, 3] * x[:, 5] * x[:, 5] + x[:, 4] * x[:, 6] * x[:, 6])
        )

        f2 = lambda x: torch.sqrt(
            torch.pow((745.0 * x[:, 3]) / (x[:, 1] * torch.round(x[:, 2])), 2.0)
            + 1.69 * 1e7
        ) / (0.1 * x[:, 5] * x[:, 5] * x[:, 5])

        def f3(x):
            x1 = x[:, 0]
            x2 = x[:, 1]
            x3 = torch.round(x[:, 2])
            x4 = x[:, 3]
            x5 = x[:, 4]
            x6 = x[:, 5]
            x7 = x[:, 6]

            # Constraint functions
            g1 = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
            g2 = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
            g3 = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
            g4 = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
            g5 = -(x2 * x3) + 40.0
            g6 = -(x1 / x2) + 12.0
            g7 = -5.0 + (x1 / x2)
            g8 = -1.9 + x4 - 1.5 * x6
            g9 = -1.9 + x5 - 1.1 * x7
            g10 = -f2(x) + 1300.0
            tmpVar = torch.pow((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
            g11 = -torch.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0

            g1 = g1.float()
            g2 = g2.float()
            g3 = g3.float()
            g4 = g4.float()
            g5 = g5.float()
            g6 = g6.float()
            g7 = g7.float()
            g8 = g8.float()
            g9 = g9.float()
            g10 = g10.float()
            g11 = g11.float()

            g = torch.stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])
            z = torch.zeros(g.shape).to(g.device)
            g = torch.where(g < 0, -g, z)
            return torch.sum(g, dim=0).float()

        return [f1, f2, f3]


class RE36(SyntheticProblem):
    def __init__(self, n_dim=4):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=3,
            lbound=torch.ones(n_dim).float() * 12,
            ubound=torch.ones(n_dim).float() * 60,
            nadir_point=[10.21185714, 60.0, 0.97335988],
            ideal_point=[3.87449393e-03, 1.30000000e01, 0.00000000e00],
        )

    def get_func(self):
        f1 = lambda x: torch.abs(
            6.931
            - (
                (torch.round(x[:, 2]) / torch.round(x[:, 0]))
                * (torch.round(x[:, 3]) / torch.round(x[:, 1]))
            )
        )

        f2 = lambda x: torch.max(
            torch.stack(
                [
                    torch.round(x[:, 0]),
                    torch.round(x[:, 1]),
                    torch.round(x[:, 2]),
                    torch.round(x[:, 3]),
                ],
                dim=1,
            ),
            dim=1,
            keepdim=True,
        )[0]

        # def f2(x):
        #     print(torch.stack([torch.round(x[:, 0]),
        #                          torch.round(x[:, 1]),
        #                          torch.round(x[:, 2]),
        #                          torch.round(x[:, 3])], dim=1))
        #     assert 0, torch.max(torch.stack([torch.round(x[:, 0]),
        #                          torch.round(x[:, 1]),
        #                          torch.round(x[:, 2]),
        #                          torch.round(x[:, 3])], dim=1), dim=1)

        def f3(x):
            g1 = 0.5 - (f1(x) / 6.931)
            g1 = g1.float()
            g = torch.stack([g1]).float()
            z = torch.zeros(g.shape).to(g.device)
            g = torch.where(g < 0, -g, z)
            return torch.sum(g, dim=0).float()

        return [f1, f2, f3]


class RE37(SyntheticProblem):
    def __init__(self, n_dim=4):

        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=3,
            lbound=torch.tensor([0, 0, 0, 0]).float(),
            ubound=torch.tensor([1, 1, 1, 1]).float(),
            nadir_point=[0.98949120096, 0.956587924661, 0.987530948586],
            ideal_point=[0.00889341391106, 0.00488, -0.431499999825],
        )
        self.func = self.get_func()

    def get_func(self):
        # f1 (TF_max)
        def f1(x):
            xAlpha, xHA, xOA, xOPTT = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
            return (
                0.692
                + (0.477 * xAlpha)
                - (0.687 * xHA)
                - (0.080 * xOA)
                - (0.0650 * xOPTT)
                - (0.167 * xAlpha * xAlpha)
                - (0.0129 * xHA * xAlpha)
                + (0.0796 * xHA * xHA)
                - (0.0634 * xOA * xAlpha)
                - (0.0257 * xOA * xHA)
                + (0.0877 * xOA * xOA)
                - (0.0521 * xOPTT * xAlpha)
                + (0.00156 * xOPTT * xHA)
                + (0.00198 * xOPTT * xOA)
                + (0.0184 * xOPTT * xOPTT)
            )

        # f2 (X_cc)
        def f2(x):
            xAlpha, xHA, xOA, xOPTT = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
            return (
                0.153
                - (0.322 * xAlpha)
                + (0.396 * xHA)
                + (0.424 * xOA)
                + (0.0226 * xOPTT)
                + (0.175 * xAlpha * xAlpha)
                + (0.0185 * xHA * xAlpha)
                - (0.0701 * xHA * xHA)
                - (0.251 * xOA * xAlpha)
                + (0.179 * xOA * xHA)
                + (0.0150 * xOA * xOA)
                + (0.0134 * xOPTT * xAlpha)
                + (0.0296 * xOPTT * xHA)
                + (0.0752 * xOPTT * xOA)
                + (0.0192 * xOPTT * xOPTT)
            )

        # f3 (TT_max)
        def f3(x):
            xAlpha, xHA, xOA, xOPTT = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
            return (
                0.370
                - (0.205 * xAlpha)
                + (0.0307 * xHA)
                + (0.108 * xOA)
                + (1.019 * xOPTT)
                - (0.135 * xAlpha * xAlpha)
                + (0.0141 * xHA * xAlpha)
                + (0.0998 * xHA * xHA)
                + (0.208 * xOA * xAlpha)
                - (0.0301 * xOA * xHA)
                - (0.226 * xOA * xOA)
                + (0.353 * xOPTT * xAlpha)
                - (0.0497 * xOPTT * xOA)
                - (0.423 * xOPTT * xOPTT)
                + (0.202 * xHA * xAlpha * xAlpha)
                - (0.281 * xOA * xAlpha * xAlpha)
                - (0.342 * xHA * xHA * xAlpha)
                - (0.245 * xHA * xHA * xOA)
                + (0.281 * xOA * xOA * xHA)
                - (0.184 * xOPTT * xOPTT * xAlpha)
                - (0.281 * xHA * xAlpha * xOA)
            )

        return [f1, f2, f3]


class RE41(SyntheticProblem):
    def __init__(self, n_dim=7):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=4,
            lbound=torch.tensor([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4]).float(),
            ubound=torch.tensor([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2]).float(),
            nadir_point=[42.64892315, 4.42692892, 13.07735381, 13.44858932],
            ideal_point=[15.87368122, 3.58726077, 10.6748894, 0.0],
        )

    def get_func(self):
        f1 = (
            lambda x: 1.98
            + 4.9 * x[:, 0]
            + 6.67 * x[:, 1]
            + 6.98 * x[:, 2]
            + 4.01 * x[:, 3]
            + 1.78 * x[:, 4]
            + 0.00001 * x[:, 5]
            + 2.73 * x[:, 6]
        )
        f2 = lambda x: 4.72 - 0.5 * x[:, 3] - 0.19 * x[:, 1] * x[:, 2]

        def f3(x):
            Vmbp = 10.58 - 0.674 * x[:, 0] * x[:, 1] - 0.67275 * x[:, 1]
            Vfd = 16.45 - 0.489 * x[:, 2] * x[:, 6] - 0.843 * x[:, 4] * x[:, 5]
            return 0.5 * (Vmbp + Vfd)

        def f4(x):
            x1 = x[:, 0]
            x2 = x[:, 1]
            x3 = x[:, 2]
            x4 = x[:, 3]
            x5 = x[:, 4]
            x6 = x[:, 5]
            x7 = x[:, 6]

            Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
            Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6

            # Constraint functions
            g1 = 1 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
            g2 = 0.32 - (
                0.261
                - 0.0159 * x1 * x2
                - 0.06486 * x1
                - 0.019 * x2 * x7
                + 0.0144 * x3 * x5
                + 0.0154464 * x6
            )
            g3 = 0.32 - (
                0.214
                + 0.00817 * x5
                - 0.045195 * x1
                - 0.0135168 * x1
                + 0.03099 * x2 * x6
                - 0.018 * x2 * x7
                + 0.007176 * x3
                + 0.023232 * x3
                - 0.00364 * x5 * x6
                - 0.018 * x2 * x2
            )
            g4 = 0.32 - (
                0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2
            )
            g5 = 32 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
            g6 = 32 - (
                33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728
            )
            g7 = 32 - (46.36 - 9.9 * x2 - 4.4505 * x1)
            g8 = 4 - f2(x)
            g9 = 9.9 - Vmbp
            g10 = 15.7 - Vfd

            g1 = g1.float()
            g2 = g2.float()
            g3 = g3.float()
            g4 = g4.float()
            g5 = g5.float()
            g6 = g6.float()
            g7 = g7.float()
            g8 = g8.float()
            g9 = g9.float()
            g10 = g10.float()

            g = torch.stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]).float()
            z = torch.zeros(g.shape).to(g.device)
            g = torch.where(g < 0, -g, z)
            return torch.sum(g, dim=0).float()

        return [f1, f2, f3, f4]


class RE42(SyntheticProblem):
    def __init__(self, n_dim=6):

        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=4,
            lbound=torch.tensor([150, 20, 13, 10, 14, 0.63]).float(),
            ubound=torch.tensor([274.32, 32.31, 25, 11.71, 18, 0.75]).float(),
            nadir_point=[-2.63907595e02, 1.99048970e04, 2.85467850e04, 1.49821739e01],
            ideal_point=[-2503.83371142, 4017.64399443, 2089.36081316, 0.0],
        )
        self.func = self.get_func()

    def get_func(self):
        def preprocess(x):
            x_L = x[:, 0]
            x_B = x[:, 1]
            x_D = x[:, 2]
            x_T = x[:, 3]
            x_Vk = x[:, 4]
            x_CB = x[:, 5]

            displacement = 1.025 * x_L * x_B * x_T * x_CB
            V = 0.5144 * x_Vk
            g = 9.8065
            Fn = V / torch.pow(g * x_L, 0.5)
            a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
            b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

            power = (torch.pow(displacement, 2.0 / 3.0) * torch.pow(x_Vk, 3.0)) / (
                a + (b * Fn)
            )
            outfit_weight = (
                1.0
                * torch.pow(x_L, 0.8)
                * torch.pow(x_B, 0.6)
                * torch.pow(x_D, 0.3)
                * torch.pow(x_CB, 0.1)
            )
            steel_weight = (
                0.034
                * torch.pow(x_L, 1.7)
                * torch.pow(x_B, 0.7)
                * torch.pow(x_D, 0.4)
                * torch.pow(x_CB, 0.5)
            )
            machinery_weight = 0.17 * torch.pow(power, 0.9)
            light_ship_weight = steel_weight + outfit_weight + machinery_weight

            ship_cost = 1.3 * (
                (2000.0 * torch.pow(steel_weight, 0.85))
                + (3500.0 * outfit_weight)
                + (2400.0 * torch.pow(power, 0.8))
            )
            capital_costs = 0.2 * ship_cost

            DWT = displacement - light_ship_weight

            running_costs = 40000.0 * torch.pow(DWT, 0.3)

            round_trip_miles = 5000.0
            sea_days = (round_trip_miles / 24.0) * x_Vk
            handling_rate = 8000.0

            daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
            fuel_price = 100.0
            fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
            port_cost = 6.3 * torch.pow(DWT, 0.8)

            fuel_carried = daily_consumption * (sea_days + 5.0)
            miscellaneous_DWT = 2.0 * torch.pow(DWT, 0.5)

            cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
            port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
            RTPA = 350.0 / (sea_days + port_days)

            voyage_costs = (fuel_cost + port_cost) * RTPA
            annual_costs = capital_costs + running_costs + voyage_costs
            annual_cargo = cargo_DWT * RTPA

            # Reformulated objective functions
            c1 = (x_L / x_B) - 6.0
            c2 = -(x_L / x_D) + 15.0
            c3 = -(x_L / x_T) + 19.0
            c4 = 0.45 * torch.pow(DWT, 0.31) - x_T
            c5 = 0.7 * x_D + 0.7 - x_T
            c6 = 500000.0 - DWT
            c7 = DWT - 3000.0
            c8 = 0.32 - Fn

            KB = 0.53 * x_T
            BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
            KG = 1.0 + 0.52 * x_D
            c9 = (KB + BMT - KG) - (0.07 * x_B)

            constraintFuncs = torch.stack([c1, c2, c3, c4, c5, c6, c7, c8, c9]).float()
            z = torch.zeros(constraintFuncs.shape).float().to(constraintFuncs.device)
            constraintFuncs = torch.where(constraintFuncs < 0, -constraintFuncs, z)

            return {
                "annual_costs": annual_costs,
                "annual_cargo": annual_cargo,
                "light_ship_weight": light_ship_weight,
                "constraintFuncs": constraintFuncs,
            }

        def f1(x):
            res = preprocess(x)
            annual_costs, annual_cargo = res["annual_costs"], res["annual_cargo"]
            return annual_costs / annual_cargo

        # f_2 is dealt as a minimization problem
        def f2(x):
            return preprocess(x)["light_ship_weight"]

        def f3(x):
            return -preprocess(x)["annual_cargo"]

        def f4(x):
            return torch.sum(preprocess(x)["constraintFuncs"], axis=0)

        return [f1, f2, f3, f4]


class RE61(SyntheticProblem):
    def __init__(self, n_dim=3):

        super().__init__(
            name=self.__class__.__name__,
            n_dim=n_dim,
            n_obj=6,
            lbound=torch.tensor([0.01, 0.01, 0.01]).float(),
            ubound=torch.tensor([0.45, 0.10, 0.10]).float(),
            nadir_point=[
                8.30600282e04,
                1.34999999e03,
                2.85346906e06,
                1.60270676e07,
                3.57719742e05,
                9.96603359e04,
            ],
            ideal_point=[
                63840.2774,
                30.0,
                285346.896494,
                183749.967061,
                7.22222222222,
                0.0,
            ],
        )
        self.func = self.get_func()

    def get_func(self):

        # First original objective function
        f1 = lambda x: 106780.37 * (x[:, 1] + x[:, 2]) + 61704.67
        # Second original objective function
        f2 = lambda x: 3000 * x[:, 0]
        # Third original objective function
        f3 = lambda x: 305700 * 2289 * x[:, 1] / (0.06 * 2289) ** 0.65
        # Fourth original objective function
        f4 = lambda x: 250 * 2289 * torch.exp(-39.75 * x[:, 1] + 9.9 * x[:, 2] + 2.74)
        # Fifth original objective function
        f5 = lambda x: 25 * (1.39 / (x[:, 0] * x[:, 1]) + 4940 * x[:, 2] - 80)

        # Constraint functions
        def f6(x):
            x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]

            g1 = 1 - (0.00139 / (x0 * x1) + 4.94 * x2 - 0.08)
            g2 = 1 - (0.000306 / (x0 * x1) + 1.082 * x2 - 0.0986)
            g3 = 50000 - (12.307 / (x0 * x1) + 49408.24 * x2 + 4051.02)
            g4 = 16000 - (2.098 / (x0 * x1) + 8046.33 * x2 - 696.71)
            g5 = 10000 - (2.138 / (x0 * x1) + 7883.39 * x2 - 705.04)
            g6 = 2000 - (0.417 * x0 * x1 + 1721.26 * x2 - 136.54)
            g7 = 550 - (0.164 / (x0 * x1) + 631.13 * x2 - 54.48)

            g = torch.stack([g1, g2, g3, g4, g5, g6, g7]).float()
            z = torch.zeros(g.shape).float().to(g.device)
            g = torch.where(g < 0, -g, z)

            return torch.sum(g, axis=0)

        return [f1, f2, f3, f4, f5, f6]


class RE91(SyntheticProblem):
    def __init__(self, n_dim=7):
        super().__init__(
            name=self.__class__.__name__,
            n_obj=9,
            n_dim=n_dim,
            lbound=torch.tensor([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4]).float(),
            ubound=torch.tensor([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2]).float(),
            nadir_point=[
                42.44101351,
                1.26868725,
                150.32957866,
                0.89856422,
                1.49791586,
                1.20601073,
                1.16156617,
                1.12194854,
                1.03077619,
            ],
            ideal_point=[
                15.90721772,
                0.17780228,
                0.0,
                0.269527,
                0.47923557,
                0.58313201,
                0.83807385,
                0.69774798,
                0.77384197,
            ],
        )

    def get_x8(self, x):
        if self.x8 is not None:
            return self.x8
        mean = 0.0
        std = 1.0
        size = (x.shape[0],)  # Modify this line to match the size you need
        self.x8 = 0.006 * torch.normal(mean, std, size=size).to(self.device) + 0.345
        return self.x8

    def get_x9(self, x):
        if self.x9 is not None:
            return self.x9
        mean = 0.0
        std = 1.0
        size = (x.shape[0],)  # Modify this line to match the size you need
        self.x9 = 0.006 * torch.normal(mean, std, size=size).to(self.device) + 0.192
        return self.x9

    def get_x10(self, x):
        if self.x10 is not None:
            return self.x10
        mean = 0.0
        std = 1.0
        size = (x.shape[0],)  # Modify this line to match the size you need
        self.x10 = 10 * torch.normal(mean, std, size=size).to(self.device) + 0.0
        return self.x10

    def get_x11(self, x):
        if self.x11 is not None:
            return self.x11
        mean = 0.0
        std = 1.0
        size = (x.shape[0],)  # Modify this line to match the size you need
        self.x11 = 10 * torch.normal(mean, std, size=size).to(self.device) + 0.0
        return self.x11

    def get_func(self):
        self.x8 = None
        self.x9 = None
        self.x10 = None
        self.x11 = None

        return [
            lambda x: 1.98
            + 4.9 * x[:, 0]
            + 6.67 * x[:, 1]
            + 6.98 * x[:, 2]
            + 4.01 * x[:, 3]
            + 1.75 * x[:, 4]
            + 0.00001 * x[:, 5]
            + 2.73 * x[:, 6],
            lambda x: (
                (
                    1.16
                    - 0.3717 * x[:, 1] * x[:, 3]
                    - 0.00931 * x[:, 1] * self.get_x10(x)
                    - 0.484 * x[:, 2] * self.get_x9(x)
                    + 0.01343 * x[:, 5] * self.get_x10(x)
                )
                / 1.0
            ).clamp(min=0.0),
            lambda x: (
                (
                    0.261
                    - 0.0159 * x[:, 0] * x[:, 1]
                    - 0.188 * x[:, 0] * self.get_x8(x)
                    - 0.019 * x[:, 1] * x[:, 6]
                    + 0.0144 * x[:, 2] * x[:, 4]
                    + 0.87570001 * x[:, 4] * self.get_x10(10)
                    + 0.08045 * x[:, 5] * self.get_x9(x)
                    + 0.00139 * self.get_x8(x) * self.get_x11(x)
                    + 0.00001575 * self.get_x10(x) * self.get_x11(x)
                )
                / 0.32
            ).clamp(min=0.0),
            lambda x: (
                (
                    0.214
                    + 0.00817 * x[:, 4]
                    - 0.131 * x[:, 0] * self.get_x8(x)
                    - 0.0704 * x[:, 0] * self.get_x9(x)
                    + 0.03099 * x[:, 1] * x[:, 5]
                    - 0.018 * x[:, 1] * x[:, 6]
                    + 0.0208 * x[:, 2] * self.get_x8(x)
                    + 0.121 * x[:, 2] * self.get_x9(x)
                    - 0.00364 * x[:, 4] * x[:, 5]
                    + 0.0007715 * x[:, 4] * self.get_x10(x)
                    - 0.0005354 * x[:, 5] * self.get_x10(x)
                    + 0.00121 * self.get_x8(x) * self.get_x11(x)
                    + 0.00184 * self.get_x9(x) * self.get_x10(x)
                    - 0.018 * x[:, 1] * x[:, 1]
                )
                / 0.32
            ).clamp(min=0.0),
            lambda x: (
                (
                    0.74
                    - 0.61 * x[:, 1]
                    - 0.163 * x[:, 2] * self.get_x8(x)
                    + 0.001232 * x[:, 2] * self.get_x10(x)
                    - 0.166 * x[:, 6] * self.get_x9(x)
                    + 0.227 * x[:, 1] * x[:, 1]
                )
                / 0.32
            ).clamp(min=0.0),
            lambda x: (
                (
                    (
                        (
                            28.98
                            + 3.818 * x[:, 2]
                            - 4.2 * x[:, 0] * x[:, 1]
                            + 0.0207 * x[:, 4] * self.get_x10(x)
                            + 6.63 * x[:, 5] * self.get_x9(x)
                            - 7.77 * x[:, 6] * self.get_x8(x)
                            + 0.32 * self.get_x9(x) * self.get_x10(x)
                        )
                        + (
                            33.86
                            + 2.95 * x[:, 2]
                            + 0.1792 * self.get_x10(x)
                            - 5.057 * x[:, 0] * x[:, 1]
                            - 11 * x[:, 1] * self.get_x8(x)
                            - 0.0215 * x[:, 4] * self.get_x10(x)
                            - 9.98 * x[:, 6] * self.get_x8(x)
                            + 22 * self.get_x8(x) * self.get_x9(x)
                        )
                        + (
                            46.36
                            - 9.9 * x[:, 1]
                            - 12.9 * x[:, 0] * self.get_x8(x)
                            + 0.1107 * x[:, 2] * self.get_x10(x)
                        )
                    )
                    / 3
                )
                / 32
            ).clamp(min=0.0),
            lambda x: (
                (
                    4.72
                    - 0.5 * x[:, 3]
                    - 0.19 * x[:, 1] * x[:, 2]
                    - 0.0122 * x[:, 3] * self.get_x10(x)
                    + 0.009325 * x[:, 5] * self.get_x10(x)
                    + 0.000191 * self.get_x11(x) * self.get_x11(x)
                )
                / 4.0
            ).clamp(min=0.0),
            lambda x: (
                (
                    10.58
                    - 0.674 * x[:, 0] * x[:, 1]
                    - 1.95 * x[:, 1] * self.get_x8(x)
                    + 0.02054 * x[:, 2] * self.get_x10(x)
                    - 0.0198 * x[:, 3] * self.get_x10(x)
                    + 0.028 * x[:, 5] * self.get_x10(x)
                )
                / 9.9
            ).clamp(min=0.0),
            lambda x: (
                (
                    16.45
                    - 0.489 * x[:, 2] * x[:, 6]
                    - 0.843 * x[:, 4] * x[:, 5]
                    + 0.0432 * self.get_x9(x) * self.get_x10(x)
                    - 0.0556 * self.get_x9(x) * self.get_x11(x)
                    - 0.000786 * self.get_x11(x) * self.get_x11(x)
                )
                / 15.7
            ).clamp(min=0.0),
        ]
