from offline_moo.off_moo_bench.datasets.continuous_dataset import ContinuousDataset
from offline_moo.off_moo_bench.disk_resource import DiskResource

SyntheticFuncNames = [
    "vlmop1",
    "vlmop2",
    "vlmop3",
    "omnitest",
    "dtlz1",
    "dtlz2",
    "dtlz3",
    "dtlz4",
    "dtlz5",
    "dtlz6",
    "dtlz7",
    "zdt1",
    "zdt2",
    "zdt3",
    "zdt4",
    "zdt6",
]


def _get_x_files_from_name(env_name):
    return [f"{env_name}/{env_name}-x-0.npy"]


def _get_x_test_files_from_name(env_name):
    return [f"{env_name}/{env_name}-test-x-0.npy"]


class SyntheticDataset(ContinuousDataset):

    name = "synthetic_functions"
    x_name = "input_values"
    y_name = "output_values"

    @classmethod
    def register_x_shards(cls):
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in _get_x_files_from_name(cls.name)
        ]

    @classmethod
    def register_y_shards(cls):
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in _get_x_files_from_name(cls.name)
        ]

    @classmethod
    def register_x_test_shards(cls):
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in _get_x_test_files_from_name(cls.name)
        ]

    @classmethod
    def register_y_test_shards(cls):
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in _get_x_test_files_from_name(cls.name)
        ]

    def __init__(self, **kwargs):
        self.name = self.name.lower()
        assert self.name in SyntheticFuncNames
        super(SyntheticDataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            self.register_x_test_shards(),
            self.register_y_test_shards(),
            **kwargs,
        )


class VLMOP1Dataset(SyntheticDataset):
    name = "vlmop1"


class VLMOP2Dataset(SyntheticDataset):
    name = "vlmop2"


class VLMOP3Dataset(SyntheticDataset):
    name = "vlmop3"


class OmniTestDataset(SyntheticDataset):
    name = "omnitest"


class DTLZ1Dataset(SyntheticDataset):
    name = "dtlz1"


class DTLZ2Dataset(SyntheticDataset):
    name = "dtlz2"


class DTLZ3Dataset(SyntheticDataset):
    name = "dtlz3"


class DTLZ4Dataset(SyntheticDataset):
    name = "dtlz4"


class DTLZ5Dataset(SyntheticDataset):
    name = "dtlz5"


class DTLZ6Dataset(SyntheticDataset):
    name = "dtlz6"


class DTLZ7Dataset(SyntheticDataset):
    name = "dtlz7"


class ZDT1Dataset(SyntheticDataset):
    name = "zdt1"


class ZDT2Dataset(SyntheticDataset):
    name = "zdt2"


class ZDT3Dataset(SyntheticDataset):
    name = "zdt3"


class ZDT4Dataset(SyntheticDataset):
    name = "zdt4"


class ZDT6Dataset(SyntheticDataset):
    name = "zdt6"


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    dataset = VLMOP1Dataset()
