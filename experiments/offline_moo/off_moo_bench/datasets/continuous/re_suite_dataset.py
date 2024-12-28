from offline_moo.off_moo_bench.datasets.continuous_dataset import ContinuousDataset
from offline_moo.off_moo_bench.disk_resource import DiskResource

RESuiteNames = [
    "re21",
    "re22",
    "re23",
    "re24",
    "re25",
    "re31",
    "re32",
    "re33",
    "re34",
    "re35",
    "re36",
    "re37",
    "re41",
    "re42",
    "re61",
]


def _get_x_files_from_name(env_name):
    return [f"{env_name}/{env_name}-x-0.npy"]


def _get_x_test_files_from_name(env_name):
    return [f"{env_name}/{env_name}-test-x-0.npy"]


class RESuiteDataset(ContinuousDataset):

    name = "re_suite"
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
        assert self.name in RESuiteNames
        super(RESuiteDataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            self.register_x_test_shards(),
            self.register_y_test_shards(),
            **kwargs,
        )


class RE21Dataset(RESuiteDataset):
    name = "re21"


class RE22Dataset(RESuiteDataset):
    name = "re22"


class RE23Dataset(RESuiteDataset):
    name = "re23"


class RE24Dataset(RESuiteDataset):
    name = "re24"


class RE25Dataset(RESuiteDataset):
    name = "re25"


class RE31Dataset(RESuiteDataset):
    name = "re31"


class RE32Dataset(RESuiteDataset):
    name = "re32"


class RE33Dataset(RESuiteDataset):
    name = "re33"


class RE34Dataset(RESuiteDataset):
    name = "re34"


class RE35Dataset(RESuiteDataset):
    name = "re35"


class RE36Dataset(RESuiteDataset):
    name = "re36"


class RE37Dataset(RESuiteDataset):
    name = "re37"


class RE41Dataset(RESuiteDataset):
    name = "re41"


class RE42Dataset(RESuiteDataset):
    name = "re42"


class RE61Dataset(RESuiteDataset):
    name = "re61"


if __name__ == "__main__":
    dataset = RE41Dataset()
