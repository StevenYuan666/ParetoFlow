from offline_moo.off_moo_bench.datasets.sequence_dataset import SequenceDataset
from offline_moo.off_moo_bench.disk_resource import DiskResource

MONASNames = [
    "c10mop1",
    "c10mop2",
    "c10mop3",
    "c10mop4",
    "c10mop5",
    "c10mop6",
    "c10mop7",
    "c10mop8",
    "c10mop9",
    "in1kmop1",
    "in1kmop2",
    "in1kmop3",
    "in1kmop4",
    "in1kmop5",
    "in1kmop6",
    "in1kmop7",
    "in1kmop8",
    "in1kmop9",
]


def _get_x_files_from_name(env_name):
    return [f"{env_name}/{env_name}-x-0.npy"]


def _get_x_test_files_from_name(env_name):
    return [f"{env_name}/{env_name}-test-x-0.npy"]


class MONASDataset(SequenceDataset):
    name = "mo_nas"
    x_name = "architecture"
    y_name = "mo_nas_metric"

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
        assert self.name in MONASNames
        super(MONASDataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            self.register_x_test_shards(),
            self.register_y_test_shards(),
            **kwargs,
        )


class C10MOP1Dataset(MONASDataset):
    name = "c10mop1"


class C10MOP2Dataset(MONASDataset):
    name = "c10mop2"


class C10MOP3Dataset(MONASDataset):
    name = "c10mop3"


class C10MOP4Dataset(MONASDataset):
    name = "c10mop4"


class C10MOP5Dataset(MONASDataset):
    name = "c10mop5"


class C10MOP6Dataset(MONASDataset):
    name = "c10mop6"


class C10MOP7Dataset(MONASDataset):
    name = "c10mop7"


class C10MOP8Dataset(MONASDataset):
    name = "c10mop8"


class C10MOP9Dataset(MONASDataset):
    name = "c10mop9"


class IN1KMOP1Dataset(MONASDataset):
    name = "in1kmop1"


class IN1KMOP2Dataset(MONASDataset):
    name = "in1kmop2"


class IN1KMOP3Dataset(MONASDataset):
    name = "in1kmop3"


class IN1KMOP4Dataset(MONASDataset):
    name = "in1kmop4"


class IN1KMOP5Dataset(MONASDataset):
    name = "in1kmop5"


class IN1KMOP6Dataset(MONASDataset):
    name = "in1kmop6"


class IN1KMOP7Dataset(MONASDataset):
    name = "in1kmop7"


class IN1KMOP8Dataset(MONASDataset):
    name = "in1kmop8"


class IN1KMOP9Dataset(MONASDataset):
    name = "in1kmop9"
