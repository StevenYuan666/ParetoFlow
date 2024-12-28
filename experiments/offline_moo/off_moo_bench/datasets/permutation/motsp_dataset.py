from offline_moo.off_moo_bench.datasets.permutation_dataset import PermutationDataset
from offline_moo.off_moo_bench.disk_resource import DiskResource

MOTSPNames = [
    "bi_tsp_20",
    "bi_tsp_50",
    "bi_tsp_100",
    "bi_tsp_500",
    "tri_tsp_20",
    "tri_tsp_50",
    "tri_tsp_100",
]


def _get_x_files_from_name(env_name):
    return [f"{env_name}/{env_name}-x-0.npy"]


def _get_x_test_files_from_name(env_name):
    return [f"{env_name}/{env_name}-test-x-0.npy"]


class MOTSPDataset(PermutationDataset):

    name = "mo_travel_salesman_problem"
    x_name = "order_of_travel_nodes"
    y_name = "total_costs"

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
        assert self.name in MOTSPNames
        super(MOTSPDataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            self.register_x_test_shards(),
            self.register_y_test_shards(),
            **kwargs,
        )


class BiTSP20Dataset(MOTSPDataset):
    name = "bi_tsp_20"


class BiTSP50Dataset(MOTSPDataset):
    name = "bi_tsp_50"


class BiTSP100Dataset(MOTSPDataset):
    name = "bi_tsp_100"


class BiTSP500Dataset(MOTSPDataset):
    name = "bi_tsp_500"


class TriTSP20Dataset(MOTSPDataset):
    name = "tri_tsp_20"


class TriTSP50Dataset(MOTSPDataset):
    name = "tri_tsp_50"


class TriTSP100Dataset(MOTSPDataset):
    name = "tri_tsp_100"
