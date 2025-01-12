from offline_moo.off_moo_bench.datasets.permutation_dataset import PermutationDataset
from offline_moo.off_moo_bench.disk_resource import DiskResource

MOCVRPNames = [
    "bi_cvrp_20",
    "bi_cvrp_50",
    "bi_cvrp_100",
]


def _get_x_files_from_name(env_name):
    return [f"{env_name}/{env_name}-x-0.npy"]


def _get_x_test_files_from_name(env_name):
    return [f"{env_name}/{env_name}-test-x-0.npy"]


class MOCVRPDataset(PermutationDataset):

    name = "mo_capacitated_vehicle_routing_problem"
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
        assert self.name in MOCVRPNames
        super(MOCVRPDataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            self.register_x_test_shards(),
            self.register_y_test_shards(),
            **kwargs,
        )


class BiCVRP20Dataset(MOCVRPDataset):
    name = "bi_cvrp_20"


class BiCVRP50Dataset(MOCVRPDataset):
    name = "bi_cvrp_50"


class BiCVRP100Dataset(MOCVRPDataset):
    name = "bi_cvrp_100"
