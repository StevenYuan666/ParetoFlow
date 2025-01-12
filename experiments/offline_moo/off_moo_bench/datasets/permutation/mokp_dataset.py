from offline_moo.off_moo_bench.datasets.permutation_dataset import PermutationDataset
from offline_moo.off_moo_bench.disk_resource import DiskResource

MOKPNames = [
    "bi_kp_50",
    "bi_kp_100",
    "bi_kp_200",
]


def _get_x_files_from_name(env_name):
    return [f"{env_name}/{env_name}-x-0.npy"]


def _get_x_test_files_from_name(env_name):
    return [f"{env_name}/{env_name}-test-x-0.npy"]


class MOKPDataset(PermutationDataset):

    name = "mo_knapsack_problem"
    x_name = "order_of_chosen_item"
    y_name = "total_values"

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
        assert self.name in MOKPNames
        super(MOKPDataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            self.register_x_test_shards(),
            self.register_y_test_shards(),
            **kwargs,
        )


class BiKP50Dataset(MOKPDataset):
    name = "bi_kp_50"


class BiKP100Dataset(MOKPDataset):
    name = "bi_kp_100"


class BiKP200Dataset(MOKPDataset):
    name = "bi_kp_200"


if __name__ == "__main__":
    dataset = BiKP100Dataset()
