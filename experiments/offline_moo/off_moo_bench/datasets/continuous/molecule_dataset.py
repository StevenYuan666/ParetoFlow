from offline_moo.off_moo_bench.datasets.continuous_dataset import ContinuousDataset
from offline_moo.off_moo_bench.disk_resource import DiskResource

MOLECULE_FILES = ["molecule/molecule-x-0.npy"]
MOLECULE_TEST_FILES = ["molecule/molecule-test-x-0.npy"]


class MoleculeDataset(ContinuousDataset):

    name = "molecule"
    x_name = "latent_input_values"
    y_name = "properties"

    @staticmethod
    def register_x_shards():
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in MOLECULE_FILES
        ]

    @staticmethod
    def register_y_shards():
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in MOLECULE_FILES
        ]

    @staticmethod
    def register_x_test_shards():
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in MOLECULE_TEST_FILES
        ]

    @staticmethod
    def register_y_test_shards():
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in MOLECULE_TEST_FILES
        ]

    def __init__(self, **kwargs):
        super(MoleculeDataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            self.register_x_test_shards(),
            self.register_y_test_shards(),
            **kwargs
        )
