import pickle

from offline_moo.off_moo_bench.datasets.continuous_dataset import ContinuousDataset
from offline_moo.off_moo_bench.disk_resource import DiskResource

MO_HOPPER_V2_FILES = ["mo_hopper_v2/mo_hopper_v2-x-0.npy"]
MO_HOPPER_V2_TEST_FILES = ["mo_hopper_v2/mo_hopper_v2-test-x-0.npy"]
PARAMS_SHAPES_FILE = "mo_hopper_v2/params_shapes.pkl"


class MOHopperV2Dataset(ContinuousDataset):

    name = "mo_hopper_v2"
    x_name = "policy_weights"
    y_name = "sum_return"

    @staticmethod
    def register_x_shards():
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in MO_HOPPER_V2_FILES
        ]

    @staticmethod
    def register_y_shards():
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in MO_HOPPER_V2_FILES
        ]

    @staticmethod
    def register_x_test_shards():
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in MO_HOPPER_V2_TEST_FILES
        ]

    @staticmethod
    def register_y_test_shards():
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in MO_HOPPER_V2_TEST_FILES
        ]

    def __init__(self, **kwargs):
        self.params_shapes = DiskResource(PARAMS_SHAPES_FILE, is_absolute=False)
        assert (
            self.params_shapes.is_downloaded
        ), f"{self.params_shapes.disk_target} not found"

        super(MOHopperV2Dataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            self.register_x_test_shards(),
            self.register_y_test_shards(),
            **kwargs,
        )
        with open(self.params_shapes.disk_target, "rb+") as f:
            self.params_shapes = pickle.load(f)
