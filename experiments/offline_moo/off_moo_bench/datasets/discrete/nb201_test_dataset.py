from offline_moo.off_moo_bench.datasets.discrete_dataset import DiscreteDataset
from offline_moo.off_moo_bench.disk_resource import DiskResource

NASBENCH201TESTFILES = ["nb201_test/nb201_test-x-0.npy"]
NASBENCH201TESTTESTFILES = ["nb201_test/nb201_test-test-x-0.npy"]


class NB201TestDataset(DiscreteDataset):

    name = "nb201_test"
    x_name = "architecture"
    y_name = "test_error&params&edgegpu_latency"

    @staticmethod
    def register_x_shards():
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in NASBENCH201TESTFILES
        ]

    @staticmethod
    def register_y_shards():
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in NASBENCH201TESTFILES
        ]

    @staticmethod
    def register_x_test_shards():
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in NASBENCH201TESTTESTFILES
        ]

    @staticmethod
    def register_y_test_shards():
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in NASBENCH201TESTTESTFILES
        ]

    def __init__(self, soft_interpolation=0.6, **kwargs):
        super(NB201TestDataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            self.register_x_test_shards(),
            self.register_y_test_shards(),
            is_logits=False,
            num_classes=5,
            soft_interpolation=soft_interpolation,
            **kwargs
        )


if __name__ == "__main__":
    dataset = NB201TestDataset()
