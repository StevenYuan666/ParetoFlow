from offline_moo.off_moo_bench.datasets.continuous_dataset import ContinuousDataset
from offline_moo.off_moo_bench.disk_resource import DiskResource

PORTFOLIO_FILES = ["portfolio/portfolio-x-0.npy"]
PORTFOLIO_TEST_FILES = ["portfolio/portfolio-test-x-0.npy"]


class PortfolioDataset(ContinuousDataset):

    name = "portfolio"
    x_name = "portfolio"
    y_name = "return&risk"

    @staticmethod
    def register_x_shards():
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in PORTFOLIO_FILES
        ]

    @staticmethod
    def register_y_shards():
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in PORTFOLIO_FILES
        ]

    @staticmethod
    def register_x_test_shards():
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in PORTFOLIO_TEST_FILES
        ]

    @staticmethod
    def register_y_test_shards():
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in PORTFOLIO_TEST_FILES
        ]

    def __init__(self, **kwargs):
        super(PortfolioDataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            self.register_x_test_shards(),
            self.register_y_test_shards(),
            **kwargs
        )
