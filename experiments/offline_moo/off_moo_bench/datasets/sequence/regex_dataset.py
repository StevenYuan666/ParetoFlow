from offline_moo.off_moo_bench.datasets.sequence_dataset import SequenceDataset
from offline_moo.off_moo_bench.disk_resource import DiskResource

REGEX_FILES = ["regex/regex-x-0.npy"]
REGEX_TEST_FILES = ["regex/regex-test-x-0.npy"]


class RegexDataset(SequenceDataset):

    name = "regex"
    x_name = "bigrams_sequence"
    y_name = "counts_of_each_bigram"

    @staticmethod
    def register_x_shards():
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in REGEX_FILES
        ]

    @staticmethod
    def register_y_shards():
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in REGEX_FILES
        ]

    @staticmethod
    def register_x_test_shards():
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in REGEX_TEST_FILES
        ]

    @staticmethod
    def register_y_test_shards():
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in REGEX_TEST_FILES
        ]

    def __init__(self, **kwargs):
        super(RegexDataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            self.register_x_test_shards(),
            self.register_y_test_shards(),
            **kwargs
        )
