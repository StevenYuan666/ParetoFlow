from offline_moo.off_moo_bench.datasets.dataset_builder import DatasetBuilder


class ContinuousDataset(DatasetBuilder):

    name = "ContinuousDataset"
    x_name = "Design"
    y_name = "Prediction"

    @property
    def subclass_kwargs(self):
        return dict()

    @property
    def subclass(self):
        return ContinuousDataset
