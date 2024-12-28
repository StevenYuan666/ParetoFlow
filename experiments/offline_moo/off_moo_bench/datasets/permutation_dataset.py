from offline_moo.off_moo_bench.datasets.dataset_builder import DatasetBuilder


class PermutationDataset(DatasetBuilder):

    name = "PermutationDataset"
    x_name = "Design"
    y_name = "Prediction"

    @property
    def subclass_kwargs(self):
        return dict(forbidden_normalize_x=self.forbidden_normalize_x)

    @property
    def subclass(self):
        return PermutationDataset

    def __init__(self, *args, **kwargs):
        super(PermutationDataset, self).__init__(
            forbidden_normalize_x=False, *args, **kwargs
        )
