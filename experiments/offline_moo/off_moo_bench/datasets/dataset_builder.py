import abc
import os
from typing import Optional, Union

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from offline_moo.off_moo_bench.disk_resource import DiskResource
from offline_moo.off_moo_bench.utils import get_N_nondominated_indices


class DatasetBuilder(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def x_name(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def y_name(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def subclass_kwargs(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def subclass(self):
        raise NotImplementedError

    def __init__(
        self,
        x_shards: Union[DiskResource, np.ndarray],
        y_shards: Union[DiskResource, np.ndarray],
        x_test_shards: Union[DiskResource, np.ndarray],
        y_test_shards: Union[DiskResource, np.ndarray],
        x_normalize_method: Optional[str] = "z-score",
        y_normalize_method: Optional[str] = "z-score",
        internal_batch_size: Optional[int] = 32,
        is_normalized_x: Optional[bool] = False,
        is_normalized_y: Optional[bool] = False,
        forbidden_normalize_x: Optional[bool] = False,
        max_samples: Optional[Union[int, None]] = None,
        max_percentile: Optional[float] = 100.0,
        min_percentile: Optional[float] = 0.0,
    ) -> None:

        assert x_normalize_method in ["min-max", "z-score"]
        assert y_normalize_method in ["min-max", "z-score"]
        assert 0.0 <= min_percentile <= max_percentile <= 100.0

        self.x_shards = (
            (x_shards,)
            if isinstance(x_shards, np.ndarray) or isinstance(x_shards, DiskResource)
            else x_shards
        )
        self.y_shards = (
            (y_shards,)
            if isinstance(y_shards, np.ndarray) or isinstance(y_shards, DiskResource)
            else y_shards
        )

        self.x_test_shards = (
            (x_test_shards,)
            if isinstance(x_test_shards, np.ndarray)
            or isinstance(x_test_shards, DiskResource)
            else x_test_shards
        )
        self.y_test_shards = (
            (y_test_shards,)
            if isinstance(y_test_shards, np.ndarray)
            or isinstance(y_test_shards, DiskResource)
            else y_test_shards
        )

        self.num_shards = 0
        for x_shard, y_shard in zip(self.x_shards, self.y_shards):
            self.num_shards += 1
            if isinstance(x_shard, DiskResource) and not x_shard.is_downloaded:
                raise FileNotFoundError(f"x data not found in {x_shard.disk_target}")
            if isinstance(y_shard, DiskResource) and not y_shard.is_downloaded:
                raise FileNotFoundError(f"y data not found in {y_shard.disk_target}")

        self.num_test_shards = 0
        for x_test_shard, y_test_shard in zip(self.x_test_shards, self.y_test_shards):
            self.num_test_shards += 1
            if (
                isinstance(x_test_shard, DiskResource)
                and not x_test_shard.is_downloaded
            ):
                raise FileNotFoundError(
                    f"x data not found in {x_test_shard.disk_target}"
                )
            if (
                isinstance(y_test_shard, DiskResource)
                and not y_test_shard.is_downloaded
            ):
                raise FileNotFoundError(
                    f"y data not found in {y_test_shard.disk_target}"
                )

        self.internal_batch_size = internal_batch_size
        self.forbidden_normalize_x = forbidden_normalize_x
        self.is_normalized_x = False
        self.is_normalized_y = False

        self.x_normalize_method = x_normalize_method
        self.y_normalize_method = y_normalize_method

        self.dataset_min_percentile = 0.0
        self.dataset_max_percentile = 100.0

        self.freeze_statistics = False
        self._disable_transform = False
        self._disable_subsample = False

        self.x_mean = None
        self.y_mean = None
        self.x_standard_dev = None
        self.y_standard_dev = None

        self._disable_transform = True
        self._disable_subsample = True

        self.fronts = None
        self.top_k_solutions = None

        for x, y in self.iterate_samples():
            self.input_shape = x.shape
            self.input_size = int(np.prod(x.shape))
            self.input_dtype = x.dtype
            self.output_shape = y.shape
            self.output_size = int(np.prod(y.shape))
            self.output_dtype = y.dtype
            break

        self.dataset_size = 0
        for i, y in enumerate(self.iterate_samples(return_x=False)):
            self.dataset_size += 1
            if i == 0 and y.shape[0] != self.output_size:
                raise ValueError(f"Predictions must have shape [N, n_obj]")

        (self.x_min, self.x_max, self.y_min, self.y_max) = self.get_xy_min_max()

        self._disable_transform = False
        self._disable_subsample = False
        self.dataset_visible_mask = np.full([self.dataset_size], True, dtype=np.bool_)

        if not forbidden_normalize_x and is_normalized_x:
            self.map_normalize_x()
        if is_normalized_y:
            self.map_normalize_y()
        self.subsample(
            max_samples=max_samples,
            min_percentile=min_percentile,
            max_percentile=max_percentile,
        )

    def get_num_shards(self):
        return self.num_shards

    def get_num_test_shards(self):
        return self.num_shards

    def get_shard_x(self, shard_id):
        if 0 < shard_id >= self.get_num_shards():
            raise ValueError(f"shard id={shard_id} out of bounds")

        if isinstance(self.x_shards[shard_id], np.ndarray):
            return self.x_shards[shard_id]

        elif isinstance(self.x_shards[shard_id], DiskResource):
            return np.load(self.x_shards[shard_id].disk_target)

    def get_shard_y(self, shard_id):
        if 0 < shard_id >= self.get_num_shards():
            raise ValueError(f"shard id={shard_id} out of bounds")

        if isinstance(self.y_shards[shard_id], np.ndarray):
            return self.y_shards[shard_id]

        elif isinstance(self.y_shards[shard_id], DiskResource):
            return np.load(self.y_shards[shard_id].disk_target)

    def set_shard_x(
        self, shard_id, shard_data, to_disk=None, disk_target=None, is_absolute=None
    ):

        # check that all arguments are set when saving to disk
        if (
            to_disk is not None
            and to_disk
            and (disk_target is None or is_absolute is None)
        ):
            raise ValueError("must specify location when saving to disk")

        # check the shard id is in bounds
        if 0 < shard_id >= self.get_num_shards():
            raise ValueError(f"shard id={shard_id} out of bounds")

        # store shard in memory as a numpy array
        if (to_disk is not None and not to_disk) or (
            to_disk is None and isinstance(self.x_shards[shard_id], np.ndarray)
        ):
            self.x_shards[shard_id] = shard_data

        # write shard to a new resource file given by "disk_target"
        if to_disk is not None and to_disk:
            disk_target = f"{disk_target}-x-{shard_id}.npy"
            self.x_shards[shard_id] = DiskResource(disk_target, is_absolute=is_absolute)

        # possibly write shard to an existing file on disk
        if isinstance(self.x_shards[shard_id], DiskResource):
            np.save(self.x_shards[shard_id].disk_target, shard_data)

    def set_shard_y(
        self, shard_id, shard_data, to_disk=None, disk_target=None, is_absolute=None
    ):
        # check that all arguments are set when saving to disk
        if (
            to_disk is not None
            and to_disk
            and (disk_target is None or is_absolute is None)
        ):
            raise ValueError("must specify location when saving to disk")

        # check the shard id is in bounds
        if 0 < shard_id >= self.get_num_shards():
            raise ValueError(f"shard id={shard_id} out of bounds")

        # store shard in memory as a numpy array
        if (to_disk is not None and not to_disk) or (
            to_disk is None and isinstance(self.y_shards[shard_id], np.ndarray)
        ):
            self.y_shards[shard_id] = shard_data

        # write shard to a new resource file given by "disk_target"
        if to_disk is not None and to_disk:
            disk_target = f"{disk_target}-y-{shard_id}.npy"
            self.y_shards[shard_id] = DiskResource(disk_target, is_absolute=is_absolute)

        # possibly write shard to an existing file on disk
        if isinstance(self.y_shards[shard_id], DiskResource):
            np.save(self.y_shards[shard_id].disk_target, shard_data)

    def get_shard_x_test(self, shard_id):
        if 0 < shard_id >= self.get_num_test_shards():
            raise ValueError(f"shard id={shard_id} out of bounds")

        if isinstance(self.x_test_shards[shard_id], np.ndarray):
            return self.x_test_shards[shard_id]

        elif isinstance(self.x_test_shards[shard_id], DiskResource):
            return np.load(self.x_test_shards[shard_id].disk_target)

    def get_shard_y_test(self, shard_id):
        if 0 < shard_id >= self.get_num_test_shards():
            raise ValueError(f"shard id={shard_id} out of bounds")

        if isinstance(self.y_test_shards[shard_id], np.ndarray):
            return self.y_test_shards[shard_id]

        elif isinstance(self.y_test_shards[shard_id], DiskResource):
            return np.load(self.y_test_shards[shard_id].disk_target)

    def set_shard_x_test(
        self, shard_id, shard_data, to_disk=None, disk_target=None, is_absolute=None
    ):

        # check that all arguments are set when saving to disk
        if (
            to_disk is not None
            and to_disk
            and (disk_target is None or is_absolute is None)
        ):
            raise ValueError("must specify location when saving to disk")

        # check the shard id is in bounds
        if 0 < shard_id >= self.get_num_test_shards():
            raise ValueError(f"shard id={shard_id} out of bounds")

        # store shard in memory as a numpy array
        if (to_disk is not None and not to_disk) or (
            to_disk is None and isinstance(self.x_test_shards[shard_id], np.ndarray)
        ):
            self.x_test_shards[shard_id] = shard_data

        # write shard to a new resource file given by "disk_target"
        if to_disk is not None and to_disk:
            disk_target = f"{disk_target}-test-x-{shard_id}.npy"
            self.x_test_shards[shard_id] = DiskResource(
                disk_target, is_absolute=is_absolute
            )

        # possibly write shard to an existing file on disk
        if isinstance(self.x_test_shards[shard_id], DiskResource):
            np.save(self.x_test_shards[shard_id].disk_target, shard_data)

    def set_shard_y_test(
        self, shard_id, shard_data, to_disk=None, disk_target=None, is_absolute=None
    ):
        # check that all arguments are set when saving to disk
        if (
            to_disk is not None
            and to_disk
            and (disk_target is None or is_absolute is None)
        ):
            raise ValueError("must specify location when saving to disk")

        # check the shard id is in bounds
        if 0 < shard_id >= self.get_num_test_shards():
            raise ValueError(f"shard id={shard_id} out of bounds")

        # store shard in memory as a numpy array
        if (to_disk is not None and not to_disk) or (
            to_disk is None and isinstance(self.y_test_shards[shard_id], np.ndarray)
        ):
            self.y_test_shards[shard_id] = shard_data

        # write shard to a new resource file given by "disk_target"
        if to_disk is not None and to_disk:
            disk_target = f"{disk_target}-test-y-{shard_id}.npy"
            self.y_test_shards[shard_id] = DiskResource(
                disk_target, is_absolute=is_absolute
            )

        # possibly write shard to an existing file on disk
        if isinstance(self.y_test_shards[shard_id], DiskResource):
            np.save(self.y_test_shards[shard_id].disk_target, shard_data)

    def batch_transform(self, x_batch, y_batch, return_x=True, return_y=True):
        if not self.forbidden_normalize_x and self.is_normalized_x and return_x:
            x_batch = self.normalize_x(x_batch)

        if self.is_normalized_y and return_y:
            y_batch = self.normalize_y(y_batch)

        return (x_batch if return_x else None, y_batch if return_y else None)

    def iterate_batches(
        self, batch_size, return_x=True, return_y=True, drop_remainer=False
    ):
        if batch_size < 1 or (not return_x and not return_y):
            raise ValueError("Invalid arguments passed to batch generator")

        y_batch_size = 0
        x_batch = [] if return_x else None
        y_batch = [] if return_y else None

        sample_id = 0
        for shard_id in range(self.get_num_shards()):
            x_shard_data = self.get_shard_x(shard_id) if return_x else None
            y_shard_data = self.get_shard_y(shard_id)

            shard_position = 0
            while shard_position < y_shard_data.shape[0]:
                target_size = batch_size - y_batch_size

                # slice out a component of the current shard
                x_sliced = (
                    x_shard_data[shard_position : (shard_position + target_size)]
                    if return_x
                    else None
                )
                y_sliced = y_shard_data[shard_position : (shard_position + target_size)]

                samples_read = y_sliced.shape[0]

                if not self._disable_subsample:
                    indices = np.where(
                        self.dataset_visible_mask[
                            sample_id : sample_id + y_sliced.shape[0]
                        ]
                    )[0]

                    x_sliced = x_sliced[indices] if return_x else None
                    y_sliced = y_sliced[indices] if return_y else None

                if not self._disable_transform:
                    x_sliced, y_sliced = self.batch_transform(
                        x_sliced, y_sliced, return_x=return_x, return_y=return_y
                    )

                shard_position += target_size
                sample_id += samples_read

                y_batch_size += (y_sliced if return_y else x_sliced).shape[0]
                x_batch.append(x_sliced) if return_x else None
                y_batch.append(y_sliced) if return_y else None

                if y_batch_size >= batch_size or (
                    shard_position >= y_shard_data.shape[0]
                    and shard_id + 1 == self.get_num_shards()
                    and not drop_remainer
                ):

                    try:
                        if return_x and return_y:
                            yield np.concatenate(x_batch, axis=0), np.concatenate(
                                y_batch, axis=0
                            )
                        elif return_x:
                            yield np.concatenate(x_batch, axis=0)
                        elif return_y:
                            yield np.concatenate(y_batch, axis=0)

                        y_batch_size = 0
                        x_batch = [] if return_x else None
                        y_batch = [] if return_y else None

                    except GeneratorExit:
                        return

    def iterate_samples(self, return_x=True, return_y=True):
        for batch in self.iterate_batches(
            self.internal_batch_size, return_x=return_x, return_y=return_y
        ):
            if return_x and return_y:
                for i in range(batch[0].shape[0]):
                    yield batch[0][i], batch[1][i]

            elif return_x or return_y:
                for i in range(batch.shape[0]):
                    yield batch[i]

    def iterate_test_batches(
        self, batch_size, return_x=True, return_y=True, drop_remainer=False
    ):
        if batch_size < 1 or (not return_x and not return_y):
            raise ValueError("Invalid arguments passed to batch generator")

        y_batch_size = 0
        x_test_batch = [] if return_x else None
        y_test_batch = [] if return_y else None

        sample_id = 0
        for shard_id in range(self.get_num_test_shards()):
            x_shard_data = self.get_shard_x_test(shard_id) if return_x else None
            y_shard_data = self.get_shard_y_test(shard_id)

            shard_position = 0
            while shard_position < y_shard_data.shape[0]:
                target_size = batch_size - y_batch_size

                # slice out a component of the current shard
                x_sliced = (
                    x_shard_data[shard_position : (shard_position + target_size)]
                    if return_x
                    else None
                )
                y_sliced = y_shard_data[shard_position : (shard_position + target_size)]

                samples_read = y_sliced.shape[0]

                if not self._disable_subsample:
                    indices = np.where(
                        self.dataset_visible_mask[
                            sample_id : sample_id + y_sliced.shape[0]
                        ]
                    )[0]

                    x_sliced = x_sliced[indices] if return_x else None
                    y_sliced = y_sliced[indices] if return_y else None

                if not self._disable_transform:
                    x_sliced, y_sliced = self.batch_transform(
                        x_sliced, y_sliced, return_x=return_x, return_y=return_y
                    )

                shard_position += target_size
                sample_id += samples_read

                y_batch_size += (y_sliced if return_y else x_sliced).shape[0]
                x_test_batch.append(x_sliced) if return_x else None
                y_test_batch.append(y_sliced) if return_y else None

                if y_batch_size >= batch_size or (
                    shard_position >= y_shard_data.shape[0]
                    and shard_id + 1 == self.get_num_test_shards()
                    and not drop_remainer
                ):

                    try:
                        if return_x and return_y:
                            yield np.concatenate(x_test_batch, axis=0), np.concatenate(
                                y_test_batch, axis=0
                            )
                        elif return_x:
                            yield np.concatenate(x_test_batch, axis=0)
                        elif return_y:
                            yield np.concatenate(y_test_batch, axis=0)

                        y_batch_size = 0
                        x_test_batch = [] if return_x else None
                        y_test_batch = [] if return_y else None

                    except GeneratorExit:
                        return

    def iterate_test_samples(self, return_x=True, return_y=True):
        for batch in self.iterate_test_batches(
            self.internal_batch_size, return_x=return_x, return_y=return_y
        ):
            if return_x and return_y:
                for i in range(batch[0].shape[0]):
                    yield batch[0][i], batch[1][i]

            elif return_x or return_y:
                for i in range(batch.shape[0]):
                    yield batch[i]

    def get_xy_min_max(self):
        x_min = None
        x_max = None
        y_min = None
        y_max = None
        for x, y in self.iterate_samples():
            if x_min is None:
                x_min = x.copy()
                x_max = x.copy()
                y_min = y.copy()
                y_max = y.copy()
            else:
                x_min = np.minimum(x_min, x)
                x_max = np.maximum(x_max, x)
                y_min = np.minimum(y_min, y)
                y_max = np.maximum(y_max, y)

        for x, y in self.iterate_test_samples():
            x_min = np.minimum(x_min, x)
            x_max = np.maximum(x_max, x)
            y_min = np.minimum(y_min, y)
            y_max = np.maximum(y_max, y)

        return x_min, x_max, y_min, y_max

    def __iter__(self):
        for x_batch, y_batch in self.iterate_batches(self.internal_batch_size):
            yield x_batch, y_batch

    def update_x_statistics(self):
        if self.forbidden_normalize_x:
            raise ValueError(f"normalizing x is not allowed in {self.name} dataset")

        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen.")

        original_is_normalized_x = self.is_normalized_x
        self.is_normalized_x = False

        samples = x_mean = 0
        for x_batch in self.iterate_batches(self.internal_batch_size, return_y=False):
            # calculate how many samples are actually in the current batch
            batch_size = np.array(x_batch.shape[0], dtype=np.float32)

            # update x_statistics using dynamic programming
            x_mean = x_mean * (samples / (samples + batch_size)) + np.sum(
                x_batch, axis=0, keepdims=True
            ) / (samples + batch_size)

            samples += batch_size

        samples = x_variance = 0
        for x_batch in self.iterate_batches(self.internal_batch_size, return_y=False):

            batch_size = np.array(x_batch.shape[0], dtype=np.float32)

            x_variance = x_variance * (samples / (samples + batch_size)) + np.sum(
                np.square(x_batch - x_mean), axis=0, keepdims=True
            ) / (samples + batch_size)

            samples += batch_size

        self.x_mean = x_mean
        self.x_standard_dev = np.sqrt(x_variance)

        # remove zero standard deviations to prevent singularities
        self.x_standard_dev = np.where(
            self.x_standard_dev == 0.0, 1.0, self.x_standard_dev
        )

        (self.x_min, self.x_max, self.y_min, self.y_max) = self.get_xy_min_max()

        # reset the normalized state to what it originally was
        self.is_normalized_x = original_is_normalized_x

    def update_y_statistics(self):
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen.")

        original_is_normalized_y = self.is_normalized_y
        self.is_normalized_y = False

        samples = y_mean = 0
        for y_batch in self.iterate_batches(self.internal_batch_size, return_x=False):
            # calculate how many samples are actually in the current batch
            batch_size = np.array(y_batch.shape[0], dtype=np.float32)

            # update x_statistics using dynamic programming
            y_mean = y_mean * (samples / (samples + batch_size)) + np.sum(
                y_batch, axis=0, keepdims=True
            ) / (samples + batch_size)

            samples += batch_size

        samples = y_variance = 0
        for y_batch in self.iterate_batches(self.internal_batch_size, return_x=False):
            batch_size = np.array(y_batch.shape[0], dtype=np.float32)

            y_variance = y_variance * (samples / (samples + batch_size)) + np.sum(
                np.square(y_batch - y_mean), axis=0, keepdims=True
            ) / (samples + batch_size)

            samples += batch_size

        self.y_mean = y_mean
        self.y_standard_dev = np.sqrt(y_variance)

        # remove zero standard deviations to prevent singularities
        self.y_standard_dev = np.where(
            self.y_standard_dev == 0.0, 1.0, self.y_standard_dev
        )

        (self.x_min, self.x_max, self.y_min, self.y_max) = self.get_xy_min_max()

        # reset the normalized state to what it originally was
        self.is_normalized_y = original_is_normalized_y

    def subsample(self, max_samples=None, max_percentile=100.0, min_percentile=0.0):

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # return an error is the arguments are invalid
        if max_samples is not None and max_samples <= 0:
            raise ValueError("dataset cannot be made empty")

        # return an error is the arguments are invalid
        if min_percentile > max_percentile:
            raise ValueError("invalid arguments provided")

        # convert the original prediction generator to a numpy tensor
        self._disable_subsample = True
        self._disable_transform = True

        y = np.concatenate(
            list(self.iterate_batches(self.internal_batch_size, return_x=False)), axis=0
        )
        fronts = self.regain_fronts(y)

        self._disable_subsample = False
        self._disable_transform = False

        if min_percentile != 0.0 or max_percentile != 100.0:

            best_min_percentile_indices = get_N_nondominated_indices(
                Y=y, num_ret=y.shape[0] * min_percentile, fronts=fronts
            )
            best_max_percentile_indices = get_N_nondominated_indices(
                Y=y, num_ret=y.shape[0] * max_percentile, fronts=fronts
            )

            visible_mask = np.full([y.shape[0]], False, dtype=np.bool_)
            visible_mask[best_max_percentile_indices] = True
            visible_mask[best_min_percentile_indices] = False

            self.dataset_visible_mask = visible_mask
            self.dataset_size = len(best_max_percentile_indices) - len(
                best_min_percentile_indices
            )

        else:
            self.fronts = fronts

        if not self.forbidden_normalize_x and self.is_normalized_x:
            self.update_x_statistics()

        if self.is_normalized_y:
            self.update_y_statistics()

    def regain_fronts(self, y):
        return NonDominatedSorting().do(y)

    def get_N_non_dominated_solutions(
        self, N: int, return_x=True, return_y=True, regain_fronts=False
    ):
        assert return_x or return_y, "invalid parameter setting."

        if regain_fronts or self.fronts is None:
            self.fronts = self.regain_fronts(self.y)

        N_best_indexes = get_N_nondominated_indices(
            Y=self.y, num_ret=N, fronts=self.fronts
        )

        return (
            self.x[N_best_indexes] if return_x else None,
            self.y[N_best_indexes] if return_y else None,
        )

    @property
    def x(self) -> np.ndarray:
        return np.concatenate(
            [x for x in self.iterate_batches(self.internal_batch_size, return_y=False)],
            axis=0,
        )

    @property
    def y(self) -> np.ndarray:
        return np.concatenate(
            [y for y in self.iterate_batches(self.internal_batch_size, return_x=False)],
            axis=0,
        )

    @property
    def x_test(self) -> np.ndarray:
        return np.concatenate(
            [
                x
                for x in self.iterate_test_batches(
                    self.internal_batch_size, return_y=False
                )
            ],
            axis=0,
        )

    @property
    def y_test(self) -> np.ndarray:
        return np.concatenate(
            [
                y
                for y in self.iterate_test_batches(
                    self.internal_batch_size, return_x=False
                )
            ],
            axis=0,
        )

    def relabel(
        self, relabel_function, to_disk=None, disk_target=None, is_absolute=None
    ):

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check that all arguments are set when saving to disk
        if (
            to_disk is not None
            and to_disk
            and (disk_target is None or is_absolute is None)
        ):
            raise ValueError("must specify location when saving to disk")

        # prevent the data set for being sub-sampled or normalized
        self._disable_subsample = True
        examples = self.y.shape[0]
        examples_processed = 0

        # track a list of incomplete batches between shards
        y_shard = []
        y_shard_size = 0

        # calculate the appropriate size of the first shard
        shard_id = 0
        shard = self.get_shard_y(shard_id)
        shard_size = shard.shape[0]

        # relabel the prediction values of the internal data set
        for x_batch, y_batch in self.iterate_batches(self.internal_batch_size):

            # calculate the new prediction values to be stored as shards
            y_batch = relabel_function(x_batch, y_batch)
            read_position = 0

            # remove potential normalization on the predictions
            if self.is_normalized_y:
                y_batch = self.denormalize_y(y_batch)

            # loop once per batch contained in the shard
            while read_position < y_batch.shape[0]:

                # calculate the intended number of samples to serialize
                target_size = shard_size - y_shard_size

                # slice out a component of the current shard
                y_slice = y_batch[read_position : read_position + target_size]
                samples_read = y_slice.shape[0]

                # increment the read position in the prediction tensor
                # and update the number of shards and examples processed
                read_position += target_size
                examples_processed += samples_read

                # update the current shard to be serialized
                y_shard.append(y_slice)
                y_shard_size += samples_read

                # yield the current batch when enough samples are loaded
                if y_shard_size >= shard_size or examples_processed >= examples:

                    # serialize the value of the new shard data
                    self.set_shard_y(
                        shard_id,
                        np.concatenate(y_shard, axis=0),
                        to_disk=to_disk,
                        disk_target=disk_target,
                        is_absolute=is_absolute,
                    )

                    # reset the buffer for incomplete batches
                    y_shard = []
                    y_shard_size = 0

                    # calculate the appropriate size for the next shard
                    if not examples_processed >= examples:
                        shard_id += 1
                        shard = self.get_shard_y(shard_id)
                        shard_size = shard.shape[0]

        # re-sample the data set and recalculate statistics
        self._disable_subsample = False
        self.subsample(
            max_samples=self.dataset_size,
            distribution=self.dataset_distribution,
            max_percentile=self.dataset_max_percentile,
            min_percentile=self.dataset_min_percentile,
        )

    def map_normalize_x(self):

        if self.forbidden_normalize_x:
            raise ValueError(f"normalizing x is not allowed in {self.name} dataset")

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check design values and prediction values are not normalized
        if not self.is_normalized_x:
            self.is_normalized_x = True

        # calculate the normalization statistics in advance
        self.update_x_statistics()

    def map_normalize_y(self):

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check design values and prediction values are not normalized
        if not self.is_normalized_y:
            self.is_normalized_y = True

        # calculate the normalization statistics in advance
        self.update_y_statistics()

    def normalize_x(self, x):

        if self.forbidden_normalize_x:
            raise ValueError(f"normalizing x is not allowed in {self.name} dataset")

        # calculate the mean and standard deviation of the prediction values
        if self.x_mean is None or self.x_standard_dev is None:
            self.update_x_statistics()

        # normalize the prediction values
        if self.x_normalize_method == "z-score":
            return (x - self.x_mean) / self.x_standard_dev
        elif self.x_normalize_method == "min-max":
            return (x - self.x_min) / (self.x_max - self.x_min)
        else:
            raise NotImplementedError

    def normalize_y(self, y):

        # calculate the mean and standard deviation of the prediction values
        if self.y_mean is None or self.y_standard_dev is None:
            self.update_y_statistics()

        # normalize the prediction values
        if self.y_normalize_method == "z-score":
            return (y - self.y_mean) / self.y_standard_dev
        elif self.y_normalize_method == "min-max":
            return (y - self.y_min) / (self.y_max - self.y_min)
        else:
            raise NotImplementedError

    def map_denormalize_x(self):

        if self.forbidden_normalize_x:
            raise ValueError(f"denormalizing x is not allowed in {self.name} dataset")

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check design values and prediction values are normalized
        if self.is_normalized_x:
            self.is_normalized_x = False

    def map_denormalize_y(self):

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check design values and prediction values are normalized
        if self.is_normalized_y:
            self.is_normalized_y = False

    def denormalize_x(self, x):

        if self.forbidden_normalize_x:
            raise ValueError(f"denormalizing x is not allowed in {self.name} dataset")

        # calculate the mean and standard deviation
        if self.x_mean is None or self.x_standard_dev is None:
            self.update_x_statistics()

        # denormalize the prediction values
        if self.x_normalize_method == "z-score":
            return x * self.x_standard_dev + self.x_mean
        elif self.x_normalize_method == "min-max":
            return x * (self.x_max - self.x_min) + self.x_min
        else:
            raise NotImplementedError

    def denormalize_y(self, y):

        # calculate the mean and standard deviation
        if self.y_mean is None or self.y_standard_dev is None:
            self.update_y_statistics()

        # denormalize the prediction values
        if self.y_normalize_method == "z-score":
            return y * self.y_standard_dev + self.y_mean
        elif self.y_normalize_method == "min-max":
            return y * (self.y_max - self.y_min) + self.y_min
        else:
            raise NotImplementedError


if __name__ == "__main__":
    pass
