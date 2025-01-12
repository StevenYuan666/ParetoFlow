import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


class DiskResource:

    @staticmethod
    def get_data_path(file_path):
        return os.path.join(DATA_DIR, file_path)

    def __init__(self, disk_target, is_absolute=True):
        self.disk_target = (
            os.path.abspath(disk_target)
            if is_absolute
            else DiskResource.get_data_path(disk_target)
        )
        os.makedirs(os.path.dirname(self.disk_target), exist_ok=True)

    @property
    def is_downloaded(self):
        return os.path.exists(self.disk_target)
