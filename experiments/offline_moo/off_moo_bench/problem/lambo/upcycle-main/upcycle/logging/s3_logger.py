import os
import pickle

import pandas as pd
import s3fs
from omegaconf import OmegaConf

from .df_logger import DataFrameLogger


class S3Logger(DataFrameLogger):
    def __init__(self, log_dir, bucket_name):
        super().__init__(log_dir)
        self.s3_file_sys = s3fs.S3FileSystem()
        self.log_dir = os.path.join(f"s3://{bucket_name}", self.log_dir)

    def write_csv(self, *args):
        for table_name, records in self.data.items():
            save_df = pd.DataFrame(records)
            df_binary = save_df.to_csv(None, index=False).encode()
            save_path = os.path.join(self.log_dir, f"{table_name}.csv")
            with self.s3_file_sys.open(save_path, "wb") as f:
                f.write(df_binary)

    def write_hydra_yaml(self, cfg):
        save_path = os.path.join(self.log_dir, ".hydra", "config.yaml")
        with self.s3_file_sys.open(save_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    def save_obj(self, obj, filename):
        save_path = os.path.join(self.log_dir, filename)
        with self.s3_file_sys.open(save_path, "wb") as f:
            pickle.dump(obj, f)

    def load_obj(self, filename):
        save_path = os.path.join(self.log_dir, filename)
        with self.s3_file_sys.open(save_path, "rb") as f:
            obj = pickle.load(f)
        return obj
