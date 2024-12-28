import pickle as pkl
from fnmatch import fnmatch

import boto3
import s3fs
import yaml


def s3_rglob(bucket_name, root_dir, glob_pattern):
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects")
    results = []
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page["Contents"]:
            if root_dir in obj["Key"] and fnmatch(obj["Key"], glob_pattern):
                results.append(f's3://{bucket_name}/{obj["Key"]}')
    results.sort()
    return results


def s3_load_yaml(bucket_name, root_dir, glob_pattern):
    remote_files = s3_rglob(bucket_name, root_dir, glob_pattern)
    if len(remote_files) < 1:
        msg = (
            f"no files matching {glob_pattern} found in  s3://{bucket_name}/{root_dir}"
        )
        raise FileNotFoundError(msg)

    file_sys = s3fs.S3FileSystem()
    results = []
    for file in remote_files:
        with file_sys.open(file, "r") as f:
            results.append(yaml.full_load(f))
    return results, remote_files


def s3_load_obj(bucket_name, root_dir, glob_pattern):
    remote_files = s3_rglob(bucket_name, root_dir, glob_pattern)
    if len(remote_files) < 1:
        msg = (
            f"no files matching {glob_pattern} found in  s3://{bucket_name}/{root_dir}"
        )
        raise FileNotFoundError(msg)

    file_sys = s3fs.S3FileSystem()
    results = []
    for file in remote_files:
        with file_sys.open(file, "rb") as f:
            results.append(pkl.load(f))
    return results, remote_files
