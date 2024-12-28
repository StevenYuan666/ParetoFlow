import json
import os
from time import time

from gfmo_args import parse_args
from gfmo_experiments import evaluation, sampling, train_flow_matching, train_proxies


def main(args):
    # log config
    if not os.path.exists(args.config_store_path):
        os.makedirs(args.config_store_path)
    name = args.task_name + "_" + args.mode + "_" + str(args.seed)
    config_dict = vars(args)
    with open(os.path.join(args.config_store_path, f"{name}.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    start_time = time()
    if args.mode == "train_proxies":
        train_proxies(args)
    elif args.mode == "train_flow_matching":
        train_flow_matching(args)
    elif args.mode == "sampling":
        sampling(args)
    elif args.mode == "evaluation":
        evaluation(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    print(f"Total time: {time() - start_time} seconds")


if __name__ == "__main__":
    args = parse_args()
    main(args)
