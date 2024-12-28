"""This module contains the function to parse command line arguments."""

from argparse import ArgumentParser, ArgumentTypeError

from gfmo_utils import all_task_names


def str2bool(v):
    """
    Convert a string to a boolean value. This function is used as a type for the ArgumentParser
    to avoid potential errors when parsing boolean values.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise ArgumentTypeError("Boolean value expected.")


def parse_args():  # Parse command line arguments
    """
    Parse the command line arguments for the GFMO-Guided Flows in Multi-Objective
    """
    parser = ArgumentParser(
        description="GFMO-Guided Flows in Multi-Objective Optimization"
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="The random seed for the experiment"
    )
    parser.add_argument(
        "--task_name",
        required=True,
        type=str,
        choices=all_task_names,
        help="Which task we will use",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train_proxies", "train_flow_matching", "sampling", "evaluation"],
        help="True denotes we need to train the flow matching model",
    )
    parser.add_argument(
        "--normalization",
        type=str2bool,
        nargs="?",
        default=True,
        help="True denotes normalizing the inputs and outputs",
    )
    parser.add_argument(
        "--fm_adaptive",
        type=str2bool,
        nargs="?",
        default=False,
        help="True denotes using adaptive time step in the flow matching model",
    )
    parser.add_argument(
        "--fm_validation_size",
        default=1000,
        type=int,
        help="Size of the validation set for training the flow matching model",
    )
    parser.add_argument(
        "--fm_batch_size", default=128, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--fm_prob_path", default="icfm", type=str, help="The path to the probability"
    )
    parser.add_argument(
        "--fm_hidden_size",
        default=512,
        type=int,
        help="The number of neurons in scale (s) and translation (t) nets",
    )
    parser.add_argument(
        "--fm_sigma", default=0.0, type=float, help="Sigma used in flow matching"
    )
    parser.add_argument(
        "--fm_gamma", default=2.0, type=float, help="Gamma used to control the guidance"
    )
    parser.add_argument(
        "--fm_sampling_steps",
        default=1000,
        type=int,
        help="The number of sampling steps, i.e., T",
    )
    parser.add_argument(
        "--fm_epochs",
        default=1000,
        type=int,
        help="Number of epochs to train the flow matching model",
    )
    parser.add_argument(
        "--fm_lr",
        default=1e-3,
        type=float,
        help="Learning rate for the optimizer for training the flow matching model",
    )
    parser.add_argument(
        "--fm_patience",
        default=20,
        type=int,
        help="Number of epochs to wait before early stopping",
    )
    parser.add_argument(
        "--fm_O",
        default=5,
        type=int,
        help="The number of stochastic samples we generated for each objective weights",
    )
    parser.add_argument(
        "--fm_K",
        default=0,
        type=int,
        help="The number of neighbors we used for each objective weights",
    )
    parser.add_argument(
        "--fm_num_solutions",
        default=256,
        type=int,
        help="The number of solutions we keep by using non-dominated sorting",
    )
    parser.add_argument(
        "--fm_gt",
        default=0.1,
        type=float,
        help="The coefficient for the random noise in the stochastic euler method",
    )
    parser.add_argument(
        "--fm_threshold",
        default=0.8,
        type=float,
        help="When time > threshold, we use our algorithm to generate samples",
    )
    parser.add_argument(
        "--fm_distance_metrics",
        default="cosine",
        choices=["cosine", "euclidean"],
        type=str,
        help="The metrtics used to calculate the distance between objectives weights",
    )
    parser.add_argument(
        "--fm_init_method",
        default="d_best",
        choices=["empty_init", "d_best"],
        type=str,
        help="empty_init: initialize the PS with empty solutions, d_best: initialize the PS with the best solutions in the offline dataset",
    )
    parser.add_argument(
        "--fm_store_path",
        default="flow_matching_models/",
        type=str,
        help="The folder to store the trained flow matching model",
    )
    parser.add_argument(
        "--samples_store_path",
        default="generated_samples/",
        type=str,
        help="The folder to store the generated samples",
    )
    parser.add_argument(
        "--results_store_path",
        default="saved_metrics/",
        type=str,
        help="The folder to store the computed metrics in hyper-volume",
    )
    parser.add_argument(
        "--config_store_path",
        default="log_configs/",
        type=str,
        help="The folder to store the log configurations",
    )
    parser.add_argument(
        "--proxies_store_path",
        default="proxies_model/",
        type=str,
        help="The folder to store the trained proxies model",
    )
    parser.add_argument(
        "--proxies_epochs",
        default=200,
        type=int,
        help="Number of epochs to train the proxies model",
    )
    parser.add_argument(
        "--proxies_lr",
        default=1e-3,
        type=float,
        help="Learning rate for the optimizer for training proxies model",
    )
    parser.add_argument(
        "--proxies_lr_decay",
        default=0.98,
        type=float,
        help="Learning rate decay for the optimizer for training proxies model",
    )
    parser.add_argument(
        "--proxies_batch_size",
        default=128,
        type=int,
        help="Batch size for training the proxies model",
    )
    parser.add_argument(
        "--proxies_val_ratio",
        default=0.1,
        type=float,
        help="The ratio of the validation set",
    )

    args = parser.parse_args()
    return args
