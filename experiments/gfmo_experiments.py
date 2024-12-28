import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import offline_moo.off_moo_bench as ob
from gfmo_args import parse_args
from gfmo_nets import FlowMatching, VectorFieldNet
from gfmo_utils import (
    ALLTASKSDICT,
    DesignDataset,
    MultipleModels,
    SingleModel,
    SingleModelBaseTrainer,
    get_dataloader,
    tkwargs,
    training,
)
from offline_moo.off_moo_bench.evaluation.metrics import hv
from offline_moo.utils import get_quantile_solutions, set_seed

# get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def train_proxies(args):
    task_name = ALLTASKSDICT[args.task_name]
    print(f"Task: {task_name}")

    set_seed(args.seed)

    task = ob.make(task_name)

    X = task.x.copy()
    y = task.y.copy()

    if task.is_discrete:
        X = task.to_logits(X)
        data_size, n_dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, n_dim * n_classes)
    if task.is_sequence:
        X = task.to_logits(X)

    # For usual cases, we normalize the inputs and outputs with z-score normalization
    if args.normalization:
        X = task.normalize_x(X)
        y = task.normalize_y(y)

    n_obj = y.shape[1]
    data_size, n_dim = tuple(X.shape)
    model_save_dir = args.proxies_store_path
    os.makedirs(model_save_dir, exist_ok=True)

    model = MultipleModels(
        n_dim=n_dim,
        n_obj=n_obj,
        train_mode="Vallina",
        hidden_size=[2048, 2048],
        save_dir=args.proxies_store_path,
        save_prefix=f"MultipleModels-Vallina-{task_name}-{args.seed}",
    )
    model.set_kwargs(**tkwargs)

    trainer_func = SingleModelBaseTrainer

    for which_obj in range(n_obj):

        y0 = y[:, which_obj].copy().reshape(-1, 1)

        trainer = trainer_func(
            model=list(model.obj2model.values())[which_obj],
            which_obj=which_obj,
            args=args,
        )

        (train_loader, val_loader) = get_dataloader(
            X,
            y0,
            val_ratio=(
                1 - args.proxies_val_ratio
            ),  # means 0.9 for training and 0.1 for validation
            batch_size=args.proxies_batch_size,
        )

        trainer.launch(train_loader, val_loader)


def train_flow_matching(args):
    # Set the seed
    set_seed(args.seed)

    # Get the task
    task_name = ALLTASKSDICT[args.task_name]
    task = ob.make(task_name)
    print(f"Task: {task_name}")

    # Get the data
    X = task.x.copy()
    y = task.y.copy()

    if task.is_discrete:
        X = task.to_logits(X)
        data_size, n_dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, n_dim * n_classes)
    if task.is_sequence:
        X = task.to_logits(X)

    # For usual cases, we normalize the inputs and outputs with z-score normalization
    if args.normalization:
        X = task.normalize_x(X)
        y = task.normalize_y(y)

    # Use a subset of the data
    if args.fm_validation_size is not None:
        data_size = int(X.shape[0] - args.fm_validation_size)
        X_test = X[data_size:]
        y_test = y[data_size:]
        X = X[:data_size]
        y = y[:data_size]

    # Obtain the number of objectives
    n_obj = y.shape[1]

    # Obtain the number of data points and the number of dimensions
    data_size, n_dim = tuple(X.shape)

    print(f"Data size: {data_size}")
    print(f"Number of objectives: {n_obj}")
    print(f"Number of dimensions: {n_dim}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Create datasets
    training_dataset = DesignDataset(X)
    val_dataset = DesignDataset(X_test)

    # Create dataloaders
    training_loader = DataLoader(
        training_dataset, batch_size=args.fm_batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.fm_batch_size, shuffle=False)

    # Create the model
    name = (
        args.fm_prob_path
        + "_"
        + str(args.fm_sampling_steps)
        + "_"
        + task_name
        + "_"
        + str(args.seed)
    )
    model_store_dir = args.fm_store_path
    if not (os.path.exists(model_store_dir)):
        os.makedirs(model_store_dir)

    net = VectorFieldNet(n_dim, args.fm_hidden_size)
    net = net.to(device)
    model = FlowMatching(
        net, args.fm_sigma, n_dim, args.fm_sampling_steps, prob_path=args.fm_prob_path
    )
    model = model.to(device)

    # OPTIMIZER
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad == True], lr=args.fm_lr
    )

    # Training procedure
    nll_val = training(
        name=model_store_dir + name,
        max_patience=args.fm_patience,
        num_epochs=args.fm_epochs,
        model=model,
        optimizer=optimizer,
        training_loader=training_loader,
        val_loader=val_loader,
    )

    return nll_val


def sampling(args):
    # Set the seed
    set_seed(args.seed)

    # Get the task
    task_name = ALLTASKSDICT[args.task_name]
    task = ob.make(task_name)
    print(f"Task: {task_name}")

    # Get the data
    X = task.x.copy()
    y = task.y.copy()

    if task.is_discrete:
        X = task.to_logits(X)
        data_size, dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, dim * n_classes)
    if task.is_sequence:
        X = task.to_logits(X)

    if args.normalization:
        X = task.normalize_x(X)
        y = task.normalize_y(y)

    # Obtain the number of objectives
    n_obj = y.shape[1]

    # Set K to the number of objectives if args.K is 0
    if args.fm_K == 0:
        fm_K = n_obj + 1  # K = n_obj + 1
    else:
        fm_K = args.fm_K

    # Obtain the number of data points and the number of dimensions
    data_size, n_dim = tuple(X.shape)

    name = (
        args.fm_prob_path
        + "_"
        + str(args.fm_sampling_steps)
        + "_"
        + task_name
        + "_"
        + str(args.seed)
        + f"_K={args.fm_K}"
        + "_"
        + f"O={args.fm_O}"
        + "_"
        + f"gt={args.fm_gt}"
        + "_"
        + f"gamma={args.fm_gamma}"
    )
    model_name = args.fm_prob_path + "_" + str(1000) + "_" + task_name + "_" + str(0)
    model_store_dir = args.fm_store_path

    # Load the best model
    net = VectorFieldNet(n_dim, args.fm_hidden_size)
    net = net.to(device)
    model_best = FlowMatching(
        net, args.fm_sigma, n_dim, args.fm_sampling_steps, prob_path=args.fm_prob_path
    )
    model_best = model_best.to(device)
    model_best = torch.load(model_store_dir + model_name + ".model")
    model_best = model_best.to(device)
    print(
        f"Succesfully loaded the model from {model_store_dir + model_name + '.model'}"
    )

    # Load the classifiers
    list_of_classifiers = []
    for i in range(n_obj):
        classifier = SingleModel(
            input_size=n_dim,
            which_obj=i,
            hidden_size=[2048, 2048],
            save_dir=args.proxies_store_path,
            save_prefix=f"MultipleModels-Vallina-{task_name}-{0}",
        )
        classifier.load()
        classifier = classifier.to(device)
        classifier.eval()
        list_of_classifiers.append(classifier)
    print(f"Loaded {len(list_of_classifiers)} classifiers successfully.")

    # Conditional sampling
    x_samples, hv_results = model_best.gfmo_sample(
        list_of_classifiers,
        T=args.fm_sampling_steps,
        O=args.fm_O,
        K=fm_K,
        num_solutions=args.fm_num_solutions,
        distance=args.fm_distance_metrics,
        init_method=args.fm_init_method,
        g_t=args.fm_gt,
        task=task,
        task_name=task_name,
        t_threshold=args.fm_threshold,
        adaptive=args.fm_adaptive,
        gamma=args.fm_gamma,
    )

    # Denormalize the solutions
    res_x = x_samples
    if args.normalization:
        res_x = task.denormalize_x(res_x)

    if task.is_discrete:
        res_x = res_x.reshape(-1, dim, n_classes)
        res_x = task.to_integers(res_x)
    if task.is_sequence:
        res_x = task.to_integers(res_x)

    res_y = task.predict(res_x)
    # I noticed that there's a weird bug for task DTLZ2, the shape of res_x
    # is (batch_size, n_dim), but the shape of res_y is (n_obj, batch_size)
    # Need to fix this issue and understand why this happens
    # Simply transpose the res_y
    if res_y.shape[0] != res_x.shape[0]:
        res_y = res_y.T

    # Store the results
    if not (os.path.exists(args.samples_store_path)):
        os.makedirs(args.samples_store_path)

    np.save(args.samples_store_path + name + "_x.npy", res_x)
    np.save(args.samples_store_path + name + "_y.npy", res_y)

    if not (os.path.exists(args.results_store_path)):
        os.makedirs(args.results_store_path)

    with open(args.results_store_path + name + "_hv_results.json", "w") as f:
        json.dump(hv_results, f, indent=4)

    return res_x, res_y


def evaluation(args):
    # Set the seed
    set_seed(args.seed)

    # Get the task
    task_name = ALLTASKSDICT[args.task_name]
    task = ob.make(task_name)
    print(f"Task: {task_name}")

    # For usual cases, we normalize with z-score normalization
    # if args.normalization:
    #     task.map_normalize_x()
    #     task.map_normalize_y()

    # Get the data
    name = (
        args.fm_prob_path
        + "_"
        + str(args.fm_sampling_steps)
        + "_"
        + task_name
        + "_"
        + str(args.seed)
        + f"_K={args.fm_K}"
        + "_"
        + f"O={args.fm_O}"
        + "_"
        + f"gt={args.fm_gt}"
        + "_"
        + f"gamma={args.fm_gamma}"
    )
    res_x = np.load(args.samples_store_path + name + "_x.npy")
    res_y = np.load(args.samples_store_path + name + "_y.npy")
    print(f"Loaded the generated samples from {args.samples_store_path}")

    visible_masks = np.ones(len(res_y))
    visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
    visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0
    res_x = res_x[np.where(visible_masks == 1)[0]]
    res_y = res_y[np.where(visible_masks == 1)[0]]

    res_y_75_percent = get_quantile_solutions(res_y, 0.75)
    res_y_50_percent = get_quantile_solutions(res_y, 0.50)

    nadir_point = task.nadir_point
    # For calculating hypervolume, we use the min-max normalization
    res_y = task.normalize_y(res_y, normalization_method="min-max")
    nadir_point = task.normalize_y(nadir_point, normalization_method="min-max")
    res_y_50_percent = task.normalize_y(
        res_y_50_percent, normalization_method="min-max"
    )
    res_y_75_percent = task.normalize_y(
        res_y_75_percent, normalization_method="min-max"
    )

    _, d_best = task.get_N_non_dominated_solutions(N=256, return_x=False, return_y=True)

    d_best = task.normalize_y(d_best, normalization_method="min-max")

    d_best_hv = hv(nadir_point, d_best, task_name)
    hv_value = hv(nadir_point, res_y, task_name)
    hv_value_50_percentile = hv(nadir_point, res_y_50_percent, task_name)
    hv_value_75_percentile = hv(nadir_point, res_y_75_percent, task_name)

    print(f"Hypervolume (100th): {hv_value:4f}")
    print(f"Hypervolume (75th): {hv_value_75_percentile:4f}")
    print(f"Hypervolume (50th): {hv_value_50_percentile:4f}")
    print(f"Hypervolume (D(best)): {d_best_hv:4f}")

    # Save the results
    hv_results = {
        "hypervolume/D(best)": d_best_hv,
        "hypervolume/100th": hv_value,
        "hypervolume/75th": hv_value_75_percentile,
        "hypervolume/50th": hv_value_50_percentile,
    }

    if not (os.path.exists(args.results_store_path)):
        os.makedirs(args.results_store_path)

    with open(args.results_store_path + name + "_hv_results.json", "w") as f:
        json.dump(hv_results, f, indent=4)
