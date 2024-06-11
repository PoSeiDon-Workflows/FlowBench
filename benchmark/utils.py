import argparse

from flowbench import list_workflows


def process_args():
    r""" Process args of inputs

    Returns:
        dict: Parsed arguments.
    """
    workflows = list_workflows()

    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", "-w",
                        type=str,
                        default="1000genome",
                        help="Name of workflow.",
                        choices=workflows)
    parser.add_argument("--binary",
                        action="store_true",
                        help="Toggle binary classification.")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="GPU id. `-1` for CPU only.")
    parser.add_argument("--epoch",
                        type=int,
                        default=500,
                        help="Number of epoch in training.")
    parser.add_argument("--hidden_size",
                        type=int,
                        default=64,
                        help="Hidden channel size.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Batch size.")
    parser.add_argument("--conv_blocks",
                        type=int,
                        default=2,
                        help="Number of convolutional blocks")
    parser.add_argument("--train_size",
                        type=float,
                        default=0.6,
                        help="Train size [0.5, 1). And equal split on validation and testing.")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay for Adam.")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="Dropout in neural networks.")
    parser.add_argument("--feature_option",
                        type=str,
                        default="v1",
                        help="Feature option.")
    parser.add_argument("--seed",
                        type=int,
                        default=-1,
                        help="Fix the random seed. `-1` for no random seed.")
    parser.add_argument("--path", "-p",
                        type=str,
                        default=".",
                        help="Specify the root path of file.")
    parser.add_argument("--log",
                        action="store_true",
                        help="Toggle to log the training")
    parser.add_argument("--logdir",
                        type=str,
                        default="runs",
                        help="Specify the log directory.")
    parser.add_argument("--force",
                        action="store_true",
                        help="To force reprocess datasets.")
    parser.add_argument("--balance",
                        action="store_true",
                        help="Enforce the weighted loss function.")
    parser.add_argument("--verbose", "-v",
                        action="store_true",
                        help="Toggle for verbose output.")
    parser.add_argument("--output", "-o",
                        action="store_true",
                        help="Toggle for pickle output file.")
    parser.add_argument("--anomaly_cat",
                        type=str,
                        default="all",
                        help="Specify the anomaly set.")
    parser.add_argument("--anomaly_level",
                        nargs="*",
                        help="Specify the anomaly levels. Multiple inputs.")
    parser.add_argument("--anomaly_num",
                        type=str,
                        help="Specify the anomaly num from nodes.")
    args = vars(parser.parse_args())

    return args
