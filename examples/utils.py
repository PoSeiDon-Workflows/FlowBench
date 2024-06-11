""" Utility functions for demostrations.

Author: PoSeiDon Team
Copyright: MIT
"""
import os.path as osp

import torch_geometric.transforms as T

from flowbench.dataset import FlowDataset
from flowbench.transforms import MinMaxNormalizeFeatures


def load_workflow(workflow="1000genome", root="/tmp"):
    r""" Load workflow dataset.

    Args:
        workflow (str): Name of the workflow dataset. Default: "1000genome".

    Returns:
        pyg.data.Dataset: The dataset.
    """

    ROOT = osp.join(root, "data", "flowbench")
    pre_transform = T.Compose([MinMaxNormalizeFeatures(),
                               T.ToUndirected(),
                               T.RandomNodeSplit(split="train_rest",
                                                 num_val=0.1,
                                                 num_test=0.2)])
    ds = FlowDataset(root=ROOT,
                     name=workflow,
                     binary_labels=True,
                     node_level=True,
                     pre_transform=pre_transform,
                     force_reprocess=True)
    return ds
