import os.path as osp

import torch_geometric.transforms as T

from flowbench.dataset import FlowDataset
from flowbench.transforms import MinMaxNormalizeFeatures


def load_workflow(workflow="1000genome"):
    ROOT = osp.join("/tmp", "data", "poseidon")
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
