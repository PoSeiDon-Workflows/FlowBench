""" Transformations for graphs.

* Column wise standardization of node features.
* Column wise min-max normalization of node features.
"""
from typing import List, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('customize_normalize_features')
class CustomizeNormalizeFeatures(BaseTransform):
    r"""Column-normalizes the attributes given in :obj:`attrs` to sum-up to one
    (functional name: :obj:`customize_normalize_features`).

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
        dim (int): The axis for normalization 0 (column-wise). 1(row-wise).
    """

    def __init__(self, attrs: List[str] = ["x"],
                 dim: int = 1) -> None:
        self.attrs = attrs
        self.dim = dim

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                value = value - value.min()
                value.div_(value.sum(dim=self.dim, keepdim=True).clamp_(min=1.))
                store[key] = value
        return data


@functional_transform('minmax_normalize_features')
class MinMaxNormalizeFeatures(BaseTransform):
    r"""Min-max normalizes the attributes given in :obj:`attrs` to scale between 0 and 1.
    (functional name: :obj:`minmax_normalize_features`).
    Args:
        attrs (List[str], optional): The names of attributes to normalize. Defaults to ["x"].
    """

    def __init__(self, attrs: List[str] = ["x"],
                 min: int = 0,
                 max: int = 1) -> None:
        self.attrs = attrs
        self.min = min
        self.max = max

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                # add a small eps for nan values
                value = value.sub(value.min(dim=0)[0]).div(value.max(dim=0)[0].sub(
                    value.min(dim=0)[0] + 1e-10))
                value = value * (self.max - self.min) + self.min
                store[key] = value
        return data


class MyFilter(object):
    r""" Filter class for the dataset.

    Args:
        data (Data): The data object.

    Example:
        `return data.num_nodes > 1` will return a boolean list of whether the graph has more than 1 node.
    """

    def __call__(self, data):
        raise NotImplementedError
