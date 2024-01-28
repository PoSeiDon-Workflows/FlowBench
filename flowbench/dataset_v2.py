r""" FlowBench dataset interface for graph and tabular data. """

import glob
import os
import os.path as osp
import shutil

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import (Batch, Data, InMemoryDataset, download_url,
                                  extract_zip)

from .utils import parse_adj
from flowbench import list_workflows


class FlowDataset(InMemoryDataset):
    r"""FlowBench dataset interface for graph and tabular data.

    Args:
        root (str): Root for processing the dataset.
        name (str, optional): Name of workflow.
            Defaults to "1000genome".
        binary_labels (bool, optional): Specify the problem as binary classification if `True`.
            Defaults to True.
        feature_option (str, optional): Specify the feature options. More detailed options are available in `README.md`.
            Defaults to "v1".
        anomaly_cat (str, optional): Specify the anomaly category.
            Defaults to "all".
        force_reprocess (bool, optional): Force to reprocess the parsed data if `True`.
            Defaults to False.
        transform (callable, optional): Module for transform operations.
            Defaults to None.
        pre_transform (callable, optional): Module for pre_transform operations.
            Defaults to None.
        pre_filter (callable, optional): Module for pre_filter operations.
            Defaults to None.
    """
    # TODO: replace the url for the dataset
    real_dir = os.path.realpath(os.path.dirname(__file__))
    # Define the relative path to the data
    relative_path = os.path.join(real_dir, '..', 'data')
    # Use the file protocol for local file access
    url = f"file://{relative_path}"

    features = ['auxiliary',
                'compute',
                'transfer',
                'is_clustered',
                'ready',
                'pre_script_start',
                'pre_script_end',
                'submit',
                'execute_start',
                'execute_end',
                'post_script_start',
                'post_script_end',
                'wms_delay',
                'pre_script_delay',
                'queue_delay',
                'runtime',
                'post_script_delay',
                'stage_in_delay',
                'stage_in_bytes',
                'stage_out_delay',
                'stage_out_bytes',
                'kickstart_executables_cpu_time',
                'kickstart_status',
                'kickstart_executables_exitcode']
    ts_features = ['ready',
                   'submit',
                   'execute_start',
                   'execute_end',
                   'post_script_start',
                   'post_script_end']
    delay_features = ["wms_delay",
                      "queue_delay",
                      "runtime",
                      "post_script_delay",
                      "stage_in_delay",
                      "stage_out_delay"]
    bytes_features = ["stage_in_bytes",
                      "stage_out_bytes"]
    kickstart_features = ["kickstart_executables_cpu_time",
                          "kickstart_status",
                          "kickstart_executables_exitcode",
                          'kickstart_online_iowait',
                          'kickstart_online_bytes_read',
                          'kickstart_online_bytes_written',
                          'kickstart_online_read_system_calls',
                          'kickstart_online_write_system_calls',
                          'kickstart_online_utime',
                          'kickstart_online_stime',
                          'kickstart_online_bytes_read_per_second',
                          'kickstart_online_bytes_written_per_second']

    def __init__(self, root,
                 name="1000genome",
                 binary_labels=True,
                 node_level=True,
                 feature_option="v1",
                 anomaly_cat="all",
                 force_reprocess=False,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 **kwargs):
        self.root = root
        self.name = name.lower()
        self.binary_labels = binary_labels
        self.node_level = node_level
        self.feature_option = feature_option
        self.anomaly_cat = anomaly_cat
        self.force_reprocess = force_reprocess
        self.shift_ts_by_node = kwargs.get("shift_ts_by_node", True)
        self.include_hops = kwargs.get("include_hops", True)

        self.processed_fn = f'binary_{self.binary_labels}.pt'

        # force to reprocess the dataset by removing the processed files
        if self.force_reprocess:
            if osp.exists(self.processed_dir):
                shutil.rmtree(self.processed_dir)
            if osp.exists(self.raw_dir):
                shutil.rmtree(self.raw_dir)

        super(FlowDataset, self).__init__(root, transform, pre_transform, pre_filter, **kwargs)

        out = torch.load(self.processed_paths[0])
        if not isinstance(out, tuple) or len(out) != 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, add 'force_reprocess=True' to the dataset "
                "constructor.")
        self.data, self.slices, self.sizes = out

    @property
    def num_node_attributes(self):
        return self.sizes['num_node_attributes']

    @property
    def num_node_labels(self):
        return self.sizes['num_node_labels']

    @property
    def num_nodes_per_graph(self):
        return self.sizes['num_nodes_per_graph']

    @property
    def num_edges_per_graph(self):
        return self.sizes['num_edges_per_graph']

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return glob.glob(osp.join(self.raw_dir, "*.csv"))

    @property
    def processed_file_names(self):
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.

        Returns:
            list: List of file names.
        """
        return [f'{self.processed_dir}/{self.processed_fn}']

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder.
        Override `torch_geometric.data.Dataset`.
        """
        to_folder = self.raw_dir
        path = download_url(f'{self.url}/{self.name}.zip', to_folder)
        extract_zip(path, to_folder)
        os.unlink(path)
        # shutil.rmtree(self.raw_dir)
        # os.rename(osp.join(to_folder, self.name), self.raw_dir)

    def process(self):
        r""" Processes the raw data to the graphs and saves it in :obj:`self.processed_dir`. """
        nodes, edges = parse_adj(self.name)
        n_nodes, n_edges = len(nodes), len(edges)
        sizes = {"num_nodes_per_graph": n_nodes,
                 "num_edges_per_graph": n_edges}

        self.nx_graph = nx.DiGraph(edges)
        edge_index = torch.tensor(edges).T
        raw_data_files = f"{self.raw_dir}/*.csv"
        raw_data_files = glob.glob(raw_data_files)
        assert len(raw_data_files) > 0, f"No csv files found under raw folder {self.raw_dir}"

        data_list = []
        for fn in raw_data_files:
            _df = pd.read_csv(fn, index_col=[0])
            # check the original _df dimension
            assert _df.shape == (n_nodes, 44), f"{fn} has incorrect shape {_df.shape}"
            if self.shift_ts_by_node:
                # shift ts by node level
                _df[self.ts_features] = _df[self.ts_features].sub(_df[self.ts_features].ready, axis="rows")
            else:
                _df[self.ts_features] = _df[self.ts_features].min().min()

            if self.include_hops:
                # DEBUG: check the hops for predict_future_sales
                if self.name != "predict_future_sales":
                    hops = np.array([nx.shortest_path_length(self.nx_graph, 0, i) for i in range(len(nodes))])
                    _df['node_hop'] = hops
                else:
                    _df['node_hop'] = np.zeros(len(nodes))

            # sort node name in json matches with node in csv.
            _df = _df.iloc[_df.index.map(nodes).argsort()]

            # convert `type` to dummy features
            _df = pd.concat([pd.get_dummies(_df.type), _df], axis=1)
            _df = _df.drop(["type"], axis=1)
            _df = _df.fillna(0)

            # update ys from df
            if self.binary_labels:
                _df['anomaly_type'] = _df['anomaly_type'].apply(lambda x: 1 if x != 0 else 0)
            else:
                _labels = {0: 0, "cpu_2": 1, "cpu_3": 2, "hdd_5": 3, "hdd_10": 4}
                _df['anomaly_type'] = _df['anomaly_type'].map(_labels)

            if self.node_level:
                y = torch.tensor(_df['anomaly_type'].to_numpy(), dtype=torch.long)
            else:
                y = torch.tensor(_df['anomaly_type'].to_numpy().max(), dtype=torch.long)

            # extract based on selected features
            # TODO: review the feature options
            if self.feature_option == "v1":
                selected_features = self.features
            elif self.feature_option == "v2":
                selected_features = self.delay_features + self.bytes_features + self.kickstart_features
            elif self.feature_option == "v3":
                selected_features = self.features + self.kickstart_features

            if self.include_hops:
                _df = _df[selected_features + ['node_hop']]
            else:
                _df = _df[selected_features]
            x = torch.tensor(_df.to_numpy().astype(np.float32), dtype=torch.float32)
            # TODO: add y label to filter data based on label str
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        if self.node_level:
            data_batch = Batch.from_data_list(data_list)
            data = Data(x=data_batch.x, edge_index=data_batch.edge_index, y=data_batch.y)
            data = data if self.pre_transform is None else self.pre_transform(data)
            data, slices = self.collate([data])
        else:
            data, slices = self.collate(data_list)
        torch.save((data, slices, sizes), self.processed_paths[0])

    def to_graph_level(self):
        r""" Convert the node_level data to graph_level setting. """
        assert hasattr(self, "data"), "Please load the dataset first"
        _data = self.data.x.reshape(-1, self.num_nodes_per_graph, self.data.x.shape[-1])
        _y = self.data.y.reshape(-1, self.num_nodes_per_graph).max(-1)[0]
        return _data, _y

    def __repr__(self):
        return f'{self.name}(nodes: {self.num_nodes_per_graph}, edges: {self.num_edges_per_graph})'


def filter_dataset(dataset, anomaly_cat):
    r""" Filter the dataset based on the anomaly category.

    Args:
        dataset (FlowBench): The dataset to be filtered.
        anomaly_cat (str): The anomaly category to be filtered.
    """
    assert anomaly_cat in ["all", "cpu", "hdd"], f"Invalid anomaly category {anomaly_cat}"
    if anomaly_cat == "all":
        return dataset
    else:
        if dataset.binary_labels:
            anomaly_type = {"cpu": 1, "hdd": 2}
            dataset.data.y = dataset.data.y.apply(lambda x: 1 if x == anomaly_type[anomaly_cat] else 0)
        else:
            anomaly_type = {"cpu": 1, "hdd": 2}
            dataset.data.y = dataset.data.y.apply(lambda x: x if x == anomaly_type[anomaly_cat] else 0)
        return dataset


class MergeFlowDataset(InMemoryDataset):
    r""" A merged dataset of multiple FlowBench datasets.
    # TODO: verify the MergeFlowBench class
    """

    def __init__(self, root,
                 name="all",
                 binary_labels=True,
                 node_level=True,
                 feature_option="v1",
                 anomaly_cat="all",
                 force_reprocess=False,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 **kwargs):
        self.root = root
        if isinstance(name, str):
            self.workflows = list_workflows()
            self.name = name.lower()
        elif isinstance(name, list):
            self.workflows = [n.lower() for n in name if n.lower() in list_workflows()]
            # name the merged dataset as `merge_{wf1}_{wf2}_...` with first init
            self.name = "merge_" + "_".join([wf[0] for wf in self.workflows])
        self.binary_labels = binary_labels
        self.node_level = node_level
        self.feature_option = feature_option
        self.anomaly_cat = anomaly_cat
        self.force_reprocess = force_reprocess
        self.shift_ts_by_node = kwargs.get("shift_ts_by_node", True)
        self.include_hops = kwargs.get("include_hops", True)

        self.processed_fn = f'binary_{self.binary_labels}.pt'

        # force to reprocess the dataset by removing the processed files
        if self.force_reprocess:
            if osp.exists(self.processed_dir):
                shutil.rmtree(self.processed_dir)
            if osp.exists(self.raw_dir):
                shutil.rmtree(self.raw_dir)

        # process the dataset on disk
        for wf in self.workflows:
            ds = FlowDataset(self.root, wf, binary_labels, node_level,
                             feature_option, anomaly_cat, force_reprocess, **kwargs)

        super(MergeFlowDataset, self).__init__(root, transform, pre_transform, pre_filter, **kwargs)

        out = torch.load(self.processed_paths[0])
        if not isinstance(out, tuple) or len(out) != 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, add 'force_reprocess=True' to the dataset "
                "constructor.")
        self.data, self.slices, self.sizes = out

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.

        Returns:
            list: List of file names.
        """
        return [f'{self.processed_dir}/{self.processed_fn}']

    def process(self):
        data_list = []
        sizes = {}
        for wf in self.workflows:
            nodes, edges = parse_adj(wf)
            n_nodes, n_edges = len(nodes), len(edges)
            sizes_ = {"num_nodes_per_graph": n_nodes,
                      "num_edges_per_graph": n_edges}
            sizes[wf] = sizes_

        for wf in self.workflows:
            wf_path = osp.join(self.root, wf, "processed")
            wf_data = torch.load(f"{wf_path}/binary_{self.binary_labels}.pt")[0]
            data_list.append(wf_data)

        if self.node_level:
            data_batch = Batch.from_data_list(data_list, exclude_keys=["node_index"])
            data = Data(x=data_batch.x,
                        # node_index=torch.concat([d.node_index for d in data_list]),
                        edge_index=data_batch.edge_index,
                        y=data_batch.y)
            data = data if self.pre_transform is None else self.pre_transform(data)
            # torch.save(self.collate([data]), self.processed_paths[0])
            data, slices = self.collate([data])
        else:
            data, slices = self.collate(data_list)
        torch.save((data, slices, sizes), self.processed_paths[0])

    def __repr__(self):
        return f'{self.name}({len(self)})'
