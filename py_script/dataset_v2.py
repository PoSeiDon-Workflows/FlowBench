"""
* re-organize the file folders
  * zip file
"""

r""" FlowBench dataset interface for graph and tabular data."""

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

from .utils import create_dir, parse_adj


class FlowBench(InMemoryDataset):
    r"""FlowBench dataset interface for graph and tabular data.

    Args:
        root (str): Root for processing the dataset.
        name (str, optional): Name of workflow.
            Defaults to "1000genome".
        node_level (bool, optional): Use the node_level dataset if `True`.
            Defaults to True.
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
    # TODO: add the url of zip file
    cwd = os.path.abspath(__file__)
    # url = osp.join(cwd, "..", "data")
    url = "file:///home/jinh/PoSeiDon/poseidon-data/data"
    # TODO: add the url of zip file for cleaned dataset
    cleaned_url = ""

    def __init__(self,
                 root,
                 name="1000genome",
                 node_level=True,
                 binary_labels=True,
                 feature_option="v1",
                 anomaly_cat="all",
                 force_reprocess=False,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 cleaned=False,
                 **kwargs):
        self.root = root
        self.name = name.lower()
        self.node_level = node_level
        self.binary_labels = binary_labels
        self.feature_option = feature_option
        self.anomaly_cat = anomaly_cat
        self.force_reprocess = force_reprocess

        # force to reprocess the dataset by removing the processed files
        if self.force_reprocess:
            # SAVED_PATH = osp.join(osp.abspath(self.root), "processed", self.name)
            SAVED_PATH = osp.join(self.processed_dir)
            SAVED_FILE = f'{SAVED_PATH}/binary_{self.binary_labels}_node_{self.node_level}.pt'
            if osp.exists(SAVED_FILE):
                os.remove(SAVED_FILE)

        super(FlowBench, self).__init__(root, transform, pre_transform, pre_filter, **kwargs)

        out = torch.load(self.processed_paths[0])
        if not isinstance(out, tuple) or len(out) != 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, add 'force_reprocess=True' to the dataset "
                "constructor.")
        self.data, self.slices, self.sizes = out

        # if self._data.x is not None and not use_node_attr:
        #     num_node_attributes = self.num_node_attributes
        #     self._data.x = self._data.x[:, num_node_attributes:]
        # if self._data.edge_attr is not None and not use_edge_attr:
        #     num_edge_attrs = self.num_edge_attributes
        #     self._data.edge_attr = self._data.edge_attr[:, num_edge_attrs:]

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
        # SAVED_PATH = osp.join(osp.abspath(self.root), "processed", self.name)
        # SAVED_PATH = self.processed_dir
        # create_dir(SAVED_PATH)
        return [f'{self.processed_dir}/binary_{self.binary_labels}_node_{self.node_level}.pt']

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder. Override `torch_geometric.data.Dataset`.
        """
        to_folder = self.raw_dir
        path = download_url(f'{self.url}/{self.name}.zip', to_folder)
        extract_zip(path, to_folder)
        os.unlink(path)
        # shutil.rmtree(self.raw_dir)
        # os.rename(osp.join(to_folder, self.name), self.raw_dir)

    def process(self):
        r""" Processes the raw data to the graphs and saves it in :obj:`self.processed_dir`.
        """
        data_folder = osp.join(osp.dirname(osp.abspath(__file__)), "..", "parsed")

        nodes, edges = parse_adj(self.name)
        n_nodes, n_edges = len(nodes), len(edges)

        sizes = {"num_nodes_per_graph": n_nodes,
                 "num_edges_per_graph": n_edges}

        self.nx_graph = nx.DiGraph(edges)

        edge_index = torch.tensor(edges).T

        # select features
        self.features = ['auxiliary', 'compute', 'transfer'] + \
            ['is_clustered', 'ready', 'pre_script_start',
             'pre_script_end', 'submit', 'execute_start', 'execute_end',
             'post_script_start', 'post_script_end', 'wms_delay', 'pre_script_delay',
             'queue_delay', 'runtime', 'post_script_delay', 'stage_in_delay',
             'stage_in_bytes', 'stage_out_delay', 'stage_out_bytes', 'kickstart_executables_cpu_time',
             'kickstart_status', 'kickstart_executables_exitcode']
        self.new_features = ['kickstart_online_iowait',
                             'kickstart_online_bytes_read', 'kickstart_online_bytes_written',
                             'kickstart_online_read_system_calls',
                             'kickstart_online_write_system_calls', 'kickstart_online_utime',
                             'kickstart_online_stime', 'kickstart_online_bytes_read_per_second',
                             'kickstart_online_bytes_written_per_second']

        self.ts_features = ['ready', 'submit', 'execute_start', 'execute_end', 'post_script_start', 'post_script_end']
        self.delay_features = ["wms_delay", "queue_delay", "runtime",
                               "post_script_delay", "stage_in_delay", "stage_out_delay"]
        self.bytes_features = ["stage_in_bytes", "stage_out_bytes"]
        self.kickstart_features = ["kickstart_executables_cpu_time"]

        data_list = []
        feat_list = []

        if self.anomaly_cat == "all":
            self.y_labels = ["normal", "cpu", "hdd"]
        else:
            self.y_labels = ["normal", self.anomaly_cat]

        file_prefix = {"1000genome": "1000-genome",
                       "montage": "montage",
                       "predict_future_sales": "predict-future-sales",
                       "casa-wind-full": "casa-wind-full", }
        for y_idx, y_label in enumerate(self.y_labels):
            raw_data_files = f"{data_folder}/{y_label}*/{file_prefix[self.name]}*.csv"

            assert len(glob.glob(raw_data_files)) > 0, f"Incorrect anomaly cat and level {y_label}"

            for fn in glob.glob(raw_data_files):
                # read from csv file
                df = pd.read_csv(fn, index_col=[0])
                # handle missing features
                if "kickstart_executables_cpu_time" not in df.columns:
                    continue
                # handle missing nodes (the nowind workflow)
                if df.shape[0] != len(nodes):
                    continue

                # convert `type` to dummy features
                df = pd.concat([pd.get_dummies(df.type), df], axis=1)
                df = df.drop(["type"], axis=1)
                df = df.fillna(0)

                # shift timestamp by node level
                df[self.ts_features] = df[self.ts_features].sub(df[self.ts_features].ready, axis="rows")

                # process hops
                if self.name != "predict_future_sales":
                    hops = np.array([nx.shortest_path_length(self.nx_graph, 0, i) for i in range(len(nodes))])
                    df['node_hop'] = hops
                else:
                    df['node_hop'] = np.zeros(len(nodes))

                # change the index the same as `nodes`
                for i, node in enumerate(df.index.values):
                    if node.startswith("create_dir_") or node.startswith("cleanup_"):
                        new_name = node.split("-")[0]
                        df.index.values[i] = new_name

                # sort node name in json matches with node in csv.
                df = df.iloc[df.index.map(nodes).argsort()]

                # update ys from df
                if self.binary_labels:
                    if "normal" in fn:
                        y = [0] * n_nodes if self.node_level else [0]
                    else:
                        y = [1] * n_nodes if self.node_level else [1]
                else:
                    y = [y_idx] * n_nodes if self.node_level else [y_idx]
                if self.name in ['1000genome_new_2022', 'montage', 'predict_future_sales']:
                    if self.node_level:
                        # binary labels 0/1
                        y = pd.factorize(df.anomaly_type)[0]
                        # convert the `1`s to `2`s in HDD
                        if not self.binary_labels:
                            if y_label == "hdd":
                                y[np.where(y == 1)] = y_idx
                    else:
                        if not self.binary_labels:
                            y = np.array([y_idx])
                        else:
                            y = np.array([1 if y_idx > 0 else 0])
                y = torch.tensor(y)

                # extract based on selected features
                if self.feature_option == "v1":
                    selected_features = self.features + ['node_hop']
                elif self.feature_option == "v2":
                    selected_features = self.delay_features + self.bytes_features \
                        + self.kickstart_features + ['node_hop']
                elif self.feature_option == "v3":
                    selected_features = self.features + ['node_hop'] + self.new_features

                df = df[selected_features]

                x = torch.tensor(df.to_numpy().astype(np.float32), dtype=torch.float32)
                feat_list.append(df.to_numpy().astype(np.float32))
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
        # TODO: convert the node-level problem to graph-level problem
        # NOTE: no need to specify the node-level in data constructor
        raise NotImplementedError

    def __repr__(self):
        return f'{self.name({len(self)})}'
