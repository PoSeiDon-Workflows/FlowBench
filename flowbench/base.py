""" Base class for the FlowBench dataset.

Author: PoSeiDon Team
License: MIT
"""

from torch_geometric.data import InMemoryDataset
from abc import ABC
import os.path as osp
import torch


class BaseBench(InMemoryDataset, ABC):
    r""" Base class for the FlowBench dataset.

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
    url = "../data"

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
                 name=None,
                 binary_labels=True,
                 feature_option="v1",
                 anomaly_cat="all",
                 force_reprocess=False,
                 shift_ts_by_node=True,
                 include_hops=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 **kwargs):
        self.root = root
        self.name = name
        self.binary_labels = binary_labels
        self.features_option = feature_option
        self.anomaly_cat = anomaly_cat
        self.force_reprocess = force_reprocess
        self.shift_ts_by_node = shift_ts_by_node
        self.include_hops = include_hops
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        # NOTE: new processed location and files
        self.processed_fn = f'binary_{self.binary_labels}.pt'

        super(BaseBench, self).__init__(root, transform, pre_transform, pre_filter, **kwargs)
        out = torch.load(self.processed_paths[0])
        self.data, self.slices, self.sizes = out

    @property
    def processed_file_names(self):
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.

        Returns:
            list: List of file names.
        """
        return [f'{self.processed_dir}/{self.processed_fn}']
