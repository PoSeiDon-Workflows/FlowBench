"""
Unsupervised learning benchmark methods for anomaly detection.
Methods are imported from PyGOD.

We will use the same models from `pyod.models` and their default parameters.
For detailed information about the models, please refer to the documentation
of `pyod.detector`.

Citation:
    @article{liu2022bond,
        title={Bond: Benchmarking unsupervised outlier node detection on static attributed graphs},
        author={Liu, Kay and Dou, Yingtong and Zhao, Yue and Ding, Xueying and Hu, Xiyang and Zhang, Ruitong and Ding, Kaize and Chen, Canyu and Peng, Hao and Shu, Kai and Sun, Lichao and Li, Jundong and Chen, George H. and Jia, Zhihao and Yu, Philip S.},
        journal={Advances in Neural Information Processing Systems},
        volume={35},
        pages={27021--27035},
        year={2022}
    }

For more information, please refer to https://docs.pygod.org/en/latest/.

License: MIT
"""

from pygod.detector import (ANOMALOUS, CONAD, DMGD, DOMINANT, DONE, GAAN,
                            GADNR, GAE, GUIDE, OCGNN, ONE, SCAN, AdONE,
                            AnomalyDAE, CoLA, Radar)


def list_pygod():
    r""" List of models from `pygod.detector`.

    Returns:
        list: List of models.
    """
    pygod_models = [
        "AdONE",
        "ANOMALOUS",
        "AnomalyDAE",
        "CONAD",
        "DOMINANT",
        "DONE",
        "GAAN",
        "GAE",
        "GUIDE",
        "OCGNN",
        "ONE",
        "Radar",
        "SCAN",
    ]

    return pygod_models
