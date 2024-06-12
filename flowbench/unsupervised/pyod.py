"""
Unsupervised learning benchmark methods for anomaly detection.
Methods are imported from PyOD.

We will use the same models from `pyod.models` and their default parameters.
For detailed information about the models, please refer to the documentation
of `pyod.models`.

Citation:
    @article{PyOD2019,
        author = {Zhao, Yue},
        title = {PyOD: A Python Toolbox for Scalable Outlier Detection},
        year = {2019},
        publisher = {Journal of Machine Learning Research},
        journal = {JMLR},
        volume = {20},
        number = {96},
        pages = {1-7},
        url = {http://jmlr.org/papers/v20/19-011.html},
    }

For more information, please refer to https://pyod.readthedocs.io/.

License: MIT
"""
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.gmm import GMM
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.lmdd import LMDD
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA


def list_pyod():
    r""" List of models from `pyod.models`.

    Returns:
        list: List of models.
    """
    pyod_models = ['ABOD',
                   'CBLOF',
                   'FeatureBagging',
                   'GMM',
                   'HBOS',
                   'IForest',
                   'INNE',
                   'KDE',
                   'KNN',
                   'LMDD',
                   'LOF',
                   'LSCP',
                   'MCD',
                   'OCSVM',
                   'PCA']
    return pyod_models
