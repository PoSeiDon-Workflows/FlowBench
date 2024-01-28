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
from pyod.models.lscp import LSCP
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA


def list_models():
    r""" List of models from `pyod.models`. """
    return ['ABOD',
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
