import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)


def eval_precision_at_k(y_true, y_pred, k=None):
    r"""
    Precision score for top k instances with the highest outlier scores.

    Args:
        y_true (array-like): Ground truth (true labels vector).
        y_pred (array-like): Predicted labels, as returned by a classifier.
        k (int, optional): Top K instances to be considered.

    Returns:
        float: Precision at k (between 0 and 1).
    """
    if k is None:
        k = sum(y_true)

    # Get the indices that would sort the scores array in descending order
    sorted_indices = np.argsort(y_pred)[::-1]

    # Get the top-k indices
    top_k_indices = sorted_indices[:k]

    # Calculate precision at k
    precision_at_k = sum(y_true[top_k_indices]) / k

    return precision_at_k


def eval_recall_at_k(y_true, y_pred, k=None):
    r"""
    Recall score for top k instances with the highest outlier scores.

    Args:
        y_true (array-like): Ground truth (true labels vector).
        y_pred (array-like): Predicted labels, as returned by a classifier.
        k (int, optional): Top K instances to be considered.

    Returns:
        float: Recall score at k (between 0 and 1).
    """
    if k is None:
        k = sum(y_true)

    # Get the indices that would sort the scores array in descending order
    sorted_indices = np.argsort(y_pred)[::-1]

    # Get the top-k indices
    top_k_indices = sorted_indices[:k]

    # Calculate recall at k
    recall_at_k = sum(y_true[top_k_indices]) / sum(y_true)

    return recall_at_k


def eval_roc_auc(y_true, y_pred, **kwargs):
    r"""
    ROC-AUC score for binary classification.

    Args:
        y_true (array-like): Ground truth (true labels vector).
        y_pred (array-like): Predicted labels, as returned by a classifier.

    Returns:
        float: ROC-AUC score.

    See Also:
        roc_auc_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    """
    return roc_auc_score(y_true, y_pred, **kwargs)


def eval_average_precision(y_true, y_pred, **kwargs):
    r"""
    Average precision score for binary classification.

    Args:
        y_true (array-like): Ground truth (true labels vector).
        y_pred (array-like): Predicted labels, as returned by a classifier.

    Returns:
        float: Average precision score.

    See Also:
        average_precision_score:
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    """
    return average_precision_score(y_true, y_pred, **kwargs)


def eval_accuracy(y_true, y_pred, **kwargs):
    r"""
    Accuracy score for binary classification.

    Args:
        y_true (array-like): Ground truth (true labels vector).
        y_pred (array-like): Predicted labels, as returned by a classifier.

    Returns:
        float: Accuracy score.

    See Also:
        accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    """
    return accuracy_score(y_true, y_pred, **kwargs)


def eval_f1(y_true, y_pred, **kwargs):
    r"""
    F1 score for binary classification.

    Args:
        y_true (array-like): Ground truth (true labels vector).
        y_pred (array-like): Predicted labels, as returned by a classifier.

    Returns:
        float: F1 score.

    See Also:
        f1_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    return f1_score(y_true, y_pred, **kwargs)


def eval_precision(y_true, y_pred, **kwargs):
    r"""
    Precision score for binary classification.

    Args:
        y_true (array-like): Ground truth (true labels vector).
        y_pred (array-like): Predicted labels, as returned by a classifier.

    Returns:
        float: Precision score.

    See Also:
        precision_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    return precision_score(y_true, y_pred, **kwargs)


def eval_recall(y_true, y_pred, **kwargs):
    r"""
    Recall score for binary classification.

    Args:
        y_true (array-like): Ground truth (true labels vector).
        y_pred (array-like): Predicted labels, as returned by a classifier.

    Returns:
        float: Recall score.

    See Also:
        recall_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    return recall_score(y_true, y_pred, **kwargs)


def eval_metrics(y_true, y_pred, metric=None, average="weighted", **kwargs):
    # TODO: unify the args for different methods
    r"""Evaluate the models

    Args:
        y_true (np.array): True y labels.
        y_pred (np.array): Predicted y labels.
        metric (str, optional): Option of ['acc', 'f1', 'prec', 'roc_auc', 'conf_mat'].
                                Defaults to None, which eval all metrics
        average (str, optional): This parameter is required for multiclass/multilabel targets.
                                Defaults to "weighted".

    Returns:
        dict or float: metric results
    """
    if metric is None:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average=average)
        prec = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        roc_auc = roc_auc_score(y_true, y_pred, average=average)
        conf_mat = confusion_matrix(y_true, y_pred)
        return {"acc": acc,
                "f1": f1,
                "prec": prec,
                "recall": recall,
                "roc_auc": roc_auc,
                "conf_mat": conf_mat}
    else:
        if metric == 'acc':
            res = accuracy_score(y_true, y_pred)
        elif metric == "f1":
            res = f1_score(y_true, y_pred, average=average)
        elif metric == "prec":
            res = precision_score(y_true, y_pred, average=average)
        elif metric == "recall":
            res = recall_score(y_true, y_pred, average=average)
        elif metric == "roc_auc":
            res = roc_auc_score(y_true, y_pred, average=average)
        elif metric == "conf_mat":
            res = confusion_matrix(y_true, y_pred)
        return res
