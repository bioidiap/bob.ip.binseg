#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import deque
import numpy
import torch


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)

    def update(self, value):
        self.deque.append(value)

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()


def base_measures(tp, fp, tn, fn):
    """Calculates measures from true/false positive and negative counts

    This function can return standard machine learning measures from true and
    false positive counts of positives and negatives.  For a thorough look into
    these and alternate names for the returned values, please check Wikipedia's
    entry on `Precision and Recall
    <https://en.wikipedia.org/wiki/Precision_and_recall>`_.


    Parameters
    ----------

    tp : int
        True positive count, AKA "hit"

    fp : int
        False positive count, AKA, "correct rejection"

    tn : int
        True negative count, AKA "false alarm", or "Type I error"

    fn : int
        False Negative count, AKA "miss", or "Type II error"


    Returns
    -------

    precision : float
        P, AKA positive predictive value (PPV).

    recall : float
        R, AKA sensitivity, hit rate, or true positive rate (TPR).

    specificity : float
        S, AKA selectivity or true negative rate (TNR).

    accuracy : float
        A

    jaccard : float
        J, see `Jaccard Index <https://en.wikipedia.org/wiki/Jaccard_index>`_

    f1_score : float
        F1, see `F1-score <https://en.wikipedia.org/wiki/F1_score>`_

    """

    tp = float(tp)
    tn = float(tn)
    precision = tp / (tp + fp + ((tp + fp) == 0))
    recall = tp / (tp + fn + ((tp + fn) == 0))
    specificity = tn / (fp + tn + ((fp + tn) == 0))
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    jaccard = tp / (tp + fp + fn + ((tp + fp + fn) == 0))
    f1_score = (2.0 * tp) / (2.0 * tp + fp + fn + ((2.0 * tp + fp + fn) == 0))
    # f1_score = (2.0 * precision * recall) / (precision + recall)
    return [precision, recall, specificity, accuracy, jaccard, f1_score]


def auc(x, y):
    """Calculates the area under the precision-recall curve (AUC)

    This function requires a minimum of 2 points and will use the trapezoidal
    method to calculate the area under a curve bound between ``[0.0, 1.0]``.
    It interpolates missing points if required.  The input ``x`` should be
    continuously increasing or decreasing.


    Parameters
    ----------

    x : numpy.ndarray
        A 1D numpy array containing continuously increasing or decreasing
        values for the X coordinate.

    y : numpy.ndarray
        A 1D numpy array containing the Y coordinates of the X values provided
        in ``x``.

    """

    x = numpy.array(x)
    y = numpy.array(y)

    assert len(x) == len(y), "x and y sequences must have the same length"

    dx = numpy.diff(x)
    if numpy.any(dx < 0):
        if numpy.all(dx <= 0):
            # invert direction
            x = x[::-1]
            y = y[::-1]
        else:
            raise ValueError("x is neither increasing nor decreasing "
                             ": {}.".format(x))

    y_interp = numpy.interp(
        numpy.arange(0, 1, 0.001),
        numpy.array(x),
        numpy.array(y),
        left=1.0,
        right=0.0,
    )
    return y_interp.sum() * 0.001
