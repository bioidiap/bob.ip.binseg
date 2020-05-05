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


def base_metrics(tp, fp, tn, fn):
    """
    Calculates Precision, Recall (=Sensitivity), Specificity, Accuracy, Jaccard and F1-score (Dice)


    Parameters
    ----------

    tp : float
        True positives

    fp : float
        False positives

    tn : float
        True negatives

    fn : float
        False Negatives


    Returns
    -------

    metrics : list

    """
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

    assert len(x) == len(y)

    dx = numpy.diff(x)
    if numpy.any(dx < 0):
        if numpy.all(dx <= 0):
            # invert direction
            x = x[::-1]
            y = y[::-1]
        else:
            raise ValueError("x is neither increasing nor decreasing "
                             ": {}.".format(x))

    # avoids repeated sums for every y
    y_unique, y_unique_ndx = numpy.unique(y, return_index=True)
    x_unique = x[y_unique_ndx]

    if y_unique.shape[0] > 1:
        x_interp = numpy.interp(
            numpy.arange(0, 1, 0.001),
            y_unique,
            x_unique,
            left=0.0,
            right=0.0,
        )
        return x_interp.sum() * 0.001

    return 0.0
