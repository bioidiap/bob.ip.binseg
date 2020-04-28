#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import deque
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


def auc(precision, recall):
    """Calculates the area under the precision-recall curve (AUC)

    .. todo:: Integrate this to metrics reporting in compare.py
    """

    rec_unique, rec_unique_ndx = numpy.unique(recall, return_index=True)

    prec_unique = precision[rec_unique_ndx]

    if rec_unique.shape[0] > 1:
        prec_interp = numpy.interp(
            numpy.arange(0, 1, 0.01),
            rec_unique,
            prec_unique,
            left=0.0,
            right=0.0,
        )
        return prec_interp.sum() * 0.01

    return 0.0
