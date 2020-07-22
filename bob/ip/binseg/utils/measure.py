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


def tricky_division(n, d):
    """Divides n by d.  Returns 0.0 in case of a division by zero"""

    return n/(d+(d==0))


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
        P, AKA positive predictive value (PPV).  It corresponds arithmetically
        to ``tp/(tp+fp)``.  In the case ``tp+fp == 0``, this function returns
        zero for precision.

    recall : float
        R, AKA sensitivity, hit rate, or true positive rate (TPR).  It
        corresponds arithmetically to ``tp/(tp+fn)``.  In the special case
        where ``tp+fn == 0``, this function returns zero for recall.

    specificity : float
        S, AKA selectivity or true negative rate (TNR).  It
        corresponds arithmetically to ``tn/(tn+fp)``.  In the special case
        where ``tn+fp == 0``, this function returns zero for specificity.

    accuracy : float
        A, see `Accuracy
        <https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers>`_. is
        the proportion of correct predictions (both true positives and true
        negatives) among the total number of pixels examined.  It corresponds
        arithmetically to ``(tp+tn)/(tp+tn+fp+fn)``.  This measure includes
        both true-negatives and positives in the numerator, what makes it
        sensitive to data or regions without annotations.

    jaccard : float
        J, see `Jaccard Index or Similarity
        <https://en.wikipedia.org/wiki/Jaccard_index>`_.  It corresponds
        arithmetically to ``tp/(tp+fp+fn)``.  In the special case where
        ``tn+fp+fn == 0``, this function returns zero for the Jaccard index.
        The Jaccard index depends on a TP-only numerator, similarly to the F1
        score.  For regions where there are no annotations, the Jaccard index
        will always be zero, irrespective of the model output.  Accuracy may be
        a better proxy if one needs to consider the true abscence of
        annotations in a region as part of the measure.

    f1_score : float
        F1, see `F1-score <https://en.wikipedia.org/wiki/F1_score>`_.  It
        corresponds arithmetically to ``2*P*R/(P+R)`` or ``2*tp/(2*tp+fp+fn)``.
        In the special case where ``P+R == (2*tp+fp+fn) == 0``, this function
        returns zero for the Jaccard index.  The F1 or Dice score depends on a
        TP-only numerator, similarly to the Jaccard index.  For regions where
        there are no annotations, the F1-score will always be zero,
        irrespective of the model output.  Accuracy may be a better proxy if
        one needs to consider the true abscence of annotations in a region as
        part of the measure.

    """

    return (
            tricky_division(tp, tp + fp),                #precision
            tricky_division(tp, tp + fn),                #recall
            tricky_division(tn, fp + tn),                #specificity
            tricky_division(tp + tn, tp + fp + fn + tn), #accuracy
            tricky_division(tp, tp + fp + fn),           #jaccard index
            tricky_division(2*tp, (2*tp) + fp + fn),     #f1-score
            )


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
