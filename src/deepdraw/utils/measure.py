# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy
import scipy.special
import torch


def tricky_division(n, d):
    """Divides n by d.

    Returns 0.0 in case of a division by zero
    """

    return n / (d + (d == 0))


def base_measures(tp, fp, tn, fn):
    """Calculates frequentist measures from true/false positive and negative
    counts.

    This function can return (frequentist versions of) standard machine
    learning measures from true and false positive counts of positives and
    negatives.  For a thorough look into these and alternate names for the
    returned values, please check Wikipedia's entry on `Precision and Recall
    <https://en.wikipedia.org/wiki/Precision_and_recall>`_.


    Parameters
    ----------

    tp : int
        True positive count, AKA "hit"

    fp : int
        False positive count, AKA "false alarm", or "Type I error"

    tn : int
        True negative count, AKA "correct rejection"

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
        tricky_division(tp, tp + fp),  # precision
        tricky_division(tp, tp + fn),  # recall
        tricky_division(tn, fp + tn),  # specificity
        tricky_division(tp + tn, tp + fp + fn + tn),  # accuracy
        tricky_division(tp, tp + fp + fn),  # jaccard index
        tricky_division(2 * tp, (2 * tp) + fp + fn),  # f1-score
    )


def beta_credible_region(k, i, lambda_, coverage):
    """Returns the mode, upper and lower bounds of the equal-tailed credible
    region of a probability estimate following Bernoulli trials.

    This implemetnation is based on [GOUTTE-2005]_.  It assumes :math:`k`
    successes and :math:`l` failures (:math:`n = k+l` total trials) are issued
    from a series of Bernoulli trials (likelihood is binomial).  The posterior
    is derivated using the Bayes Theorem with a beta prior.  As there is no
    reason to favour high vs.  low precision, we use a symmetric Beta prior
    (:math:`\\alpha=\\beta`):

    .. math::

       P(p|k,n) &= \\frac{P(k,n|p)P(p)}{P(k,n)} \\\\
       P(p|k,n) &= \\frac{\\frac{n!}{k!(n-k)!}p^{k}(1-p)^{n-k}P(p)}{P(k)} \\\\
       P(p|k,n) &= \\frac{1}{B(k+\\alpha, n-k+\beta)}p^{k+\\alpha-1}(1-p)^{n-k+\\beta-1} \\\\
       P(p|k,n) &= \\frac{1}{B(k+\\alpha, n-k+\\alpha)}p^{k+\\alpha-1}(1-p)^{n-k+\\alpha-1}

    The mode for this posterior (also the maximum a posteriori) is:

    .. math::

       \\text{mode}(p) = \\frac{k+\\lambda-1}{n+2\\lambda-2}

    Concretely, the prior may be flat (all rates are equally likely,
    :math:`\\lambda=1`) or we may use Jeoffrey's prior
    (:math:`\\lambda=0.5`), that is invariant through re-parameterisation.
    Jeffrey's prior indicate that rates close to zero or one are more likely.

    The mode above works if :math:`k+{\\alpha},n-k+{\\alpha} > 1`, which is
    usually the case for a resonably well tunned system, with more than a few
    samples for analysis.  In the limit of the system performance, :math:`k`
    may be 0, which will make the mode become zero.

    For our purposes, it may be more suitable to represent :math:`n = k + l`,
    with :math:`k`, the number of successes and :math:`l`, the number of
    failures in the binomial experiment, and find this more suitable
    representation:

    .. math::

       P(p|k,l) &= \\frac{1}{B(k+\\alpha, l+\\alpha)}p^{k+\\alpha-1}(1-p)^{l+\\alpha-1} \\\\
       \\text{mode}(p) &= \\frac{k+\\lambda-1}{k+l+2\\lambda-2}

    This can be mapped to most rates calculated in the context of binary
    classification this way:

    * Precision or Positive-Predictive Value (PPV): p = TP/(TP+FP), so k=TP, l=FP
    * Recall, Sensitivity, or True Positive Rate: r = TP/(TP+FN), so k=TP, l=FN
    * Specificity or True Negative Rage: s = TN/(TN+FP), so k=TN, l=FP
    * F1-score: f1 = 2TP/(2TP+FP+FN), so k=2TP, l=FP+FN
    * Accuracy: acc = TP+TN/(TP+TN+FP+FN), so k=TP+TN, l=FP+FN
    * Jaccard: j = TP/(TP+FP+FN), so k=TP, l=FP+FN

    Contrary to frequentist approaches, in which one can only
    say that if the test were repeated an infinite number of times,
    and one constructed a confidence interval each time, then X%
    of the confidence intervals would contain the true rate, here
    we can say that given our observed data, there is a X% probability
    that the true value of :math:`k/n` falls within the provided
    interval.


    .. note::

       For a disambiguation with Confidence Interval, read
       https://en.wikipedia.org/wiki/Credible_interval.


    Parameters
    ==========

    k : int
        Number of successes observed on the experiment

    i : int
        Number of failures observed on the experiment

    lambda__ : :py:class:`float`, Optional
        The parameterisation of the Beta prior to consider. Use
        :math:`\\lambda=1` for a flat prior.  Use :math:`\\lambda=0.5` for
        Jeffrey's prior (the default).

    coverage : :py:class:`float`, Optional
        A floating-point number between 0 and 1.0 indicating the
        coverage you're expecting.  A value of 0.95 will ensure 95%
        of the area under the probability density of the posterior
        is covered by the returned equal-tailed interval.


    Returns
    =======

    mean : float
        The mean of the posterior distribution

    mode : float
        The mode of the posterior distributA questão do volume eion

    lower, upper: float
        The lower and upper bounds of the credible region
    """

    # we return the equally-tailed range
    right = (1.0 - coverage) / 2  # half-width in each side
    lower = scipy.special.betaincinv(k + lambda_, i + lambda_, right)
    upper = scipy.special.betaincinv(k + lambda_, i + lambda_, 1.0 - right)

    # evaluate mean and mode (https://en.wikipedia.org/wiki/Beta_distribution)
    alpha = k + lambda_
    beta = i + lambda_

    E = alpha / (alpha + beta)

    # the mode of a beta distribution is a bit tricky
    if alpha > 1 and beta > 1:
        mode = (alpha - 1) / (alpha + beta - 2)
    elif alpha == 1 and beta == 1:
        # In the case of precision, if the threshold is close to 1.0, both TP
        # and FP can be zero, which may cause this condition to be reached, if
        # the prior is exactly 1 (flat prior).  This is a weird situation,
        # because effectively we are trying to compute the posterior when the
        # total number of experiments is zero.  So, only the prior counts - but
        # the prior is flat, so we should just pick a value.  We choose the
        # middle of the range.
        mode = 0.0  # any value would do, we just pick this one
    elif alpha <= 1 and beta > 1:
        mode = 0.0
    elif alpha > 1 and beta <= 1:
        mode = 1.0
    else:  # elif alpha < 1 and beta < 1:
        # in the case of precision, if the threshold is close to 1.0, both TP
        # and FP can be zero, which may cause this condition to be reached, if
        # the prior is smaller than 1.  This is a weird situation, because
        # effectively we are trying to compute the posterior when the total
        # number of experiments is zero.  So, only the prior counts - but the
        # prior is bimodal, so we should just pick a value.  We choose the
        # left of the range.
        mode = 0.0  # could also be 1.0 as the prior is bimodal

    return E, mode, lower, upper


def bayesian_measures(tp, fp, tn, fn, lambda_, coverage):
    """Calculates mean and mode from true/false positive and negative counts
    with credible regions.

    This function can return bayesian estimates of standard machine learning
    measures from true and false positive counts of positives and negatives.
    For a thorough look into these and alternate names for the returned values,
    please check Wikipedia's entry on `Precision and Recall
    <https://en.wikipedia.org/wiki/Precision_and_recall>`_.  See
    :py:func:`beta_credible_region` for details on the calculation of returned
    values.


    Parameters
    ----------

    tp : int
        True positive count, AKA "hit"

    fp : int
        False positive count, AKA "false alarm", or "Type I error"

    tn : int
        True negative count, AKA "correct rejection"

    fn : int
        False Negative count, AKA "miss", or "Type II error"

    lambda_ : float
        The parameterisation of the Beta prior to consider. Use
        :math:`\\lambda=1` for a flat prior.  Use :math:`\\lambda=0.5` for
        Jeffrey's prior.

    coverage : float
        A floating-point number between 0 and 1.0 indicating the
        coverage you're expecting.  A value of 0.95 will ensure 95%
        of the area under the probability density of the posterior
        is covered by the returned equal-tailed interval.



    Returns
    -------

    precision : (float, float, float, float)
        P, AKA positive predictive value (PPV), mean, mode and credible
        intervals (95% CI).  It corresponds arithmetically
        to ``tp/(tp+fp)``.

    recall : (float, float, float, float)
        R, AKA sensitivity, hit rate, or true positive rate (TPR), mean, mode
        and credible intervals (95% CI).  It corresponds arithmetically to
        ``tp/(tp+fn)``.

    specificity : (float, float, float, float)
        S, AKA selectivity or true negative rate (TNR), mean, mode and credible
        intervals (95% CI).  It corresponds arithmetically to ``tn/(tn+fp)``.

    accuracy : (float, float, float, float)
        A, mean, mode and credible intervals (95% CI).  See `Accuracy
        <https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers>`_. is
        the proportion of correct predictions (both true positives and true
        negatives) among the total number of pixels examined.  It corresponds
        arithmetically to ``(tp+tn)/(tp+tn+fp+fn)``.  This measure includes
        both true-negatives and positives in the numerator, what makes it
        sensitive to data or regions without annotations.

    jaccard : (float, float, float, float)
        J, mean, mode and credible intervals (95% CI).  See `Jaccard Index or
        Similarity <https://en.wikipedia.org/wiki/Jaccard_index>`_.  It
        corresponds arithmetically to ``tp/(tp+fp+fn)``.  The Jaccard index
        depends on a TP-only numerator, similarly to the F1 score.  For regions
        where there are no annotations, the Jaccard index will always be zero,
        irrespective of the model output.  Accuracy may be a better proxy if
        one needs to consider the true abscence of annotations in a region as
        part of the measure.

    f1_score : (float, float, float, float)
        F1, mean, mode and credible intervals (95% CI). See `F1-score
        <https://en.wikipedia.org/wiki/F1_score>`_.  It corresponds
        arithmetically to ``2*P*R/(P+R)`` or ``2*tp/(2*tp+fp+fn)``.  The F1 or
        Dice score depends on a TP-only numerator, similarly to the Jaccard
        index.  For regions where there are no annotations, the F1-score will
        always be zero, irrespective of the model output.  Accuracy may be a
        better proxy if one needs to consider the true abscence of annotations
        in a region as part of the measure.
    """

    return (
        beta_credible_region(tp, fp, lambda_, coverage),  # precision
        beta_credible_region(tp, fn, lambda_, coverage),  # recall
        beta_credible_region(tn, fp, lambda_, coverage),  # specificity
        beta_credible_region(tp + tn, fp + fn, lambda_, coverage),  # accuracy
        beta_credible_region(tp, fp + fn, lambda_, coverage),  # jaccard index
        beta_credible_region(2 * tp, fp + fn, lambda_, coverage),  # f1-score
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
            raise ValueError(
                "x is neither increasing nor decreasing " ": {}.".format(x)
            )

    y_interp = numpy.interp(
        numpy.arange(0, 1, 0.001),
        numpy.array(x),
        numpy.array(y),
        left=1.0,
        right=0.0,
    )
    return y_interp.sum() * 0.001


def get_intersection(pred_box, gt_box, multiplier):
    """Calculate intersection of boxes.

    Parameters
    ----------
    pred_box : torch.Tensor
        A 1D numpy array containing predicted box coords.

    gt_box : torch.Tensor
        A 1D numpy array containing groud truth box coords.

    multiplier: float
        A number to increase the predicted bounding box by.
    """
    x1t, y1t, x2t, y2t = gt_box.numpy()
    x1, y1, x2, y2 = pred_box.numpy()

    m = numpy.sqrt(multiplier)
    d_x = ((m * (x2 - x1)) - (x2 - x1)) / 2.0
    d_y = ((m * (y2 - y1)) - (y2 - y1)) / 2.0
    x1 = max(x1 - d_x, 0)
    x2 = x2 + d_x
    y1 = max(y1 - d_y, 0)
    y2 = y2 + d_y

    gt_tensor = gt_box.detach().clone().unsqueeze(0)
    pred_tensor = torch.tensor([x1, y1, x2, y2]).unsqueeze(0)

    lt = torch.max(pred_tensor[:, None, :2], gt_tensor[:, :2])  # [N,M,2]
    rb = torch.min(pred_tensor[:, None, 2:], gt_tensor[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    gt_area = (x2t - x1t) * (y2t - y1t)

    if gt_area > 0:
        return inter.item() / gt_area

    else:
        return 0
