#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import cycle

import numpy
import pandas

import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)


def precision_recall_f1iso(precision, recall, names):
    """Creates a precision-recall plot of the given data.

    The plot will be annotated with F1-score iso-lines (in which the F1-score
    maintains the same value)

    Parameters
    ----------

    precision : :py:class:`numpy.ndarray` or :py:class:`list`
        A list of 1D arrays containing the Y coordinates of the plot, or the
        precision, or a 2D np array in which the rows correspond to each of the
        system's precision coordinates.

    recall : :py:class:`numpy.ndarray` or :py:class:`list`
        A list of 1D arrays containing the X coordinates of the plot, or the
        recall, or a 2D np array in which the rows correspond to each of the
        system's recall coordinates.

    names : :py:class:`list`
        An iterable over the names of each of the systems along the rows of
        ``precision`` and ``recall``


    Returns
    -------

    figure : matplotlib.figure.Figure
        A matplotlib figure you can save or display

    """

    fig, ax1 = plt.subplots(1)
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    for p, r, n in zip(precision, recall, names):
        # Plots only from the point where recall reaches its maximum, otherwise, we
        # don't see a curve...
        i = r.argmax()
        pi = p[i:]
        ri = r[i:]
        valid = (pi + ri) > 0
        f1 = 2 * (pi[valid] * ri[valid]) / (pi[valid] + ri[valid])
        # optimal point along the curve
        argmax = f1.argmax()
        opi = pi[argmax]
        ori = ri[argmax]
        # Plot Recall/Precision as threshold changes
        ax1.plot(
            ri[pi > 0],
            pi[pi > 0],
            next(linecycler),
            label="[F={:.4f}] {}".format(f1.max(), n),
        )
        ax1.plot(
            ori, opi, marker="o", linestyle=None, markersize=3, color="black"
        )
    ax1.grid(linestyle="--", linewidth=1, color="gray", alpha=0.2)
    if len(names) > 1:
        plt.legend(loc="lower left", framealpha=0.5)
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    # Annotates plot with F1-score iso-lines
    ax2 = ax1.twinx()
    f_scores = numpy.linspace(0.1, 0.9, num=9)
    tick_locs = []
    tick_labels = []
    for f_score in f_scores:
        x = numpy.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="green", alpha=0.1)
        tick_locs.append(y[-1])
        tick_labels.append("%.1f" % f_score)
    ax2.tick_params(axis="y", which="both", pad=0, right=False, left=False)
    ax2.set_ylabel("iso-F", color="green", alpha=0.3)
    ax2.set_ylim([0.0, 1.0])
    ax2.yaxis.set_label_coords(1.015, 0.97)
    ax2.set_yticks(tick_locs)  # notice these are invisible
    for k in ax2.set_yticklabels(tick_labels):
        k.set_color("green")
        k.set_alpha(0.3)
        k.set_size(8)
    # we should see some of axes 1 axes
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["left"].set_position(("data", -0.015))
    ax1.spines["bottom"].set_position(("data", -0.015))
    # we shouldn't see any of axes 2 axes
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    plt.tight_layout()
    return fig


def precision_recall_f1iso_confintval(
    precision, recall, pr_upper, pr_lower, re_upper, re_lower, names
):
    """Creates a precision-recall plot of the given data, with confidence
    intervals

    The plot will be annotated with F1-score iso-lines (in which the F1-score
    maintains the same value)

    Parameters
    ----------

    precision : :py:class:`numpy.ndarray` or :py:class:`list`
        A list of 1D arrays containing the Y coordinates of the plot, or the
        precision, or a 2D array in which the rows correspond to each
        of the system's precision coordinates.

    recall : :py:class:`numpy.ndarray` or :py:class:`list`
        A list of 1D arrays containing the X coordinates of the plot, or
        the recall, or a 2D array in which the rows correspond to each
        of the system's recall coordinates.

    pr_upper : :py:class:`numpy.ndarray` or :py:class:`list`
        A list of 1D arrays containing the upper bound of the confidence
        interval for the Y coordinates of the plot, or the precision upper
        bound, or a 2D array in which the rows correspond to each of the
        system's precision upper-bound coordinates.

    pr_lower : :py:class:`numpy.ndarray` or :py:class:`list`
        A list of 1D arrays containing the lower bound of the confidence
        interval for the Y coordinates of the plot, or the precision lower
        bound, or a 2D array in which the rows correspond to each of the
        system's precision lower-bound coordinates.

    re_upper : :py:class:`numpy.ndarray` or :py:class:`list`
        A list of 1D arrays containing the upper bound of the confidence
        interval for the Y coordinates of the plot, or the recall upper bound,
        or a 2D array in which the rows correspond to each of the system's
        recall upper-bound coordinates.

    re_lower : :py:class:`numpy.ndarray` or :py:class:`list`
        A list of 1D arrays containing the lower bound of the confidence
        interval for the Y coordinates of the plot, or the recall lower bound,
        or a 2D array in which the rows correspond to each of the system's
        recall lower-bound coordinates.

    names : :py:class:`list`
        An iterable over the names of each of the systems along the rows of
        ``precision`` and ``recall``


    Returns
    -------
    figure : matplotlib.figure.Figure
        A matplotlib figure you can save or display

    """

    fig, ax1 = plt.subplots(1)
    lines = ["-", "--", "-.", ":"]
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    colorcycler = cycle(colors)
    linecycler = cycle(lines)
    for p, r, pu, pl, ru, rl, n in zip(
        precision, recall, pr_upper, pr_lower, re_upper, re_lower, names
    ):
        # Plots only from the point where recall reaches its maximum, otherwise, we
        # don't see a curve...
        i = r.argmax()
        pi = p[i:]
        ri = r[i:]
        pui = pu[i:]
        pli = pl[i:]
        rui = ru[i:]
        rli = rl[i:]
        valid = (pi + ri) > 0
        f1 = 2 * (pi[valid] * ri[valid]) / (pi[valid] + ri[valid])
        # optimal point along the curve
        argmax = f1.argmax()
        opi = pi[argmax]
        ori = ri[argmax]
        # Plot Recall/Precision as threshold changes
        ax1.plot(
            ri[pi > 0],
            pi[pi > 0],
            next(linecycler),
            label="[F={:.4f}] {}".format(f1.max(), n),
        )
        ax1.plot(
            ori, opi, marker="o", linestyle=None, markersize=3, color="black"
        )
        # Plot confidence
        # Upper bound
        # ax1.plot(r95ui[p95ui>0], p95ui[p95ui>0])
        # Lower bound
        # ax1.plot(r95li[p95li>0], p95li[p95li>0])
        # create the limiting polygon
        vert_x = numpy.concatenate((rui[pui > 0], rli[pli > 0][::-1]))
        vert_y = numpy.concatenate((pui[pui > 0], pli[pli > 0][::-1]))
        # hacky workaround to plot 2nd human
        if numpy.isclose(numpy.mean(rui), rui[1], rtol=1e-05):
            print("found human")
            p = plt.Polygon(
                numpy.column_stack((vert_x, vert_y)),
                facecolor="none",
                alpha=0.2,
                edgecolor=next(colorcycler),
                lw=2,
            )
        else:
            p = plt.Polygon(
                numpy.column_stack((vert_x, vert_y)),
                facecolor=next(colorcycler),
                alpha=0.2,
                edgecolor="none",
                lw=0.2,
            )
        ax1.add_artist(p)

    ax1.grid(linestyle="--", linewidth=1, color="gray", alpha=0.2)
    if len(names) > 1:
        plt.legend(loc="lower left", framealpha=0.5)
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    # Annotates plot with F1-score iso-lines
    ax2 = ax1.twinx()
    f_scores = numpy.linspace(0.1, 0.9, num=9)
    tick_locs = []
    tick_labels = []
    for f_score in f_scores:
        x = numpy.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="green", alpha=0.1)
        tick_locs.append(y[-1])
        tick_labels.append("%.1f" % f_score)
    ax2.tick_params(axis="y", which="both", pad=0, right=False, left=False)
    ax2.set_ylabel("iso-F", color="green", alpha=0.3)
    ax2.set_ylim([0.0, 1.0])
    ax2.yaxis.set_label_coords(1.015, 0.97)
    ax2.set_yticks(tick_locs)  # notice these are invisible
    for k in ax2.set_yticklabels(tick_labels):
        k.set_color("green")
        k.set_alpha(0.3)
        k.set_size(8)
    # we should see some of axes 1 axes
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["left"].set_position(("data", -0.015))
    ax1.spines["bottom"].set_position(("data", -0.015))
    # we shouldn't see any of axes 2 axes
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    plt.tight_layout()
    return fig


def loss_curve(df):
    """Creates a loss curve in a Matplotlib figure.

    Parameters
    ----------

    df : :py:class:`pandas.DataFrame`
        A dataframe containing, at least, "epoch", "median-loss" and
        "learning-rate" columns, that will be plotted.

    Returns
    -------

    figure : matplotlib.figure.Figure
        A figure, that may be saved or displayed

    """

    ax1 = df.plot(x="epoch", y="median-loss", grid=True)
    ax1.set_ylabel("Median Loss")
    ax1.grid(linestyle="--", linewidth=1, color="gray", alpha=0.2)
    ax2 = df["learning-rate"].plot(secondary_y=True, legend=True, grid=True,)
    ax2.set_ylabel("Learning Rate")
    ax1.set_xlabel("Epoch")
    plt.tight_layout()
    fig = ax1.get_figure()
    return fig


def combined_precision_recall_f1iso_confintval(data):
    """Plots comparison chart of all evaluated models

    Parameters
    ----------

    data : dict
        A dict in which keys are the names of the systems and the values are
        paths to ``metrics.csv`` style files.


    Returns
    -------

    figure : matplotlib.figure.Figure
        A figure, with all systems combined into a single plot.

    """

    precisions = []
    recalls = []
    pr_ups = []
    pr_lows = []
    re_ups = []
    re_lows = []
    names = []

    for name, metrics_path in data.items():
        logger.info(f"Loading metrics from {metrics_path}...")
        df = pandas.read_csv(metrics_path)
        precisions.append(df.precision.to_numpy())
        recalls.append(df.recall.to_numpy())
        pr_ups.append(df.pr_upper.to_numpy())
        pr_lows.append(df.pr_lower.to_numpy())
        re_ups.append(df.re_upper.to_numpy())
        re_lows.append(df.re_lower.to_numpy())
        names.append(name)

    fig = precision_recall_f1iso_confintval(
        precisions, recalls, pr_ups, pr_lows, re_ups, re_lows, names
    )

    return fig
