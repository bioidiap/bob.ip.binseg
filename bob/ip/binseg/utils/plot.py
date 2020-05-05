#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
from itertools import cycle

import numpy
import pandas

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import logging

logger = logging.getLogger(__name__)


def _concave_hull(x, y, hw, hh):
    """Calculates a approximate (concave) hull from ellipse centers and sizes

    Each ellipse is approximated as a number of discrete points distributed
    over the ellipse border following an homogeneous angle distribution.


    Parameters
    ----------

    x : numpy.ndarray
        1D array with x coordinates of ellipse centers

    y : numpy.ndarray
        1D array with y coordinates of ellipse centers

    hw : numpy.ndarray
        1D array with half-widths for each ellipse

    hh : numpy.ndarray
        1D array with half-heights for each ellipse


    Returns
    -------

    points : numpy.ndarray
        2D array containing the ``(x, y)`` coordinates of the concave hull
        encompassing all defined ellipses.

    """

    def _ellipse_points(_x, _y, _hw, _hh, steps=100):
        """Generates border points for an ellipse

        This functions distributes points according to a rotation angle rather
        than uniformily with respect to a particular axis.  The result is a
        more homogeneous border representation for the ellipse.
        """
        _hw = _hw or 1e-8
        _hh = _hh or 1e-8
        angles = numpy.arange(0, numpy.pi, step=numpy.pi / (steps / 2))
        rx1 = _hw*numpy.cos(angles)
        rx2 = _hw*numpy.cos(angles + numpy.pi)
        ry1 = (_hh/_hw) * numpy.sqrt(numpy.square(_hw) - numpy.square(rx1))
        ry2 = -(_hh/_hw) * numpy.sqrt(numpy.square(_hw) - numpy.square(rx2))
        return numpy.vstack((
                numpy.hstack((rx1+_x, rx2+_x)),
                numpy.hstack((ry1+_y, ry2+_y)),
                )).T

    retval = numpy.ndarray((0,2))
    for (k, l, m, n) in zip(x, y, hw, hh):
        retval = numpy.vstack((retval, [numpy.nan, numpy.nan],
            _ellipse_points(k, l, m, n)))
    return retval


@contextlib.contextmanager
def _precision_recall_canvas(title=None):
    """Generates a canvas to draw precision-recall curves

    Works like a context manager, yielding a figure and an axes set in which
    the precision-recall curves should be added to.  The figure already
    contains F1-ISO lines and is preset to a 0-1 square region.  Once the
    context is finished, ``fig.tight_layout()`` is called.


    Parameters
    ----------

    title : :py:class:`str`, Optional
        Optional title to add to this plot


    Yields
    ------

    figure : matplotlib.figure.Figure
        The figure that should be finally returned to the user

    axes : matplotlib.figure.Axes
        An axis set where to precision-recall plots should be added to

    """

    fig, axes1 = plt.subplots(1)

    # Names and bounds
    axes1.set_xlabel("Recall")
    axes1.set_ylabel("Precision")
    axes1.set_xlim([0.0, 1.0])
    axes1.set_ylim([0.0, 1.0])

    if title is not None:
        axes1.set_title(title)

    axes1.grid(linestyle="--", linewidth=1, color="gray", alpha=0.2)
    axes2 = axes1.twinx()

    # Annotates plot with F1-score iso-lines
    f_scores = numpy.linspace(0.1, 0.9, num=9)
    tick_locs = []
    tick_labels = []
    for f_score in f_scores:
        x = numpy.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="green", alpha=0.1)
        tick_locs.append(y[-1])
        tick_labels.append("%.1f" % f_score)
    axes2.tick_params(axis="y", which="both", pad=0, right=False, left=False)
    axes2.set_ylabel("iso-F", color="green", alpha=0.3)
    axes2.set_ylim([0.0, 1.0])
    axes2.yaxis.set_label_coords(1.015, 0.97)
    axes2.set_yticks(tick_locs)  # notice these are invisible
    for k in axes2.set_yticklabels(tick_labels):
        k.set_color("green")
        k.set_alpha(0.3)
        k.set_size(8)

    # we should see some of axes 1 axes
    axes1.spines["right"].set_visible(False)
    axes1.spines["top"].set_visible(False)
    axes1.spines["left"].set_position(("data", -0.015))
    axes1.spines["bottom"].set_position(("data", -0.015))

    # we shouldn't see any of axes 2 axes
    axes2.spines["right"].set_visible(False)
    axes2.spines["top"].set_visible(False)
    axes2.spines["left"].set_visible(False)
    axes2.spines["bottom"].set_visible(False)

    # yield execution, lets user draw precision-recall plots, and the legend
    # before tighteneing the layout
    yield fig, axes1

    plt.tight_layout()


def precision_recall_f1iso(data, confidence=True):
    """Creates a precision-recall plot with confidence intervals

    This function creates and returns a Matplotlib figure with a
    precision-recall plot containing shaded confidence intervals (standard
    deviation on the precision-recall measurements).  The plot will be
    annotated with F1-score iso-lines (in which the F1-score maintains the same
    value).

    This function specially supports "second-annotator" entries by plotting a
    line showing the comparison between the default annotator being analyzed
    and a second "opinion".  Second annotator dataframes contain a single
    entry (threshold=0.5), given the nature of the binary map comparisons.


    Parameters
    ----------

    data : dict
        A dictionary in which keys are strings defining plot labels and values
        are dictionaries with two entries:

        * ``df``: :py:class:`pandas.DataFrame`

          A dataframe that is produced by our evaluator engine, indexed by
          integer "thresholds", containing the following columns: ``threshold``
          (sorted ascending), ``precision``, ``recall``, ``pr_upper`` (upper
          precision bounds), ``pr_lower`` (lower precision bounds),
          ``re_upper`` (upper recall bounds), ``re_lower`` (lower recall
          bounds).

        * ``threshold``: :py:class:`list`

          A threshold to graph with a dot for each set.    Specific
          threshold values do not affect "second-annotator" dataframes.

    confidence : :py:class:`bool`, Optional
        If set, draw confidence intervals for each line, using ``*_upper`` and
        ``*_lower`` entries.


    Returns
    -------

    figure : matplotlib.figure.Figure
        A matplotlib figure you can save or display (uses an ``agg`` backend)

    """

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

    with _precision_recall_canvas(title=None) as (fig, axes):

        legend = []

        for name, value in data.items():

            df = value["df"]
            threshold = value["threshold"]

            # plots only from the point where recall reaches its maximum,
            # otherwise, we don't see a curve...
            max_recall = df["recall"].idxmax()
            pi = df.precision[max_recall:]
            ri = df.recall[max_recall:]

            valid = (pi + ri) > 0
            f1 = 2 * (pi[valid] * ri[valid]) / (pi[valid] + ri[valid])

            # optimal point along the curve
            bins = len(df)
            index = int(round(bins * threshold))
            index = min(index, len(df) - 1)  # avoids out of range indexing

            # plots Recall/Precision as threshold changes
            label = f"{name} (F1={df.f1_score[index]:.4f})"
            color = next(colorcycler)

            if len(df) == 1:
                # plot black dot for F1-score at select threshold
                (marker,) = axes.plot(
                    df.recall[index],
                    df.precision[index],
                    marker="*",
                    markersize=6,
                    color=color,
                    alpha=0.8,
                    linestyle="None",
                )
                (line,) = axes.plot(
                    df.recall[index],
                    df.precision[index],
                    linestyle="None",
                    color=color,
                    alpha=0.2,
                )
                legend.append(([marker, line], label))
            else:
                # line first, so marker gets on top
                style = next(linecycler)
                (line,) = axes.plot(
                    ri[pi > 0], pi[pi > 0], color=color, linestyle=style
                )
                (marker,) = axes.plot(
                    df.recall[index],
                    df.precision[index],
                    marker="o",
                    linestyle=style,
                    markersize=4,
                    color=color,
                    alpha=0.8,
                )
                legend.append(([marker, line], label))

            if confidence:

                # hacky workaround to plot 2nd human
                if len(df) == 1:  # binary system, very likely
                    logger.warning("Found 2nd human annotator - patching...")
                    p = Ellipse(
                        (df.recall.iloc[0], df.precision.iloc[0]),
                        2 * df.std_re.iloc[0],
                        2 * df.std_pr.iloc[0],
                        angle=0,
                        color=color,
                        alpha=0.1,
                        linewidth=0,
                    )
                else:
                    hull = _concave_hull(
                        df.recall, df.precision, df.std_re, df.std_pr
                    )
                    p = plt.Polygon(hull,
                        facecolor=color,
                        alpha=0.2,
                        edgecolor="none",
                        lw=0.2,
                    )
                axes.add_patch(p)
                legend[-1][0].append(p)

        if len(label) > 1:
            axes.legend(
                [tuple(k[0]) for k in legend],
                [k[1] for k in legend],
                loc="lower left",
                fancybox=True,
                framealpha=0.7,
            )

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
