# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import contextlib
import logging

from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt
import numpy

matplotlib.use("agg")
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _precision_recall_canvas(title=None, limits=[0.0, 1.0, 0.0, 1.0]):
    """Generates a canvas to draw precision-recall curves.

    Works like a context manager, yielding a figure and an axes set in which
    the precision-recall curves should be added to.  The figure already
    contains F1-ISO lines and is preset to a 0-1 square region.  Once the
    context is finished, ``fig.tight_layout()`` is called.


    Parameters
    ----------

    title : :py:class:`str`, Optional
        Optional title to add to this plot

    limits : :py:class:`tuple`, Optional
        If set, a 4-tuple containing the bounds of the plot for the x and y
        axis respectively (format: ``[x_low, x_high, y_low, y_high]``).  If not
        set, use normal bounds (``[0, 1, 0, 1]``).


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
    axes1.set_xlim(limits[:2])
    axes1.set_ylim(limits[2:])

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
    axes1.spines["left"].set_position(
        ("data", limits[0] - (0.015 * (limits[1] - limits[0])))
    )
    axes1.spines["bottom"].set_position(
        ("data", limits[2] - (0.015 * (limits[3] - limits[2])))
    )

    # we shouldn't see any of axes 2 axes
    axes2.spines["right"].set_visible(False)
    axes2.spines["top"].set_visible(False)
    axes2.spines["left"].set_visible(False)
    axes2.spines["bottom"].set_visible(False)

    # yield execution, lets user draw precision-recall plots, and the legend
    # before tighteneing the layout
    yield fig, axes1

    plt.tight_layout()


def precision_recall_f1iso(data, limits):
    """Creates a precision-recall plot.

    This function creates and returns a Matplotlib figure with a
    precision-recall plot.  The plot will be annotated with F1-score iso-lines
    (in which the F1-score maintains the same value).

    This function specially supports "second-annotator" entries by plotting a
    line showing the comparison between the default annotator being analyzed
    and a second "opinion".  Second annotator dataframes contain a single entry
    (threshold=0.5), given the nature of the binary map comparisons.


    Parameters
    ----------

    data : dict
        A dictionary in which keys are strings defining plot labels and values
        are dictionaries with two entries:

        * ``df``: :py:class:`pandas.DataFrame`

          A dataframe that is produced by our evaluator engine, indexed by
          integer "thresholds", containing the following columns:
          ``threshold``, ``tp``, ``fp``, ``tn``, ``fn``, ``mean_precision``,
          ``mode_precision``, ``lower_precision``, ``upper_precision``,
          ``mean_recall``, ``mode_recall``, ``lower_recall``, ``upper_recall``,
          ``mean_specificity``, ``mode_specificity``, ``lower_specificity``,
          ``upper_specificity``, ``mean_accuracy``, ``mode_accuracy``,
          ``lower_accuracy``, ``upper_accuracy``, ``mean_jaccard``,
          ``mode_jaccard``, ``lower_jaccard``, ``upper_jaccard``,
          ``mean_f1_score``, ``mode_f1_score``, ``lower_f1_score``,
          ``upper_f1_score``, ``frequentist_precision``,
          ``frequentist_recall``, ``frequentist_specificity``,
          ``frequentist_accuracy``, ``frequentist_jaccard``,
          ``frequentist_f1_score``.

        * ``threshold``: :py:class:`list`

          A threshold to graph with a dot for each set.    Specific
          threshold values do not affect "second-annotator" dataframes.

    limits : tuple
        A 4-tuple containing the bounds of the plot for the x and y axis
        respectively (format: ``[x_low, x_high, y_low, y_high]``).  If not set,
        use normal bounds (``[0, 1, 0, 1]``).


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

    with _precision_recall_canvas(title=None, limits=limits) as (fig, axes):
        legend = []

        for name, value in data.items():
            df = value["df"]
            threshold = value["threshold"]

            # plots only from the point where recall reaches its maximum,
            # otherwise, we don't see a curve...
            max_recall = df.mean_recall.idxmax()
            pi = df.mean_precision[max_recall:]
            ri = df.mean_recall[max_recall:]
            # valid = (pi + ri) > 0

            # optimal point along the curve
            bins = len(df)
            index = int(round(bins * threshold))
            index = min(index, len(df) - 1)  # avoids out of range indexing

            # plots Recall/Precision as threshold changes
            label = f"{name} (F1={df.mean_f1_score[index]:.4f})"
            color = next(colorcycler)

            if len(df) == 1:
                # plot black dot for F1-score at select threshold
                (marker,) = axes.plot(
                    df.mean_recall[index],
                    df.mean_precision[index],
                    marker="*",
                    markersize=6,
                    color=color,
                    alpha=0.8,
                    linestyle="None",
                )
                (line,) = axes.plot(
                    df.mean_recall[index],
                    df.mean_precision[index],
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
                    df.mean_recall[index],
                    df.mean_precision[index],
                    marker="o",
                    linestyle=style,
                    markersize=4,
                    color=color,
                    alpha=0.8,
                )
                legend.append(([marker, line], label))

        if limits:
            axes.set_xlim(limits[:2])
            axes.set_ylim(limits[2:])

        if len(legend) > 1:
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
    ax2 = df["learning-rate"].plot(
        secondary_y=True,
        legend=True,
        grid=True,
    )
    ax2.set_ylabel("Learning Rate")
    ax1.set_xlabel("Epoch")
    plt.tight_layout()
    fig = ax1.get_figure()
    return fig
