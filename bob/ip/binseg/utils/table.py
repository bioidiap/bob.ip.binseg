#!/usr/bin/env python
# coding=utf-8


import tabulate
from .measure import auc


def performance_table(data, fmt):
    """Tables result comparison in a given format


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


    fmt : str
        One of the formats supported by tabulate.


    Returns
    -------

    table : str
        A table in a specific format

    """

    headers = [
        "Dataset",
        "T",
        "E(F1)",
        "CI(F1)",
        "AUC",
        "CI(AUC)",
        ]

    table = []
    for k, v in data.items():
        entry = [k, v["threshold"], ]

        # statistics based on the "assigned" threshold (a priori, less biased)
        bins = len(v["df"])
        index = int(round(bins*v["threshold"]))
        index = min(index, len(v["df"])-1)  #avoids out of range indexing
        entry.append(v["df"].mean_f1_score[index])
        entry.append(f"{v['df'].lower_f1_score[index]:.3f}-{v['df'].upper_f1_score[index]:.3f}")

        # AUC PR curve
        entry.append(auc(v["df"]["mean_recall"].to_numpy(),
                v["df"]["mean_precision"].to_numpy()))
        lower_auc = auc(v["df"]["lower_recall"].to_numpy(),
                v["df"]["lower_precision"].to_numpy())
        upper_auc = auc(v["df"]["upper_recall"].to_numpy(),
                v["df"]["upper_precision"].to_numpy())
        entry.append(f"{lower_auc:.3f}-{upper_auc:.3f}")

        table.append(entry)

    return tabulate.tabulate(table, headers, tablefmt=fmt, floatfmt=".3f",
            stralign="right")
