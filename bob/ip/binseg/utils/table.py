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
          integer "thresholds", containing the following columns: ``threshold``
          (sorted ascending), ``precision``, ``recall``, ``pr_upper`` (upper
          precision bounds), ``pr_lower`` (lower precision bounds),
          ``re_upper`` (upper recall bounds), ``re_lower`` (lower recall
          bounds).

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
        "F1",
        "F1\nstd",
        "P",
        "R",
        "F1\nmax",
        "P\nmax",
        "R\nmax",
        "AUC",
        ]

    table = []
    for k, v in data.items():
        entry = [k, v["threshold"], ]

        # statistics based on the "assigned" threshold (a priori, less biased)
        bins = len(v["df"])
        index = int(round(bins*v["threshold"]))
        index = min(index, len(v["df"])-1)  #avoids out of range indexing
        entry.append(v["df"].f1_score[index])
        entry.append(v["df"].std_f1[index])
        entry.append(v["df"].precision[index])
        entry.append(v["df"].recall[index])

        # statistics based on the best threshold (a posteriori, biased)
        entry.append(v["df"].f1_score.max())
        f1max_idx = v["df"].f1_score.idxmax()
        entry.append(v["df"].precision[f1max_idx])
        entry.append(v["df"].recall[f1max_idx])
        entry.append(auc(v["df"]["recall"].to_numpy(),
            v["df"]["precision"].to_numpy()))

        table.append(entry)

    return tabulate.tabulate(table, headers, tablefmt=fmt, floatfmt=".3f")
