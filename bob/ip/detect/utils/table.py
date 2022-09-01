#!/usr/bin/env python
# coding=utf-8


import tabulate


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
          ``threshold``, ``iou``.

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
        "E(IoU)",
    ]

    table = []
    for k, v in data.items():
        entry = [
            k,
            v["threshold"],
        ]

        bins = len(v["df"])
        index = int(round(bins * v["threshold"]))
        index = min(index, len(v["df"]) - 1)  # avoids out of range indexing
        entry.append(v["df"].mean_iou[index])

        table.append(entry)

    return tabulate.tabulate(
        table, headers, tablefmt=fmt, floatfmt=".3f", stralign="right"
    )
