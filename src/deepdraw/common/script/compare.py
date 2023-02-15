#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
# SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
# SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
# SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
# SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import os

import click
import pandas

from tqdm import tqdm

logger = logging.getLogger(__name__)


def base_compare(
    label_path,
    output_figure,
    output_table,
    threshold,
    plot_limits,
    detection,
    verbose,
    table_format="rst",
    **kwargs,
):
    """Compare multiple systems together."""

    def _validate_threshold(t, dataset):
        """Validate the user threshold selection.

        Returns parsed threshold.
        """
        if t is None:
            return t

        try:
            # we try to convert it to float first
            t = float(t)
            if t < 0.0 or t > 1.0:
                raise ValueError(
                    "Float thresholds must be within range [0.0, 1.0]"
                )
        except ValueError:
            # it is a bit of text - assert dataset with name is available
            if not isinstance(dataset, dict):
                raise ValueError(
                    "Threshold should be a floating-point number "
                    "if your provide only a single dataset for evaluation"
                )
            if t not in dataset:
                raise ValueError(
                    f"Text thresholds should match dataset names, "
                    f"but {t} is not available among the datasets provided ("
                    f"({', '.join(dataset.keys())})"
                )

        return t

    def _load(data, detection, threshold=None):
        """Plot comparison chart of all evaluated models.

        Parameters
        ----------

        data : dict
            A dict in which keys are the names of the systems and the values are
            paths to ``measures.csv`` style files.

        threshold : :py:class:`float`, :py:class:`str`, Optional
            A value indicating which threshold to choose for selecting a score.
            If set to ``None``, then use the maximum F1-score on that measures file.
            If set to a floating-point value, then use the score that is
            obtained on that particular threshold.  If set to a string, it should
            match one of the keys in ``data``.  It then first calculate the
            threshold reaching the maximum score on that particular dataset and
            then applies that threshold to all other sets. Obs: If the task
            is segmentation, the score used is the F1-Score; for the detection
            task the score used is the Intersection Over Union (IoU).


        Returns
        -------

        data : dict
            A dict in which keys are the names of the systems and the values are
            dictionaries that contain two keys:

            * ``df``: A :py:class:`pandas.DataFrame` with the measures data loaded
              to
            * ``threshold``: A threshold to be used for summarization, depending on
              the ``threshold`` parameter set on the input
        """
        if detection:
            col_name = "mean_iou"
            score_name = "IoU-score"

        else:
            col_name = "mean_f1_score"
            score_name = "F1-score"

        if isinstance(threshold, str):
            logger.info(
                f"Calculating threshold from maximum {score_name} at "
                f"'{threshold}' dataset..."
            )
            measures_path = data[threshold]
            df = pandas.read_csv(measures_path)
            use_threshold = df.threshold[df[col_name].idxmax()]
            logger.info(f"Dataset '*': threshold = {use_threshold:.3f}'")

        elif isinstance(threshold, float):
            use_threshold = threshold
            logger.info(f"Dataset '*': threshold = {use_threshold:.3f}'")

        # loads all data
        retval = {}
        for name, measures_path in tqdm(data.items(), desc="sample"):

            logger.info(f"Loading measures from {measures_path}...")
            df = pandas.read_csv(measures_path)

            if threshold is None:

                if "threshold_a_priori" in df:
                    use_threshold = df.threshold[df.threshold_a_priori.idxmax()]
                    logger.info(
                        f"Dataset '{name}': threshold (a priori) = "
                        f"{use_threshold:.3f}'"
                    )
                else:
                    use_threshold = df.threshold[df[col_name].idxmax()]
                    logger.info(
                        f"Dataset '{name}': threshold (a posteriori) = "
                        f"{use_threshold:.3f}'"
                    )

            retval[name] = dict(df=df, threshold=use_threshold)

        return retval

    # hack to get a dictionary from arguments passed to input
    if len(label_path) % 2 != 0:
        raise click.ClickException(
            "Input label-paths should be doubles"
            " composed of name-path entries"
        )
    data = dict(zip(label_path[::2], label_path[1::2]))

    threshold = _validate_threshold(threshold, data)

    # load all data measures
    data = _load(data, detection=detection, threshold=threshold)

    if detection:
        from ..utils.table import (
            performance_table_detection as performance_table,
        )

    else:
        from ..utils.plot import precision_recall_f1iso
        from ..utils.table import performance_table

        if output_figure is not None:
            output_figure = os.path.realpath(output_figure)
            logger.info(f"Creating and saving plot at {output_figure}...")
            os.makedirs(os.path.dirname(output_figure), exist_ok=True)
            fig = precision_recall_f1iso(data, limits=plot_limits)
            fig.savefig(output_figure)
            fig.clear()

    logger.info("Tabulating performance summary...")
    table = performance_table(data, table_format)
    click.echo(table)
    if output_table is not None:
        output_table = os.path.realpath(output_table)
        logger.info(f"Saving table at {output_table}...")
        os.makedirs(os.path.dirname(output_table), exist_ok=True)
        with open(output_table, "w") as f:
            f.write(table)
