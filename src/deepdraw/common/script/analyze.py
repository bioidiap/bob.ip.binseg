# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import os

import click

from .common import save_sh_command

logger = logging.getLogger(__name__)


@click.pass_context
def base_analyze(
    ctx,
    model,
    output_folder,
    batch_size,
    dataset,
    second_annotator,
    device,
    overlayed,
    weight,
    steps,
    parallel,
    plot_limits,
    verbose,
    detection,
    **kwargs,
):
    """Create base analyze function for segmentation / detection tasks."""
    command_sh = os.path.join(output_folder, "command.sh")
    if not os.path.exists(command_sh):
        # only save if experiment has not saved yet something similar
        save_sh_command(command_sh)

    # Prediction
    logger.info("Started prediction")

    from .predict import base_predict

    predictions_folder = os.path.join(output_folder, "predictions")
    overlayed_folder = (
        os.path.join(output_folder, "overlayed", "predictions")
        if overlayed
        else None
    )

    ctx.invoke(
        base_predict,
        output_folder=predictions_folder,
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
        weight=weight,
        overlayed=overlayed_folder,
        parallel=parallel,
        detection=detection,
        verbose=verbose,
    )
    logger.info("Ended prediction")

    # Evaluation
    logger.info("Started evaluation")

    from .evaluate import base_evaluate

    overlayed_folder = (
        os.path.join(output_folder, "overlayed", "analysis")
        if overlayed
        else None
    )

    # choosing the overlayed_threshold
    if "validation" in dataset:
        threshold = "validation"
    elif "train" in dataset:
        threshold = "train"
    else:
        threshold = 0.5
    logger.info(f"Setting --threshold={threshold}...")

    analysis_folder = os.path.join(output_folder, "analysis")
    ctx.invoke(
        base_evaluate,
        output_folder=analysis_folder,
        predictions_folder=predictions_folder,
        dataset=dataset,
        second_annotator=second_annotator,
        overlayed=overlayed_folder,
        threshold=threshold,
        steps=steps,
        parallel=parallel,
        detection=detection,
        verbose=verbose,
    )

    logger.info("Ended evaluation")

    # Comparison
    logger.info("Started comparison")

    # compare performances on the various sets
    from .compare import base_compare

    systems = []
    for k, v in dataset.items():
        if k.startswith("_"):
            logger.info(f"Skipping dataset '{k}' (not to be compared)")
            continue
        candidate = os.path.join(analysis_folder, f"{k}.csv")
        if not os.path.exists(candidate):
            logger.error(
                f"Skipping dataset '{k}' "
                f"(candidate CSV file `{candidate}` does not exist!)"
            )
            continue
        systems += [k, os.path.join(analysis_folder, f"{k}.csv")]
    if second_annotator is not None:
        for k, v in second_annotator.items():
            if k.startswith("_"):
                logger.info(
                    f"Skipping second-annotator '{k}' " f"(not to be compared)"
                )
                continue
            if k not in dataset:
                logger.info(
                    f"Skipping second-annotator '{k}' "
                    f"(no equivalent `dataset[{k}]`)"
                )
                continue
            if not dataset[k].all_keys_match(v):
                logger.warning(
                    f"Skipping second-annotator '{k}' "
                    f"(keys do not match `dataset[{k}]`?)"
                )
                continue
            candidate = os.path.join(
                analysis_folder, "second-annotator", f"{k}.csv"
            )
            if not os.path.exists(candidate):
                logger.error(
                    f"Skipping second-annotator '{k}' "
                    f"(candidate CSV file `{candidate}` does not exist!)"
                )
                continue
            systems += [f"{k} (2nd. annot.)", candidate]

    output_figure = os.path.join(output_folder, "comparison.pdf")
    output_table = os.path.join(output_folder, "comparison.rst")

    ctx.invoke(
        base_compare,
        label_path=systems,
        output_figure=output_figure,
        output_table=output_table,
        threshold=threshold,
        plot_limits=plot_limits,
        detection=detection,
        verbose=verbose,
    )

    logger.info("Ended comparison")
