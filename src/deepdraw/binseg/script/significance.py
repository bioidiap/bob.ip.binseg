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

import click

from clapp.click import ConfigCommand, ResourceOption, verbosity_option
from clapp.logging import setup

logger = setup(__name__.split(".")[0], format="%(levelname)s: %(message)s")


@click.command(
    entry_point_group="binseg.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Runs a significance test using as base the calculated predictions of two
       different systems, on the **same** dataset:
\b
       $ binseg significance -vv drive --names system1 system2 --predictions=path/to/predictions/system-1 path/to/predictions/system-2
\b
    2. By default, we use a "validation" dataset if it is available, to infer
       the a priori threshold for the comparison of two systems.  Otherwise,
       you may need to specify the name of a set to be used as validation set
       for choosing a threshold.  The same goes for the set to be used for
       testing the hypothesis - by default we use the "test" dataset if it is
       available, otherwise, specify.
\b
       $ binseg significance -vv drive --names system1 system2 --predictions=path/to/predictions/system-1 path/to/predictions/system-2 --threshold=train --evaluate=alternate-test
""",
)
@click.option(
    "--names",
    "-n",
    help="Names of the two systems to compare",
    nargs=2,
    required=True,
    type=str,
    cls=ResourceOption,
)
@click.option(
    "--predictions",
    "-p",
    help="Path where predictions of system 2 are currently stored.  You may "
    "also input predictions from a second-annotator.  This application "
    "will adequately handle it.",
    nargs=2,
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    cls=ResourceOption,
)
@click.option(
    "--dataset",
    "-d",
    help="A dictionary mapping string keys to "
    "torch.utils.data.dataset.Dataset instances",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--threshold",
    "-t",
    help="This number is used to define positives and negatives from "
    "probability maps, and report F1-scores (a priori). By default, we "
    "expect a set named 'validation' to be available at the input data. "
    "If that is not the case, we use 'train', if available.  You may provide "
    "the name of another dataset to be used for threshold tunning otherwise. "
    "If not set, or a string is input, threshold tunning is done per system, "
    "individually.  Optionally, you may also provide a floating-point number "
    "between [0.0, 1.0] as the threshold to use for both systems.",
    default="validation",
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--evaluate",
    "-e",
    help="Name of the dataset to evaluate",
    default="test",
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--steps",
    "-S",
    help="This number is used to define the number of threshold steps to "
    "consider when evaluating the highest possible F1-score on train/test data.",
    default=1000,
    type=int,
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--size",
    "-s",
    help="This is a tuple with two values indicating the size of windows to "
    "be used for sliding window analysis.  The values represent height and "
    "width respectively.",
    default=(128, 128),
    nargs=2,
    type=int,
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--stride",
    "-t",
    help="This is a tuple with two values indicating the stride of windows to "
    "be used for sliding window analysis.  The values represent height and "
    "width respectively.",
    default=(32, 32),
    nargs=2,
    type=int,
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--figure",
    "-f",
    help="The name of a performance figure (e.g. f1_score, or jaccard) to "
    "use when comparing performances",
    default="accuracy",
    type=str,
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--output-folder",
    "-o",
    help="Path where to store visualizations",
    required=False,
    type=click.Path(),
    show_default=True,
    cls=ResourceOption,
)
@click.option(
    "--remove-outliers/--no-remove-outliers",
    "-R",
    help="If set, removes outliers from both score distributions before "
    "running statistical analysis.  Outlier removal follows a 1.5 IQR range "
    "check from the difference in figures between both systems and assumes "
    "most of the distribution is contained within that range (like in a "
    "normal distribution)",
    default=False,
    required=True,
    show_default=True,
    cls=ResourceOption,
)
@click.option(
    "--remove-zeros/--no-remove-zeros",
    "-R",
    help="If set, removes instances from the statistical analysis in which "
    "both systems had a performance equal to zero.",
    default=False,
    required=True,
    show_default=True,
    cls=ResourceOption,
)
@click.option(
    "--parallel",
    "-x",
    help="Set the number of parallel processes to use when running using "
    "multiprocessing.  A value of zero uses all reported cores.",
    default=1,
    type=int,
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--checkpoint-folder",
    "-k",
    help="Path where to store checkpointed versions of sliding window "
    "performances",
    required=False,
    type=click.Path(),
    show_default=True,
    cls=ResourceOption,
)
@verbosity_option(logger=logger, cls=ResourceOption)
@click.pass_context
def significance(
    ctx,
    names,
    predictions,
    dataset,
    threshold,
    evaluate,
    steps,
    size,
    stride,
    figure,
    output_folder,
    remove_outliers,
    remove_zeros,
    parallel,
    checkpoint_folder,
    verbose,
    **kwargs,
):
    """Evaluates how significantly different are two models on the same
    dataset.

    This application calculates the significance of results of two
    models operating on the same dataset, and subject to a priori
    threshold tunning.
    """
    from ...common.script.significance import base_significance

    ctx.invoke(
        base_significance,
        names=names,
        predictions=predictions,
        dataset=dataset,
        threshold=threshold,
        evaluate=evaluate,
        steps=steps,
        size=size,
        stride=stride,
        figure=figure,
        output_folder=output_folder,
        remove_outliers=remove_outliers,
        remove_zeros=remove_zeros,
        parallel=parallel,
        checkpoint_folder=checkpoint_folder,
        verbose=verbose,
    )
