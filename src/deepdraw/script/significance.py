# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys

import click
import numpy

from clapper.click import ConfigCommand, ResourceOption, verbosity_option
from clapper.logging import setup

logger = setup(__name__.split(".")[0], format="%(levelname)s: %(message)s")

from ..engine.evaluator import run as run_evaluation
from ..engine.significance import (
    PERFORMANCE_FIGURES,
    index_of_outliers,
    sliding_window_performances,
    visual_performances,
    write_analysis_figures,
    write_analysis_text,
)


@click.command(
    entry_point_group="deepdraw.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
  1. Runs a significance test using as base the calculated predictions of two
     different systems, on the **same** dataset:

     .. code:: sh

        $ deepdraw significance -vv drive --names system1 system2 --predictions=path/to/predictions/system-1 path/to/predictions/system-2

\b
  2. By default, we use a "validation" dataset if it is available, to infer
     the a priori threshold for the comparison of two systems.  Otherwise,
     you may need to specify the name of a set to be used as validation set
     for choosing a threshold.  The same goes for the set to be used for
     testing the hypothesis - by default we use the "test" dataset if it is
     available, otherwise, specify.

     .. code:: sh

        $ deepdraw significance -vv drive --names system1 system2 --predictions=path/to/predictions/system-1 path/to/predictions/system-2 --threshold=train --evaluate=alternate-test
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

    def _validate_threshold(t, dataset):
        """Validate the user threshold selection.

        Returns parsed threshold.
        """
        if t is None:
            return 0.5

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

    def _eval_sliding_windows(
        system_name,
        threshold,
        evaluate,
        preddir,
        dataset,
        steps,
        size,
        stride,
        outdir,
        figure,
        nproc,
        checkpointdir,
    ):
        """Calculates the sliding window performances on a dataset.

        Parameters
        ==========

        system_name : str
            The name of the current system being analyzed

        threshold : :py:class:`float`, :py:class:`str`
            This number is used to define positives and negatives from probability
            maps, and report F1-scores (a priori). By default, we expect a set
            named 'validation' to be available at the input data.  If that is not
            the case, we use 'train', if available.  You may provide the name of
            another dataset to be used for threshold tunning otherwise.  If not
            set, or a string is input, threshold tunning is done per system,
            individually.  Optionally, you may also provide a floating-point number
            between [0.0, 1.0] as the threshold to use for both systems.

        evaluate : str
            Name of the dataset key to use from ``dataset`` to evaluate (typically,
            ``test``)

        preddir : str
            Root path to the predictions generated by system ``system_name``.  The
            final subpath inside ``preddir`` that will be used will have the value
            of this variable suffixed with the value of ``evaluate``.  We will
            search for ``<preddir>/<evaluate>/<stems>.hdf5``.

        dataset : dict
            A dictionary mapping string keys to
            :py:class:`torch.utils.data.dataset.Dataset` instances

        steps : int
            The number of threshold steps to consider when evaluating the highest
            possible F1-score on train/test data.

        size : tuple
            Two values indicating the size of windows to be used for the sliding
            window analysis.  The values represent height and width respectively

        stride : tuple
            Two values indicating the stride of windows to be used for the sliding
            window analysis.  The values represent height and width respectively

        outdir : str
            Path where to store visualizations.  If set to ``None``, then do not
            store performance visualizations.

        figure : str
            The name of a performance figure (e.g. ``f1_score``, ``jaccard``, or
            ``accuracy``) to use when comparing performances

        nproc : int
            Sets the number of parallel processes to use when running using
            multiprocessing.  A value of zero uses all reported cores.  A value of
            ``1`` avoids completely the use of multiprocessing and runs all chores
            in the current processing context.

        checkpointdir : str
            If set to a string (instead of ``None``), then stores a cached version
            of the sliding window performances on disk, for a particular system.


        Returns
        =======

        d : dict
            A dictionary in which keys are filename stems and values are
            dictionaries with the following contents:

            ``winperf``: numpy.ndarray
                A dataframe with all the sliding window performances aggregated,
                for all input images.

            ``n`` : numpy.ndarray
                A 2D numpy array containing the number of performance scores for
                every pixel in the original image

            ``avg`` : numpy.ndarray
                A 2D numpy array containing the average performances for every
                pixel on the input image considering the sliding window sizes and
                strides applied to the image

            ``std`` : numpy.ndarray
                A 2D numpy array containing the (unbiased) standard deviations for
                the provided performance figure, for every pixel on the input image
                considering the sliding window sizes and strides applied to the
                image
        """

        if checkpointdir is not None:
            chkpt_fname = os.path.join(
                checkpointdir,
                f"{system_name}-{evaluate}-{threshold}-"
                f"{size[0]}x{size[1]}+{stride[0]}x{stride[1]}-{figure}.pkl.gz",
            )
            os.makedirs(os.path.dirname(chkpt_fname), exist_ok=True)
            if os.path.exists(chkpt_fname):
                logger.info(f"Loading checkpoint from {chkpt_fname}...")
                # loads and returns checkpoint from file
                try:
                    with __import__("gzip").GzipFile(chkpt_fname, "r") as f:
                        return __import__("pickle").load(f)
                except EOFError as e:
                    logger.warning(
                        f"Could not load sliding window performance "
                        f"from {chkpt_fname}: {e}. Calculating..."
                    )
            else:
                logger.debug(
                    f"Checkpoint not available at {chkpt_fname}. "
                    f"Calculating..."
                )
        else:
            chkpt_fname = None

        if not isinstance(threshold, float):
            assert threshold in dataset, f"No dataset named '{threshold}'"

            logger.info(
                f"Evaluating threshold on '{threshold}' set for "
                f"'{system_name}' using {steps} steps"
            )
            threshold = run_evaluation(
                dataset[threshold], threshold, preddir, steps=steps
            )
            logger.info(f"Set --threshold={threshold:.5f} for '{system_name}'")

        # for a given threshold on each system, calculate sliding window performances
        logger.info(
            f"Evaluating sliding window '{figure}' on '{evaluate}' set for "
            f"'{system_name}' using windows of size {size} and stride {stride}"
        )

        retval = sliding_window_performances(
            dataset,
            evaluate,
            preddir,
            threshold,
            size,
            stride,
            figure,
            nproc,
            outdir,
        )

        # cache sliding window performance for later use, if necessary
        if chkpt_fname is not None:
            logger.debug(f"Storing checkpoint at {chkpt_fname}...")
            with __import__("gzip").GzipFile(chkpt_fname, "w") as f:
                __import__("pickle").dump(retval, f)

        return retval

    def _eval_differences(
        names,
        perfs,
        evaluate,
        dataset,
        size,
        stride,
        outdir,
        figure,
        nproc,
        checkpointdir,
    ):
        """Evaluate differences in the performance sliding windows between two
        systems.

        Parameters
        ----------

        names : :py:class:`tuple` of :py:class:`str`
            Names of the first and second systems

        perfs : :py:class:`tuple` of :py:class:`dict`
            Dictionaries for the sliding window performances of each system, as
            returned by :py:func:`_eval_sliding_windows`

        evaluate : str
            Name of the dataset key to use from ``dataset`` to evaluate (typically,
            ``test``)

        dataset : dict
            A dictionary mapping string keys to
            :py:class:`torch.utils.data.dataset.Dataset` instances

        size : tuple
            Two values indicating the size of windows to be used for sliding window
            analysis.  The values represent height and width respectively

        stride : tuple
            Two values indicating the stride of windows to be used for sliding
            window analysis.  The values represent height and width respectively

        outdir : str
            If set to ``None``, then do not output performance visualizations.
            Otherwise, in directory ``outdir``, dumps the visualizations for the
            performance differences between both systems.

        figure : str
            The name of a performance figure (e.g. ``f1_score``, or ``jaccard``) to
            use when comparing performances

        nproc : int
            Sets the number of parallel processes to use when running using
            multiprocessing.  A value of zero uses all reported cores.  A value of
            ``1`` avoids completely the use of multiprocessing and runs all chores
            in the current processing context.

        checkpointdir : str
            If set to a string (instead of ``None``), then stores a cached version
            of the sliding window performances on disk, for a particular difference
            between systems.


        Returns
        -------

        d : dict
            A dictionary representing sliding window performance differences across
            all files and sliding windows.  The format of this is similar to the
            individual inputs ``perf1`` and ``perf2``.
        """

        if checkpointdir is not None:
            chkpt_fname = os.path.join(
                checkpointdir,
                f"{names[0]}-{names[1]}-{evaluate}-"
                f"{size[0]}x{size[1]}+{stride[0]}x{stride[1]}-{figure}.pkl.gz",
            )
            os.makedirs(os.path.dirname(chkpt_fname), exist_ok=True)
            if os.path.exists(chkpt_fname):
                logger.info(f"Loading checkpoint from {chkpt_fname}...")
                # loads and returns checkpoint from file
                try:
                    with __import__("gzip").GzipFile(chkpt_fname, "r") as f:
                        return __import__("pickle").load(f)
                except EOFError as e:
                    logger.warning(
                        f"Could not load sliding window performance "
                        f"from {chkpt_fname}: {e}. Calculating..."
                    )
            else:
                logger.debug(
                    f"Checkpoint not available at {chkpt_fname}. "
                    f"Calculating..."
                )
        else:
            chkpt_fname = None

        perf_diff = {
            k: perfs[0][k]["winperf"] - perfs[1][k]["winperf"] for k in perfs[0]
        }

        # for a given threshold on each system, calculate sliding window performances
        logger.info(
            f"Evaluating sliding window '{figure}' differences on '{evaluate}' "
            f"set on '{names[0]}-{names[1]}' using windows of size {size} and "
            f"stride {stride}"
        )

        retval = visual_performances(
            dataset,
            evaluate,
            perf_diff,
            size,
            stride,
            figure,
            nproc,
            outdir,
        )

        # cache sliding window performance for later use, if necessary
        if chkpt_fname is not None:
            logger.debug(f"Storing checkpoint at {chkpt_fname}...")
            with __import__("gzip").GzipFile(chkpt_fname, "w") as f:
                __import__("pickle").dump(retval, f)

        return retval

    # minimal validation to startup
    threshold = _validate_threshold(threshold, dataset)
    assert evaluate in dataset, f"No dataset named '{evaluate}'"

    perf1 = _eval_sliding_windows(
        names[0],
        threshold,
        evaluate,
        predictions[0],
        dataset,
        steps,
        size,
        stride,
        (
            output_folder
            if output_folder is None
            else os.path.join(output_folder, names[0])
        ),
        figure,
        parallel,
        checkpoint_folder,
    )

    perf2 = _eval_sliding_windows(
        names[1],
        threshold,
        evaluate,
        predictions[1],
        dataset,
        steps,
        size,
        stride,
        (
            output_folder
            if output_folder is None
            else os.path.join(output_folder, names[1])
        ),
        figure,
        parallel,
        checkpoint_folder,
    )

    # perf_diff = _eval_differences(
    #     names,
    #     (perf1, perf2),
    #     evaluate,
    #     dataset,
    #     size,
    #     stride,
    #     (
    #         output_folder
    #         if output_folder is None
    #         else os.path.join(output_folder, "diff")
    #     ),
    #     figure,
    #     parallel,
    #     checkpoint_folder,
    # )

    # loads all figures for the given threshold
    stems = list(perf1.keys())
    figindex = PERFORMANCE_FIGURES.index(figure)
    da = numpy.array([perf1[k]["winperf"][figindex] for k in stems]).flatten()
    db = numpy.array([perf2[k]["winperf"][figindex] for k in stems]).flatten()
    diff = da - db

    while remove_outliers:
        outliers_diff = index_of_outliers(diff)
        if sum(outliers_diff) == 0:
            break
        diff = diff[~outliers_diff]
        da = da[~outliers_diff]
        db = db[~outliers_diff]

    if remove_zeros:
        remove_zeros = (da == 0) & (db == 0)
        diff = diff[~remove_zeros]
        da = da[~remove_zeros]
        db = db[~remove_zeros]

    if output_folder is not None:
        fname = os.path.join(output_folder, "analysis.pdf")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        logger.info(f"Writing analysis figures to {fname} (multipage PDF)...")
        write_analysis_figures(names, da, db, fname)

    if output_folder is not None:
        fname = os.path.join(output_folder, "analysis.txt")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        logger.info(f"Writing analysis summary to {fname}...")
        with open(fname, "w") as f:
            write_analysis_text(names, da, db, f)
    write_analysis_text(names, da, db, sys.stdout)
