# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os

import click
import pandas
import tabulate

from clapper.click import verbosity_option
from clapper.logging import setup
from tqdm import tqdm

logger = setup(__name__.split(".")[0], format="%(levelname)s: %(message)s")


@click.command(
    epilog="""Examples:

\b
  1. Compares system A and B, with their own pre-computed measure files:

     .. code:: sh

        $ deepdraw compare -vv A path/to/A/train.csv B path/to/B/test.csv
""",
)
@click.argument(
    "label_path",
    nargs=-1,
)
@click.option(
    "--output-figure",
    "-f",
    help="Path where write the output figure (any extension supported by "
    "matplotlib is possible).  If not provided, does not produce a figure.",
    required=False,
    default=None,
    type=click.Path(dir_okay=False, file_okay=True),
)
@click.option(
    "--table-format",
    "-T",
    help="The format to use for the comparison table",
    show_default=True,
    required=True,
    default="rst",
    type=click.Choice(tabulate.tabulate_formats),
)
@click.option(
    "--output-table",
    "-u",
    help="Path where write the output table. If not provided, does not write "
    "write a table to file, only to stdout.",
    required=False,
    default=None,
    type=click.Path(dir_okay=False, file_okay=True),
)
@click.option(
    "--threshold",
    "-t",
    help="This number is used to select which F1-score to use for "
    "representing a system performance.  If not set, we report the maximum "
    "F1-score in the set, which is equivalent to threshold selection a "
    "posteriori (biased estimator), unless the performance file being "
    "considered already was pre-tunned, and contains a 'threshold_a_priori' "
    "column which we then use to pick a threshold for the dataset. "
    "You can override this behaviour by either setting this value to a "
    "floating-point number in the range [0.0, 1.0], or to a string, naming "
    "one of the systems which will be used to calculate the threshold "
    "leading to the maximum F1-score and then applied to all other sets.",
    default=None,
    show_default=False,
    required=False,
)
@click.option(
    "--plot-limits",
    "-L",
    help="""If set, must be a 4-tuple containing the bounds of the plot for
    the x and y axis respectively (format: x_low, x_high, y_low,
    y_high]).  If not set, use normal bounds ([0, 1, 0, 1]) for the
    performance curve.""",
    default=[0.0, 1.0, 0.0, 1.0],
    show_default=True,
    nargs=4,
    type=float,
)
@verbosity_option(
    logger=logger,
)
@click.pass_context
def compare(
    ctx,
    label_path,
    output_figure,
    table_format,
    output_table,
    threshold,
    plot_limits,
    verbose,
    **kwargs,
):
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

    def _load(data, threshold=None):
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
            is segmentation, the score used is the F1-Score.


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
    data = _load(data, threshold=threshold)

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
