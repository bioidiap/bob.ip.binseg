# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import click

from clapper.click import AliasedGroup

from . import (
    analyze,
    compare,
    config,
    dataset,
    evaluate,
    experiment,
    mkmask,
    predict,
    significance,
    train,
    train_analysis,
)


@click.group(
    cls=AliasedGroup,
    context_settings=dict(help_option_names=["-?", "-h", "--help"]),
)
def cli():
    """Binary Segmentation Benchmark."""
    pass


cli.add_command(analyze.analyze)
cli.add_command(compare.compare)
cli.add_command(config.config)
cli.add_command(dataset.dataset)
cli.add_command(evaluate.evaluate)
cli.add_command(experiment.experiment)
cli.add_command(mkmask.mkmask)
cli.add_command(predict.predict)
cli.add_command(significance.significance)
cli.add_command(train.train)
cli.add_command(train_analysis.train_analysis)
