# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import shutil

import click

from clapper.click import ConfigCommand, ResourceOption, verbosity_option
from clapper.logging import setup

logger = setup(__name__.split(".")[0], format="%(levelname)s: %(message)s")

from .common import save_sh_command


@click.command(
    entry_point_group="deepdraw.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
  1. Trains an M2U-Net model (VGG-16 backbone) with DRIVE (vessel
     segmentation), on the CPU, for only two epochs, then runs inference and
     evaluation on stock datasets, report performance as a table and a figure:

     .. code:: sh

        $ deepdraw experiment -vv m2unet drive --epochs=2
""",
)
@click.option(
    "--output-folder",
    "-o",
    help="Path where to store experiment outputs (created if does not exist)",
    required=True,
    type=click.Path(),
    default="results",
    cls=ResourceOption,
)
@click.option(
    "--model",
    "-m",
    help="A torch.nn.Module instance implementing the network to be trained, and then evaluated",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--dataset",
    "-d",
    help="A dictionary mapping string keys to "
    "torch.utils.data.dataset.Dataset instances implementing datasets "
    "to be used for training and validating the model, possibly including all "
    "pre-processing pipelines required or, optionally, a dictionary mapping "
    "string keys to torch.utils.data.dataset.Dataset instances.  At least "
    "one key named ``train`` must be available.  This dataset will be used for "
    "training the network model.  The dataset description must include all "
    "required pre-processing, including eventual data augmentation.  If a "
    "dataset named ``__train__`` is available, it is used prioritarily for "
    "training instead of ``train``.  If a dataset named ``__valid__`` is "
    "available, it is used for model validation (and automatic "
    "check-pointing) at each epoch.  If a dataset list named "
    "``__valid_extra__`` is available, then it will be tracked during the "
    "validation process and its loss output at the training log as well, "
    "in the format of an array occupying a single column.  All other keys "
    "are considered test datasets and only used during analysis, to report "
    "the final system performance",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--second-annotator",
    "-S",
    help="A dataset or dictionary, like in --dataset, with the same "
    "sample keys, but with annotations from a different annotator that is "
    "going to be compared to the one in --dataset",
    required=False,
    default=None,
    cls=ResourceOption,
    show_default=True,
)
@click.option(
    "--optimizer",
    help="A torch.optim.Optimizer that will be used to train the network",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--criterion",
    help="A loss function to compute the FCN error for every sample "
    "respecting the PyTorch API for loss functions (see torch.nn.modules.loss)",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--scheduler",
    help="A learning rate scheduler that drives changes in the learning "
    "rate depending on the FCN state (see torch.optim.lr_scheduler)",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--batch-size",
    "-b",
    help="Number of samples in every batch (this parameter affects "
    "memory requirements for the network).  If the number of samples in "
    "the batch is larger than the total number of samples available for "
    "training, this value is truncated.  If this number is smaller, then "
    "batches of the specified size are created and fed to the network "
    "until there are no more new samples to feed (epoch is finished).  "
    "If the total number of training samples is not a multiple of the "
    "batch-size, the last batch will be smaller than the first, unless "
    "--drop-incomplete-batch is set, in which case this batch is not used.",
    required=True,
    show_default=True,
    default=2,
    type=click.IntRange(min=1),
    cls=ResourceOption,
)
@click.option(
    "--batch-chunk-count",
    "-c",
    help="Number of chunks in every batch (this parameter affects "
    "memory requirements for the network). The number of samples "
    "loaded for every iteration will be batch-size/batch-chunk-count. "
    "batch-size needs to be divisible by batch-chunk-count, otherwise an "
    "error will be raised. This parameter is used to reduce number of "
    "samples loaded in each iteration, in order to reduce the memory usage "
    "in exchange for processing time (more iterations).  This is specially "
    "interesting whe one is running with GPUs with limited RAM. The "
    "default of 1 forces the whole batch to be processed at once.  Otherwise "
    "the batch is broken into batch-chunk-count pieces, and gradients are "
    "accumulated to complete each batch.",
    required=True,
    show_default=True,
    default=1,
    type=click.IntRange(min=1),
    cls=ResourceOption,
)
@click.option(
    "--drop-incomplete-batch/--no-drop-incomplete-batch",
    "-D",
    help="If set, then may drop the last batch in an epoch, in case it is "
    "incomplete.  If you set this option, you should also consider "
    "increasing the total number of epochs of training, as the total number "
    "of training steps may be reduced",
    required=True,
    show_default=True,
    default=False,
    cls=ResourceOption,
)
@click.option(
    "--epochs",
    "-e",
    help="Number of epochs (complete training set passes) to train for. "
    "If continuing from a saved checkpoint, ensure to provide a greater "
    "number of epochs than that saved on the checkpoint to be loaded. ",
    show_default=True,
    required=True,
    default=1000,
    type=click.IntRange(min=1),
    cls=ResourceOption,
)
@click.option(
    "--checkpoint-period",
    "-p",
    help="Number of epochs after which a checkpoint is saved. "
    "A value of zero will disable check-pointing. If checkpointing is "
    "enabled and training stops, it is automatically resumed from the "
    "last saved checkpoint if training is restarted with the same "
    "configuration.",
    show_default=True,
    required=True,
    default=0,
    type=click.IntRange(min=0),
    cls=ResourceOption,
)
@click.option(
    "--device",
    "-d",
    help='A string indicating the device to use (e.g. "cpu" or "cuda:0")',
    show_default=True,
    required=True,
    default="cpu",
    cls=ResourceOption,
)
@click.option(
    "--seed",
    "-s",
    help="Seed to use for the random number generator",
    show_default=True,
    required=False,
    default=42,
    type=click.IntRange(min=0),
    cls=ResourceOption,
)
@click.option(
    "--parallel",
    "-P",
    help="""Use multiprocessing for data loading and processing: if set to -1
    (default), disables multiprocessing altogether.  Set to 0 to enable as many
    data loading instances as processing cores as available in the system.  Set
    to >= 1 to enable that many multiprocessing instances for data
    processing.""",
    type=click.IntRange(min=-1),
    show_default=True,
    required=True,
    default=-1,
    cls=ResourceOption,
)
@click.option(
    "--monitoring-interval",
    "-I",
    help="""Time between checks for the use of resources during each training
    epoch.  An interval of 5 seconds, for example, will lead to CPU and GPU
    resources being probed every 5 seconds during each training epoch.
    Values registered in the training logs correspond to averages (or maxima)
    observed through possibly many probes in each epoch.  Notice that setting a
    very small value may cause the probing process to become extremely busy,
    potentially biasing the overall perception of resource usage.""",
    type=click.FloatRange(min=0.1),
    show_default=True,
    required=True,
    default=5.0,
    cls=ResourceOption,
)
@click.option(
    "--overlayed/--no-overlayed",
    "-O",
    help="Creates overlayed representations of the output probability maps, "
    "similar to --overlayed in prediction-mode, except it includes "
    "distinctive colours for true and false positives and false negatives.  "
    "If not set, or empty then do **NOT** output overlayed images.",
    show_default=True,
    default=False,
    required=False,
    cls=ResourceOption,
)
@click.option(
    "--steps",
    "-S",
    help="This number is used to define the number of threshold steps to "
    "consider when evaluating the highest possible F1-score on test data.",
    default=1000,
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--plot-limits",
    "-L",
    help="""If set, this option affects the performance comparison plots.  It
    must be a 4-tuple containing the bounds of the plot for the x and y axis
    respectively (format: x_low, x_high, y_low, y_high]).  If not set, use
    normal bounds ([0, 1, 0, 1]) for the performance curve.""",
    default=[0.0, 1.0, 0.0, 1.0],
    show_default=True,
    nargs=4,
    type=float,
    cls=ResourceOption,
)
@verbosity_option(logger=logger, cls=ResourceOption)
@click.pass_context
def experiment(
    ctx,
    model,
    optimizer,
    scheduler,
    output_folder,
    epochs,
    batch_size,
    batch_chunk_count,
    drop_incomplete_batch,
    criterion,
    dataset,
    second_annotator,
    checkpoint_period,
    device,
    seed,
    parallel,
    monitoring_interval,
    overlayed,
    steps,
    plot_limits,
    verbose,
    **kwargs,
):
    """Runs a complete experiment, from training, to prediction and evaluation.

    This script is just a wrapper around the individual scripts for training,
    running prediction, evaluating and comparing FCN model performance.  It
    organises the output in a preset way::

        \b
       └─ <output-folder>/
          ├── model/  #the generated model will be here
          ├── predictions/  #the prediction outputs for the train/test set
          ├── overlayed/  #the overlayed outputs for the train/test set
             ├── predictions/  #predictions overlayed on the input images
             ├── analysis/  #predictions overlayed on the input images
             ├              #including analysis of false positives, negatives
             ├              #and true positives
             └── second-annotator/  #if set, store overlayed images for the
                                    #second annotator here
          └── analysis /  #the outputs of the analysis of both train/test sets
                          #includes second-annotator "mesures" as well, if
                          # configured

    Training is performed for a configurable number of epochs, and generates at
    least a final_model.pth.  It may also generate a number of intermediate
    checkpoints.  Checkpoints are model files (.pth files) that are stored
    during the training and useful to resume the procedure in case it stops
    abruptly.

    N.B.: The tool is designed to prevent analysis bias and allows one to
    provide (potentially multiple) separate subsets for training,
    validation, and evaluation.  Instead of using simple datasets, datasets
    for full experiment running should be dictionaries with specific subset
    names:

    * ``__train__``: dataset used for training, prioritarily.  It is typically
      the dataset containing data augmentation pipelines.
    * ``__valid__``: dataset used for validation.  It is typically disjoint
      from the training and test sets.  In such a case, we checkpoint the model
      with the lowest loss on the validation set as well, throughout all the
      training, besides the model at the end of training.
    * ``train`` (optional): a copy of the ``__train__`` dataset, without data
      augmentation, that will be evaluated alongside other sets available
    * ``__valid_extra__``: a list of datasets that are tracked during
      validation, but do not affect checkpoiting. If present, an extra
      column with an array containing the loss of each set is kept on the
      training log.
    * ``*``: any other name, not starting with an underscore character (``_``),
      will be considered a test set for evaluation.

    N.B.2: The threshold used for calculating the F1-score on the test set, or
    overlay analysis (false positives, negatives and true positives overprinted
    on the original image) also follows the logic above.
    """

    command_sh = os.path.join(output_folder, "command.sh")
    if os.path.exists(command_sh):
        backup = command_sh + "~"
        if os.path.exists(backup):
            os.unlink(backup)
        shutil.move(command_sh, backup)
    save_sh_command(command_sh)

    # training
    logger.info("Started training")

    from .train import train

    train_output_folder = os.path.join(output_folder, "model")
    ctx.invoke(
        train,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_folder=train_output_folder,
        epochs=epochs,
        batch_size=batch_size,
        batch_chunk_count=batch_chunk_count,
        drop_incomplete_batch=drop_incomplete_batch,
        criterion=criterion,
        dataset=dataset,
        checkpoint_period=checkpoint_period,
        device=device,
        seed=seed,
        parallel=parallel,
        monitoring_interval=monitoring_interval,
        verbose=verbose,
    )
    logger.info("Ended training")

    from .train_analysis import train_analysis

    ctx.invoke(
        train_analysis,
        log=os.path.join(train_output_folder, "trainlog.csv"),
        constants=os.path.join(train_output_folder, "constants.csv"),
        output_pdf=os.path.join(train_output_folder, "trainlog.pdf"),
        verbose=verbose,
    )

    from .analyze import analyze

    # preferably, we use the best model on the validation set
    # otherwise, we get the last saved model
    model_file = os.path.join(
        train_output_folder, "model_lowest_valid_loss.pth"
    )
    if not os.path.exists(model_file):
        model_file = os.path.join(train_output_folder, "model_final_epoch.pth")

    ctx.invoke(
        analyze,
        model=model,
        output_folder=output_folder,
        batch_size=batch_size,
        dataset=dataset,
        second_annotator=second_annotator,
        device=device,
        overlayed=overlayed,
        weight=model_file,
        steps=steps,
        parallel=parallel,
        plot_limits=plot_limits,
        verbose=verbose,
    )
