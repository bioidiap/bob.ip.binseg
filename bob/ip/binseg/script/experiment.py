#!/usr/bin/env python
# coding=utf-8

import os
import shutil

import click

from bob.extension.scripts.click_helper import (
    verbosity_option,
    ConfigCommand,
    ResourceOption,
)

from .binseg import save_sh_command

import logging

logger = logging.getLogger(__name__)


@click.command(
    entry_point_group="bob.ip.binseg.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Trains an M2U-Net model (VGG-16 backbone) with DRIVE (vessel
       segmentation), on the CPU, for only two epochs, then runs inference and
       evaluation on stock datasets, report performance as a table and a figure:

       $ bob binseg experiment -vv m2unet drive --epochs=2

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
    "bob.ip.binseg.data.utils.SampleList2TorchDataset's.  At least one key "
    "named 'train' must be available.  This dataset will be used for training "
    "the network model.  All other datasets will be used for prediction and "
    "evaluation. Dataset descriptions include all required pre-processing, "
    "including eventual data augmentation, which may be eventually excluded "
    "for prediction and evaluation purposes",
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
    "--drop-incomplete--batch is set, in which case this batch is not used.",
    required=True,
    show_default=True,
    default=2,
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
    help="Number of epochs (complete training set passes) to train for",
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
    "--ssl/--no-ssl",
    help="Switch ON/OFF semi-supervised training mode",
    show_default=True,
    required=True,
    default=False,
    cls=ResourceOption,
)
@click.option(
    "--rampup",
    "-r",
    help="Ramp-up length in epochs (for SSL training only)",
    show_default=True,
    required=True,
    default=900,
    type=click.IntRange(min=0),
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
@verbosity_option(cls=ResourceOption)
@click.pass_context
def experiment(
    ctx,
    model,
    optimizer,
    scheduler,
    output_folder,
    epochs,
    batch_size,
    drop_incomplete_batch,
    criterion,
    dataset,
    second_annotator,
    checkpoint_period,
    device,
    seed,
    ssl,
    rampup,
    overlayed,
    steps,
    verbose,
    **kwargs,
):
    """Runs a complete experiment, from training, to prediction and evaluation

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
    provide separate subsets for training and evaluation.  Instead of using
    simple datasets, datasets for full experiment running should be
    dictionaries with specific subset names:

    * ``__train__``: dataset used for training, prioritarily.  It is typically
      the dataset containing data augmentation pipelines.
    * ``__valid__``: dataset used for validation.  It is typically disjoint
      from the training and test sets.  In such a case, we checkpoint the model
      with the lowest loss on the validation set as well, throughout all the
      training, besides the model at the end of training.
    * ``train`` (optional): a copy of the ``__train__`` dataset, without data
      augmentation, that will be evaluated alongside other sets available
    * ``*``: any other name, not starting with an underscore character (``_``),
      will be considered a test set for evaluation.

    N.B.2: The threshold used for calculating the F1-score on the test set, or
    overlay analysis (false positives, negatives and true positives overprinted
    on the original image) also follows the logic above.
    """

    command_sh = os.path.join(output_folder, "command.sh")
    if os.path.exists(command_sh):
        backup = command_sh + '~'
        if os.path.exists(backup):
            os.unlink(backup)
        shutil.move(command_sh, backup)
    save_sh_command(command_sh)

    ## Training
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
        drop_incomplete_batch=drop_incomplete_batch,
        criterion=criterion,
        dataset=dataset,
        checkpoint_period=checkpoint_period,
        device=device,
        seed=seed,
        ssl=ssl,
        rampup=rampup,
        verbose=verbose,
    )
    logger.info("Ended training")

    from .analyze import analyze

    # preferably, we use the best model on the validation set
    # otherwise, we get the last saved model
    model_file = os.path.join(train_output_folder, "model_lowest_valid_loss.pth")
    if not os.path.exists(model_file):
        model_file = os.path.join(train_output_folder, "model_final.pth")

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
            verbose=verbose,
            )
