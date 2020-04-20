#!/usr/bin/env python
# coding=utf-8

import os

import click

from bob.extension.scripts.click_helper import (
    verbosity_option,
    ConfigCommand,
    ResourceOption,
)

import logging

logger = logging.getLogger(__name__)


@click.command(
    entry_point_group="bob.ip.binseg.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Trains a M2U-Net model (VGG-16 backbone) with STARE (vessel segmentation),
       on the CPU, for only two epochs, then runs inference and evaluation on
       results from its test set:

       $ bob binseg experiment -vv m2unet stare --epochs=2

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
    "--train-dataset",
    "-d",
    help="A torch.utils.data.dataset.Dataset instance implementing a dataset "
    "to be used for training the model, possibly including all pre-processing"
    " pipelines required, including data augmentation",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--test-dataset",
    "-d",
    help="A torch.utils.data.dataset.Dataset instance implementing a dataset "
    "to be used for testing the model, possibly including all pre-processing"
    " pipelines required",
    required=True,
    cls=ResourceOption,
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
    "--pretrained-backbone",
    "-t",
    help="URL of a pre-trained model file that will be used to preset "
    "FCN weights (where relevant) before training starts "
    "(e.g. vgg16, mobilenetv2)",
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
@verbosity_option(cls=ResourceOption)
@click.pass_context
def experiment(
    ctx,
    model,
    optimizer,
    scheduler,
    output_folder,
    epochs,
    pretrained_backbone,
    batch_size,
    drop_incomplete_batch,
    criterion,
    train_dataset,
    test_dataset,
    checkpoint_period,
    device,
    seed,
    ssl,
    rampup,
    overlayed,
    verbose,
    **kwargs,
):
    """Runs a complete experiment, from training, prediction and evaluation

    This script is just a wrapper around the individual scripts for training,
    running prediction and evaluating FCN models.  It organises the output in a
    preset way:

    .. code-block:: text

       └─ <output-folder>/
          ├── model/  #the generated model will be here
          ├── predictions/  #the prediction outputs for the train/test set
          ├── overlayed/  #the overlayed outputs for the train/test set
          └── analysis /  #the outputs of the analysis of both train/test sets

    Training is performed for a configurable number of epochs, and generates at
    least a final_model.pth.  It may also generate a number of intermediate
    checkpoints.  Checkpoints are model files (.pth files) that are stored
    during the training and useful to resume the procedure in case it stops
    abruptly.

    """

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
        pretrained_backbone=pretrained_backbone,
        batch_size=batch_size,
        drop_incomplete_batch=drop_incomplete_batch,
        criterion=criterion,
        dataset=train_dataset,
        checkpoint_period=checkpoint_period,
        device=device,
        seed=seed,
        ssl=ssl,
        rampup=rampup,
        verbose=verbose,
    )
    logger.info("Ended training")

    ## Prediction
    logger.info("Started prediction")

    from .predict import predict

    model_file = os.path.join(train_output_folder, "model_final.pth")
    predictions_folder = os.path.join(output_folder, "predictions")
    overlayed_folder = (
        os.path.join(output_folder, "overlayed", "probabilities")
        if overlayed
        else None
    )

    # train set
    ctx.invoke(
        predict,
        output_folder=predictions_folder,
        model=model,
        dataset=train_dataset,
        batch_size=batch_size,
        device=device,
        weight=model_file,
        overlayed=overlayed_folder,
        verbose=verbose,
    )

    # test set
    ctx.invoke(
        predict,
        output_folder=predictions_folder,
        model=model,
        dataset=test_dataset,
        batch_size=batch_size,
        device=device,
        weight=model_file,
        overlayed=overlayed_folder,
        verbose=verbose,
    )
    logger.info("Ended prediction")

    ## Evaluation
    logger.info("Started evaluation")

    from .evaluate import evaluate

    overlayed_folder = (
        os.path.join(output_folder, "overlayed", "analysis")
        if overlayed
        else None
    )

    # train set
    train_analysis_folder = os.path.join(output_folder, "analysis", "train")
    ctx.invoke(
        evaluate,
        output_folder=train_analysis_folder,
        predictions_folder=predictions_folder,
        dataset=train_dataset,
        overlayed=overlayed_folder,
        overlay_threshold=0.5,
        verbose=verbose,
    )

    # test set
    test_analysis_folder = os.path.join(output_folder, "analysis", "test")
    ctx.invoke(
        evaluate,
        output_folder=test_analysis_folder,
        predictions_folder=predictions_folder,
        dataset=test_dataset,
        overlayed=overlayed_folder,
        overlay_threshold=0.5,
        verbose=verbose,
    )
    logger.info("Ended evaluation")

    ## Comparison
    logger.info("Started comparison")

    # compare train and test set performances
    from .compare import compare

    systems = (
            "train": os.path.join(train_analysis_folder, "metric.csv"),
            "test": os.path.join(test_analysis_folder, "metric.csv"),
            )
    output_pdf = os.path.join(output_folder, "comparison.pdf")
    ctx.invoke(compare, label_path=systems, output=output_pdf, verbose=verbose)
    logger.info("End comparison, and the experiment - bye.")
