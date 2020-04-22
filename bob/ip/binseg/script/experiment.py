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


def _save_sh_command(destfile):
    """Records command-line to reproduce this experiment"""

    import sys
    import time
    import pkg_resources

    dirname = os.path.dirname(destfile)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    logger.info(f"Writing command-line for reproduction at '{destfile}'...")

    with open(destfile, "wt") as f:
        f.write("#!/usr/bin/env sh\n")
        f.write(f"# date: {time.asctime()}\n")
        version = pkg_resources.require('bob.ip.binseg')[0].version
        f.write(f"# version: {version} (bob.ip.binseg)\n")
        f.write(f"# platform: {sys.platform}\n")
        f.write("\n")
        args = []
        for k in sys.argv:
            if " " in k: args.append(f'"{k}"')
            else: args.append(k)
        if os.environ.get('CONDA_DEFAULT_ENV') is not None:
            f.write(f"#conda activate {os.environ['CONDA_DEFAULT_ENV']}\n")
        f.write(f"#cd {os.path.realpath(os.curdir)}\n")
        f.write(" ".join(args) + "\n")
    os.chmod(destfile, 0o755)


@click.command(
    entry_point_group="bob.ip.binseg.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Trains a M2U-Net model (VGG-16 backbone) with DRIVE (vessel segmentation),
       on the CPU, for only two epochs, then runs inference and evaluation on
       results from its test set:

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
    dataset,
    second_annotator,
    checkpoint_period,
    device,
    seed,
    ssl,
    rampup,
    overlayed,
    verbose,
    **kwargs,
):
    """Runs a complete experiment, from training, to prediction and evaluation

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

    _save_sh_command(os.path.join(output_folder, "command.sh"))

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
        dataset=dataset,
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
        os.path.join(output_folder, "overlayed", "predictions")
        if overlayed
        else None
    )

    ctx.invoke(
        predict,
        output_folder=predictions_folder,
        model=model,
        dataset=dataset,
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

    analysis_folder = os.path.join(output_folder, "analysis")
    second_annotator_folder = os.path.join(analysis_folder, "second-annotator")
    ctx.invoke(
        evaluate,
        output_folder=analysis_folder,
        predictions_folder=predictions_folder,
        dataset=dataset,
        second_annotator=second_annotator,
        second_annotator_folder=second_annotator_folder,
        overlayed=overlayed_folder,
        overlay_threshold=0.5,
        verbose=verbose,
    )

    logger.info("Ended evaluation")

    ## Comparison
    logger.info("Started comparison")

    # compare performances on the various sets
    from .compare import compare

    systems = []
    for k, v in dataset.items():
        if k.startswith("_"):
            logger.info(f"Skipping dataset '{k}' (not to be compared)")
            continue
        systems += [k, os.path.join(analysis_folder, k, "metrics.csv")]
    if second_annotator is not None:
        for k, v in second_annotator.items():
            if k.startswith("_"):
                logger.info(f"Skipping dataset '{k}' (not to be compared)")
                continue
            systems += [f"{k} (2nd. annot.)",
                    os.path.join(second_annotator_folder, k, "metrics.csv")]
    output_pdf = os.path.join(output_folder, "comparison.pdf")
    ctx.invoke(compare, label_path=systems, output=output_pdf, verbose=verbose)

    logger.info("Ended comparison, and the experiment - bye.")
