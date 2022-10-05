#!/usr/bin/env python
# coding=utf-8

import click
import logging
import os
import shutil

from .common import save_sh_command

logger = logging.getLogger(__name__)

@click.pass_context
def base_experiment(
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
    detection,
    verbose,
    **kwargs,
):
    """Create base experiment function for segmentation / detection tasks."""
    command_sh = os.path.join(output_folder, "command.sh")
    if os.path.exists(command_sh):
        backup = command_sh + "~"
        if os.path.exists(backup):
            os.unlink(backup)
        shutil.move(command_sh, backup)
    save_sh_command(command_sh)

    # training
    logger.info("Started training")

    from .train import base_train

    train_output_folder = os.path.join(output_folder, "model")
    ctx.invoke(
        base_train,
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
        detection=detection,
        verbose=verbose,
    )
    logger.info("Ended training")

    from .train_analysis import base_train_analysis

    ctx.invoke(
        base_train_analysis,
        log=os.path.join(train_output_folder, "trainlog.csv"),
        constants=os.path.join(train_output_folder, "constants.csv"),
        output_pdf=os.path.join(train_output_folder, "trainlog.pdf"),
        verbose=verbose,
    )

    from .analyze import base_analyze

    # preferably, we use the best model on the validation set
    # otherwise, we get the last saved model
    model_file = os.path.join(
        train_output_folder, "model_lowest_valid_loss.pth"
    )
    if not os.path.exists(model_file):
        model_file = os.path.join(train_output_folder, "model_final_epoch.pth")

    ctx.invoke(
        base_analyze,
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
        detection=detection,
        verbose=verbose,
    )
