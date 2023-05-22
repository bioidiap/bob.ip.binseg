# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import os

import torch

logger = logging.getLogger(__name__)


def get_checkpoint(output_folder, resume_from):
    """Gets a checkpoint file.

    Can return the best or last checkpoint, or a checkpoint at a specific path.
    Ensures the checkpoint exists, raising an error if it is not the case.

    Parameters
    ----------

    output_folder : :py:class:`str`
        Directory in which checkpoints are stored.

    resume_from : :py:class:`str`
        Which model to get. Can be one of "best", "last", or a path to a checkpoint.

    Returns
    -------

    checkpoint_file : :py:class:`str`
        The requested model.
    """
    last_checkpoint_path = os.path.join(output_folder, "model_final_epoch.ckpt")
    best_checkpoint_path = os.path.join(
        output_folder, "model_lowest_valid_loss.ckpt"
    )

    if resume_from == "last":
        if os.path.isfile(last_checkpoint_path):
            checkpoint_file = last_checkpoint_path
            logger.info(f"Resuming training from {resume_from} checkpoint")
        else:
            raise FileNotFoundError(
                f"Could not find checkpoint {last_checkpoint_path}"
            )

    elif resume_from == "best":
        if os.path.isfile(best_checkpoint_path):
            checkpoint_file = last_checkpoint_path
            logger.info(f"Resuming training from {resume_from} checkpoint")
        else:
            raise FileNotFoundError(
                f"Could not find checkpoint {best_checkpoint_path}"
            )

    elif resume_from is None:
        checkpoint_file = None

    else:
        if os.path.isfile(resume_from):
            checkpoint_file = resume_from
            logger.info(f"Resuming training from checkpoint {resume_from}")
        else:
            raise FileNotFoundError(f"Could not find checkpoint {resume_from}")

    return checkpoint_file