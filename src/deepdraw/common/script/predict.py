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
import multiprocessing
import os
import sys

import torch

from torch.utils.data import DataLoader

from ..utils.checkpointer import Checkpointer
from .common import download_to_tempfile, setup_pytorch_device

logger = logging.getLogger(__name__)


def base_predict(
    output_folder,
    model,
    dataset,
    batch_size,
    device,
    weight,
    overlayed,
    parallel,
    detection,
    **kwargs,
):
    """Create base predict function for segmentation / detection tasks."""
    device = setup_pytorch_device(device)

    dataset = dataset if isinstance(dataset, dict) else dict(test=dataset)

    if weight.startswith("http"):
        logger.info(f"Temporarily downloading '{weight}'...")
        f = download_to_tempfile(weight, progress=True)
        weight_fullpath = os.path.abspath(f.name)
    else:
        weight_fullpath = os.path.abspath(weight)

    checkpointer = Checkpointer(model)
    checkpointer.load(weight_fullpath)

    # clean-up the overlayed path
    if overlayed is not None:
        overlayed = overlayed.strip()

    for k, v in dataset.items():
        if k.startswith("_"):
            logger.info(f"Skipping dataset '{k}' (not to be evaluated)")
            continue

        logger.info(f"Running inference on '{k}' set...")

        # PyTorch dataloader
        multiproc_kwargs = dict()
        if parallel < 0:
            multiproc_kwargs["num_workers"] = 0
        else:
            multiproc_kwargs["num_workers"] = (
                parallel or multiprocessing.cpu_count()
            )

        if multiproc_kwargs["num_workers"] > 0 and sys.platform.startswith(
            "darwin"
        ):
            multiproc_kwargs[
                "multiprocessing_context"
            ] = multiprocessing.get_context("spawn")

        if detection:
            from ...detect.engine.predictor import run

            def _collate_fn(batch):
                return tuple(zip(*batch))

            data_loader = DataLoader(
                dataset=v,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                collate_fn=_collate_fn,
                **multiproc_kwargs,
            )
        else:
            from ...binseg.engine.predictor import run

            data_loader = DataLoader(
                dataset=v,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                **multiproc_kwargs,
            )

        run(model, data_loader, k, device, output_folder, overlayed)
