#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import os

from .model_serialization import load_state_dict
from .model_zoo import cache_url

import logging
logger = logging.getLogger(__name__)


class Checkpointer:
    """Adapted from `maskrcnn-benchmark
    <https://github.com/facebookresearch/maskrcnn-benchmark>`_ under MIT license
    """

    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        dest_filename = f"{name}.pth"
        save_file = os.path.join(self.save_dir, dest_filename)
        logger.info(f"Saving checkpoint to {save_file}")
        torch.save(data, save_file)
        self.tag_last_checkpoint(dest_filename)

    def load(self, f=None):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            logger.warn("No checkpoint found. Initializing model from scratch")
            return {}
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        actual_file = os.path.join(self.save_dir, f)
        if "optimizer" in checkpoint and self.optimizer:
            logger.info(f"Loading optimizer from {actual_file}")
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            logger.info(f"Loading scheduler from {actual_file}")
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        actual_file = os.path.join(self.save_dir, f)
        logger.info(f"Loading checkpoint from {actual_file}")
        return torch.load(actual_file, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk
        )

    def _load_file(self, f):
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            logger.info(f"url {f} cached in {cached_f}")
            f = cached_f
        # load checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded
