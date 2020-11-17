#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import torch

import logging

logger = logging.getLogger(__name__)


class Checkpointer:
    """A simple pytorch checkpointer

    Parameters
    ----------

    model : torch.nn.Module
        Network model, eventually loaded from a checkpointed file

    optimizer : :py:mod:`torch.optim`, Optional
        Optimizer

    scheduler : :py:mod:`torch.optim`, Optional
        Learning rate scheduler

    path : :py:class:`str`, Optional
        Directory where to save checkpoints.

    """

    def __init__(self, model, optimizer=None, scheduler=None, path="."):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.path = os.path.realpath(path)

    def save(self, name, **kwargs):

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        name = f"{name}.pth"
        outf = os.path.join(self.path, name)
        logger.info(f"Saving checkpoint to {outf}")
        torch.save(data, outf)
        with open(self._last_checkpoint_filename, "w") as f:
            f.write(name)

    def load(self, f=None):
        """Loads model, optimizer and scheduler from file


        Parameters
        ==========

        f : :py:class:`str`, Optional
            Name of a file (absolute or relative to ``self.path``), that
            contains the checkpoint data to load into the model, and optionally
            into the optimizer and the scheduler.  If not specified, loads data
            from current path.

        """

        if f is None:
            f = self.last_checkpoint()

        if f is None:
            # no checkpoint could be found
            logger.warning("No checkpoint found (and none passed)")
            return {}

        # loads file data into memory
        logger.info(f"Loading checkpoint from {f}...")
        checkpoint = torch.load(f, map_location=torch.device("cpu"))

        # converts model entry to model parameters
        self.model.load_state_dict(checkpoint.pop("model"))

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        return checkpoint

    @property
    def _last_checkpoint_filename(self):
        return os.path.join(self.path, "last_checkpoint")

    def has_checkpoint(self):
        return os.path.exists(self._last_checkpoint_filename)

    def last_checkpoint(self):
        if self.has_checkpoint():
            with open(self._last_checkpoint_filename, "r") as fobj:
                return os.path.join(self.path, fobj.read().strip())
        return None
