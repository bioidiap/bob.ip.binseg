#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""The main entry for bob ip binseg (click-based) scripts."""


import os
import time
import numpy
import collections
import pkg_resources

import click
from click_plugins import with_plugins

import logging
import torch

import bob.extension
from bob.extension.scripts.click_helper import (verbosity_option,
    ConfigCommand, ResourceOption, AliasedGroup)

from bob.ip.binseg.utils.checkpointer import DetectronCheckpointer
from bob.ip.binseg.data.binsegdataset import BinSegDataset
from torch.utils.data import DataLoader
from bob.ip.binseg.engine.trainer import do_train
from bob.ip.binseg.engine.inferencer import do_inference

logger = logging.getLogger(__name__)


@with_plugins(pkg_resources.iter_entry_points('bob.ip.binseg.cli'))
@click.group(cls=AliasedGroup)
def binseg():
    """Binary 2D Fundus Image Segmentation Benchmark commands."""
    pass

# Train
@binseg.command(entry_point_group='bob.ip.binseg.config', cls=ConfigCommand)
@click.option(
    '--output-path',
    '-o',
    required=True,
    default="output",
    cls=ResourceOption
    )
@click.option(
    '--model',
    '-m',
    required=True,
    cls=ResourceOption
    )
@click.option(
    '--optimizer',
    required=True,
    cls=ResourceOption
    )
@click.option(
    '--criterion',
    required=True,
    cls=ResourceOption
    )
@click.option(
    '--scheduler',
    required=True,
    cls=ResourceOption
    )
@click.option(
    '--pretrained-backbone',
    required=True,
    cls=ResourceOption
    )
@click.option(
    '--bobdb',
    required=True,
    cls=ResourceOption
    )
@click.option(
    '--split',
    '-s',
    required=True,
    default='train',
    cls=ResourceOption)
@click.option(
    '--transforms',
    required=True,
    cls=ResourceOption)
@click.option(
    '--batch-size',
    '-b',
    required=True,
    default=2,
    cls=ResourceOption)
@click.option(
    '--epochs',
    '-e',
    help='Number of epochs used for training',
    show_default=True,
    required=True,
    default=6,
    cls=ResourceOption)
@click.option(
    '--checkpoint-period',
    '-p',
    help='Number of epochs after which a checkpoint is saved',
    show_default=True,
    required=True,
    default=2,
    cls=ResourceOption)
@click.option(
    '--device',
    '-d',
    help='A string indicating the device to use (e.g. "cpu" or "cuda:0"',
    show_default=True,
    required=True,
    default='cpu',
    cls=ResourceOption)

@verbosity_option(cls=ResourceOption)
def train(model
        ,optimizer
        ,scheduler
        ,output_path
        ,epochs
        ,pretrained_backbone
        ,split
        ,batch_size
        ,criterion
        ,bobdb
        ,transforms
        ,checkpoint_period
        ,device
        ,**kwargs):
    if not os.path.exists(output_path): os.makedirs(output_path)
    
    # PyTorch dataset
    bsdataset = BinSegDataset(bobdb, split=split, transform=transforms)

    # PyTorch dataloader
    data_loader = DataLoader(
        dataset = bsdataset
        ,batch_size = batch_size
        ,shuffle= True
        ,pin_memory = torch.cuda.is_available()
        )

    # Checkpointer
    checkpointer = DetectronCheckpointer(model, optimizer, scheduler,save_dir = output_path, save_to_disk=True)
    arguments = {}
    arguments["epoch"] = 0 
    arguments["max_epoch"] = epochs
    extra_checkpoint_data = checkpointer.load(pretrained_backbone)
    arguments.update(extra_checkpoint_data)
    
    # Train
    logger.info("Training for {} epochs".format(arguments["max_epoch"]))
    logger.info("Continuing from epoch {}".format(arguments["epoch"]))
    do_train(model
            , data_loader
            , optimizer
            , criterion
            , scheduler
            , checkpointer
            , checkpoint_period
            , device
            , arguments
            )

# Inference
@binseg.command(entry_point_group='bob.ip.binseg.config', cls=ConfigCommand)
@click.option(
    '--output-path',
    '-o',
    required=True,
    default="output",
    cls=ResourceOption
    )
@click.option(
    '--model',
    '-m',
    required=True,
    cls=ResourceOption
    )
@click.option(
    '--bobdb',
    required=True,
    cls=ResourceOption
    )
@click.option(
    '--transforms',
    required=True,
    cls=ResourceOption)
@click.option(
    '--split',
    '-s',
    required=True,
    default='test',
    cls=ResourceOption)
@click.option(
    '--batch-size',
    '-b',
    required=True,
    default=2,
    cls=ResourceOption)
@click.option(
    '--device',
    '-d',
    help='A string indicating the device to use (e.g. "cpu" or "cuda:0"',
    show_default=True,
    required=True,
    default='cpu',
    cls=ResourceOption)

@verbosity_option(cls=ResourceOption)
def test(model
        ,output_path
        ,device
        ,split
        ,batch_size
        ,bobdb
        ,transforms
        , **kwargs):

    # PyTorch dataset
    bsdataset = BinSegDataset(bobdb, split=split, transform=transforms)

    # PyTorch dataloader
    data_loader = DataLoader(
        dataset = bsdataset
        ,batch_size = batch_size
        ,shuffle= False
        ,pin_memory = torch.cuda.is_available()
        )

    # checkpointer, load last model in dir
    checkpointer = DetectronCheckpointer(model, save_dir = output_path, save_to_disk=False)
    checkpointer.load()
    do_inference(model, data_loader, device, output_path)