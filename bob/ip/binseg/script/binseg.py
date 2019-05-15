#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""The main entry for bob ip binseg (click-based) scripts."""


import os
import time
import numpy
import collections
import pkg_resources
import glob

import click
from click_plugins import with_plugins

import logging
import torch

import bob.extension
from bob.extension.scripts.click_helper import (verbosity_option,
    ConfigCommand, ResourceOption, AliasedGroup)

from bob.ip.binseg.utils.checkpointer import DetectronCheckpointer
from torch.utils.data import DataLoader
from bob.ip.binseg.engine.trainer import do_train
from bob.ip.binseg.engine.inferencer import do_inference
from bob.ip.binseg.utils.plot import plot_overview
from bob.ip.binseg.utils.click import OptionEatAll
from bob.ip.binseg.utils.pdfcreator import create_pdf, get_paths
from bob.ip.binseg.utils.rsttable import create_overview_grid

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
    '--dataset',
    '-d',
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
    '-t',
    required=True,
    cls=ResourceOption
    )
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
        ,batch_size
        ,criterion
        ,dataset
        ,checkpoint_period
        ,device
        ,**kwargs):
    """ Train a model """
    
    if not os.path.exists(output_path): os.makedirs(output_path)
    
    # PyTorch dataloader
    data_loader = DataLoader(
        dataset = dataset
        ,batch_size = batch_size
        ,shuffle= True
        ,pin_memory = torch.cuda.is_available()
        )

    # Checkpointer
    checkpointer = DetectronCheckpointer(model, optimizer, scheduler,save_dir = output_path, save_to_disk=True)
    arguments = {}
    arguments["epoch"] = 0 
    extra_checkpoint_data = checkpointer.load(pretrained_backbone)
    arguments.update(extra_checkpoint_data)
    arguments["max_epoch"] = epochs
    
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
            , output_path
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
    '--dataset',
    '-d',
    required=True,
    cls=ResourceOption
    )
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
        ,batch_size
        ,dataset
        , **kwargs):
    """ Run inference and evalaute the model performance """

    # PyTorch dataloader
    data_loader = DataLoader(
        dataset = dataset
        ,batch_size = batch_size
        ,shuffle= False
        ,pin_memory = torch.cuda.is_available()
        )
    
    # checkpointer, load last model in dir
    checkpointer = DetectronCheckpointer(model, save_dir = output_path, save_to_disk=False)
    checkpointer.load()
    do_inference(model, data_loader, device, output_path)


# Inference all checkpoints
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
    '--dataset',
    '-d',
    required=True,
    cls=ResourceOption
    )
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
def testcheckpoints(model
        ,output_path
        ,device
        ,batch_size
        ,dataset
        , **kwargs):

    """ Run inference and evaluate all checkpoints saved for a model"""
    # PyTorch dataloader
    data_loader = DataLoader(
        dataset = dataset
        ,batch_size = batch_size
        ,shuffle= False
        ,pin_memory = torch.cuda.is_available()
        )
    
    # list checkpoints
    ckpts = glob.glob(os.path.join(output_path,"*.pth"))
    # output
    for checkpoint in ckpts:
        ckpts_name = os.path.basename(checkpoint).split('.')[0]
        logger.info("Testing checkpoint: {}".format(ckpts_name))
        output_subfolder = os.path.join(output_path, ckpts_name)
        if not os.path.exists(output_subfolder): os.makedirs(output_subfolder)
        # checkpointer, load last model in dir
        checkpointer = DetectronCheckpointer(model, save_dir = output_subfolder, save_to_disk=False)
        checkpointer.load(checkpoint)
        do_inference(model, data_loader, device, output_subfolder)

# Plot comparison
@binseg.command(entry_point_group='bob.ip.binseg.config', cls=ConfigCommand)
@click.option(
    '--output-path-list',
    '-l',
    required=True,
    help='Pass all output paths as arguments',
    cls=OptionEatAll,
    )
@click.option(
    '--output-path',
    '-o',
    required=True,
    )
@verbosity_option(cls=ResourceOption)
def compare(output_path_list, output_path, **kwargs):
    """ Compares multiple metrics files that are stored in the format mymodel/results/Metrics.csv """
    logger.debug("Output paths: {}".format(output_path_list))
    logger.info('Plotting precision vs recall curves for {}'.format(output_path_list))
    fig = plot_overview(output_path_list)
    if not os.path.exists(output_path): os.makedirs(output_path)
    fig_filename = os.path.join(output_path, 'precision_recall_comparison.pdf')
    logger.info('saving {}'.format(fig_filename))
    fig.savefig(fig_filename)


# Plot overviews
@binseg.command(entry_point_group='bob.ip.binseg.config', cls=ConfigCommand)
@click.option(
    '--output-path',
    '-o',
    required=True,
    )
@verbosity_option(cls=ResourceOption)
def pdfoverview(output_path, **kwargs):
    """ Creates an overview pdf with all precision vs recall curves present in the output directory.
    Requires pdflatex to be available on the host."""
    # PR curves
    pr_filename = "precision_recall_comparison.pdf"
    pr_filenames = get_paths(output_path,pr_filename)
    create_pdf(output_path, pr_filenames, title='Precision vs Recall', tex_filename='pr_overview.tex')
    
    # Training curves
    trainlog_filename = "*trainlog.pdf"
    tl_file_names = get_paths(output_path,trainlog_filename)
    create_pdf(output_path, tl_file_names, title='Training', tex_filename='training_overview.tex')


# Create grid table with results
@binseg.command(entry_point_group='bob.ip.binseg.config', cls=ConfigCommand)
@click.option(
    '--output-path',
    '-o',
    required=True,
    )
@verbosity_option(cls=ResourceOption)
def gridtable(output_path, **kwargs):
    """ Creates an overview table in grid rst format for all Metrics.csv in the output_path
    tree structure: ``outputpath/DATABASE/MODEL`` """
    create_overview_grid(output_path)
    