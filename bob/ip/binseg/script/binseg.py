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
from bob.ip.binseg.engine.ssltrainer import do_ssltrain
from bob.ip.binseg.engine.inferencer import do_inference
from bob.ip.binseg.utils.plot import plot_overview
from bob.ip.binseg.utils.click import OptionEatAll
from bob.ip.binseg.utils.rsttable import create_overview_grid
from bob.ip.binseg.utils.plot import metricsviz, overlay,savetransformedtest
from bob.ip.binseg.utils.transformfolder import transformfolder as transfld
from bob.ip.binseg.utils.evaluate import do_eval
from bob.ip.binseg.engine.predicter import do_predict

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
    default=1000,
    cls=ResourceOption)
@click.option(
    '--checkpoint-period',
    '-p',
    help='Number of epochs after which a checkpoint is saved',
    show_default=True,
    required=True,
    default=100,
    cls=ResourceOption)
@click.option(
    '--device',
    '-d',
    help='A string indicating the device to use (e.g. "cpu" or "cuda:0"',
    show_default=True,
    required=True,
    default='cpu',
    cls=ResourceOption)
@click.option(
    '--seed',
    '-s',
    help='torch random seed',
    show_default=True,
    required=False,
    default=42,
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
        ,seed
        ,**kwargs):
    """ Train a model """
    
    if not os.path.exists(output_path): os.makedirs(output_path)
    torch.manual_seed(seed)
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
@click.option(
    '--weight',
    '-w',
    help='Path or URL to pretrained model',
    required=False,
    default=None,
    cls=ResourceOption
    )
@verbosity_option(cls=ResourceOption)
def test(model
        ,output_path
        ,device
        ,batch_size
        ,dataset
        ,weight
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
    checkpointer.load(weight)
    do_inference(model, data_loader, device, output_path)



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
@click.option(
    '--title',
    '-t',
    required=False,
    )
@verbosity_option(cls=ResourceOption)
def compare(output_path_list, output_path, title, **kwargs):
    """ Compares multiple metrics files that are stored in the format mymodel/results/Metrics.csv """
    logger.debug("Output paths: {}".format(output_path_list))
    logger.info('Plotting precision vs recall curves for {}'.format(output_path_list))
    fig = plot_overview(output_path_list,title)
    if not os.path.exists(output_path): os.makedirs(output_path)
    fig_filename = os.path.join(output_path, 'precision_recall_comparison.pdf')
    logger.info('saving {}'.format(fig_filename))
    fig.savefig(fig_filename)


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
    tree structure: 
        ├── DATABASE
        ├── MODEL
            ├── images
            └── results
    """
    logger.info('Creating grid for all results in {}'.format(output_path))
    create_overview_grid(output_path)


# Create metrics viz
@binseg.command(entry_point_group='bob.ip.binseg.config', cls=ConfigCommand)
@click.option(
    '--dataset',
    '-d',
    required=True,
    cls=ResourceOption
    )
@click.option(
    '--output-path',
    '-o',
    required=True,
    )
@verbosity_option(cls=ResourceOption)
def visualize(dataset, output_path, **kwargs):
    """ Creates the following visualizations of the probabilties output maps:
    overlayed: test images overlayed with prediction probabilities vessel tree
    tpfnfpviz: highlights true positives, false negatives and false positives

    Required tree structure: 
    ├── DATABASE
        ├── MODEL
            ├── images
            └── results
    """
    logger.info('Creating TP, FP, FN visualizations for {}'.format(output_path))
    metricsviz(dataset=dataset, output_path=output_path)
    logger.info('Creating overlay visualizations for {}'.format(output_path))
    overlay(dataset=dataset, output_path=output_path)
    logger.info('Saving transformed test images {}'.format(output_path))
    savetransformedtest(dataset=dataset, output_path=output_path)


# SSLTrain
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
    default=1000,
    cls=ResourceOption)
@click.option(
    '--checkpoint-period',
    '-p',
    help='Number of epochs after which a checkpoint is saved',
    show_default=True,
    required=True,
    default=100,
    cls=ResourceOption)
@click.option(
    '--device',
    '-d',
    help='A string indicating the device to use (e.g. "cpu" or "cuda:0"',
    show_default=True,
    required=True,
    default='cpu',
    cls=ResourceOption)
@click.option(
    '--rampup',
    '-r',
    help='Ramp-up length in epochs',
    show_default=True,
    required=True,
    default='900',
    cls=ResourceOption)
@click.option(
    '--seed',
    '-s',
    help='torch random seed',
    show_default=True,
    required=False,
    default=42,
    cls=ResourceOption)

@verbosity_option(cls=ResourceOption)
def ssltrain(model
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
        ,rampup
        ,seed
        ,**kwargs):
    """ Train a model """
    
    if not os.path.exists(output_path): os.makedirs(output_path)
    torch.manual_seed(seed)
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
    do_ssltrain(model
            , data_loader
            , optimizer
            , criterion
            , scheduler
            , checkpointer
            , checkpoint_period
            , device
            , arguments
            , output_path
            , rampup
            )

# Apply image transforms to a folder containing images
@binseg.command(entry_point_group='bob.ip.binseg.config', cls=ConfigCommand)
@click.option(
    '--source-path',
    '-s',
    required=True,
    cls=ResourceOption
    )
@click.option(
    '--target-path',
    '-t',
    required=True,
    cls=ResourceOption
    )
@click.option(
    '--transforms',
    '-a',
    required=True,
    cls=ResourceOption
    )

@verbosity_option(cls=ResourceOption)
def transformfolder(source_path ,target_path,transforms,**kwargs):
    logger.info('Applying transforms to images in {} and saving them to {}'.format(source_path, target_path))
    transfld(source_path,target_path,transforms)


# Run inference and create predictions only (no ground truth available)
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
@click.option(
    '--weight',
    '-w',
    help='Path or URL to pretrained model',
    required=False,
    default=None,
    cls=ResourceOption
    )
@verbosity_option(cls=ResourceOption)
def predict(model
        ,output_path
        ,device
        ,batch_size
        ,dataset
        ,weight
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
    checkpointer.load(weight)
    do_predict(model, data_loader, device, output_path)

    # Overlayed images
    overlay(dataset=dataset, output_path=output_path)



# Evaluate only. Runs evaluation on predicted probability maps (--prediction-folder)
@binseg.command(entry_point_group='bob.ip.binseg.config', cls=ConfigCommand)
@click.option(
    '--output-path',
    '-o',
    required=True,
    default="output",
    cls=ResourceOption
    )
@click.option(
    '--prediction-folder',
    '-p',
    help = 'Path containing output probability maps',
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
    '--title',
    required=False,
    cls=ResourceOption
    )
@click.option(
    '--legend',
    cls=ResourceOption
    )

@verbosity_option(cls=ResourceOption)
def evalpred(
        output_path
        ,prediction_folder
        ,dataset
        ,title
        ,legend
        , **kwargs):
    """ Run inference and evalaute the model performance """

    # PyTorch dataloader
    data_loader = DataLoader(
        dataset = dataset
        ,batch_size = 1
        ,shuffle= False
        ,pin_memory = torch.cuda.is_available()
        )
    
    # Run eval
    do_eval(prediction_folder, data_loader, output_folder = output_path, title= title, legend=legend)


    