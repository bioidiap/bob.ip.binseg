#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging
import time
import datetime
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

from bob.ip.binseg.utils.metric import SmoothedValue
from bob.ip.binseg.utils.plot import loss_curve

def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(dim=1, keepdim=True)

def mix_up(alpha, input, target, unlabeled_input, unlabled_target):
    """Applies mix up as described in [MIXMATCH_19].
    
    Parameters
    ----------
    alpha : float
    input : :py:class:`torch.Tensor`
    target : :py:class:`torch.Tensor`
    unlabeled_input : :py:class:`torch.Tensor`
    unlabled_target : :py:class:`torch.Tensor`
    
    Returns
    -------
    list
    """
    # TODO: 
    with torch.no_grad():
        l = np.random.beta(alpha, alpha) # Eq (8)
        l = max(l, 1 - l) # Eq (9)
        # Shuffle and concat. Alg. 1 Line: 12
        w_inputs = torch.cat([input,unlabeled_input],0)
        w_targets = torch.cat([target,unlabled_target],0)
        idx = torch.randperm(w_inputs.size(0)) # get random index 
        
        # Apply MixUp to labeled data and entries from W. Alg. 1 Line: 13
        input_mixedup = l * input + (1 - l) * w_inputs[idx[len(input):]] 
        target_mixedup = l * target + (1 - l) * w_targets[idx[len(target):]]
        
        # Apply MixUp to unlabeled data and entries from W. Alg. 1 Line: 14
        unlabeled_input_mixedup = l * unlabeled_input + (1 - l) * w_inputs[idx[:len(unlabeled_input)]]
        unlabled_target_mixedup =  l * unlabled_target + (1 - l) * w_targets[idx[:len(unlabled_target)]]
        return input_mixedup, target_mixedup, unlabeled_input_mixedup, unlabled_target_mixedup


def square_rampup(current, rampup_length=16):
    """slowly ramp-up ``lambda_u``
    
    Parameters
    ----------
    current : int
        current epoch
    rampup_length : int, optional
        how long to ramp up, by default 16
    
    Returns
    -------
    float
        ramp up factor
    """
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip((current/ float(rampup_length))**2, 0.0, 1.0)
    return float(current)

def linear_rampup(current, rampup_length=16):
    """slowly ramp-up ``lambda_u``
    
    Parameters
    ----------
    current : int
        current epoch
    rampup_length : int, optional
        how long to ramp up, by default 16
    
    Returns
    -------
    float
        ramp up factor
    """
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
    return float(current)

def guess_labels(unlabeled_images, model):
    """
    Calculate the average predictions by 2 augmentations: horizontal and vertical flips
    Parameters
    ----------
    unlabeled_images : :py:class:`torch.Tensor`
        shape: ``[n,c,h,w]``
    target : :py:class:`torch.Tensor`
    
    Returns
    -------
    :py:class:`torch.Tensor`
        shape: ``[n,c,h,w]``.
    """
    with torch.no_grad():
        guess1 = torch.sigmoid(model(unlabeled_images)).unsqueeze(0)
        # Horizontal flip and unsqueeze to work with batches (increase flip dimension by 1)
        hflip = torch.sigmoid(model(unlabeled_images.flip(2))).unsqueeze(0)
        guess2  = hflip.flip(3)
        # Vertical flip and unsqueeze to work with batches (increase flip dimension by 1)
        vflip = torch.sigmoid(model(unlabeled_images.flip(3))).unsqueeze(0)
        guess3 = vflip.flip(4)
        # Concat
        concat = torch.cat([guess1,guess2,guess3],0)
        avg_guess = torch.mean(concat,0)
        return avg_guess

def do_ssltrain(
    model,
    data_loader,
    optimizer,
    criterion,
    scheduler,
    checkpointer,
    checkpoint_period,
    device,
    arguments,
    output_folder,
    rampup_length
):
    """ 
    Train model and save to disk.
    
    Parameters
    ----------
    model : :py:class:`torch.nn.Module` 
        Network (e.g. DRIU, HED, UNet)
    data_loader : :py:class:`torch.utils.data.DataLoader`
    optimizer : :py:mod:`torch.optim`
    criterion : :py:class:`torch.nn.modules.loss._Loss`
        loss function
    scheduler : :py:mod:`torch.optim`
        learning rate scheduler
    checkpointer : :py:class:`bob.ip.binseg.utils.checkpointer.DetectronCheckpointer`
        checkpointer
    checkpoint_period : int
        save a checkpoint every n epochs
    device : str  
        device to use ``'cpu'`` or ``'cuda'``
    arguments : dict
        start end end epochs
    output_folder : str 
        output path
    rampup_Length : int
        rampup epochs
    """
    logger = logging.getLogger("bob.ip.binseg.engine.trainer")
    logger.info("Start training")
    start_epoch = arguments["epoch"]
    max_epoch = arguments["max_epoch"]

    # Logg to file
    with open (os.path.join(output_folder,"{}_trainlog.csv".format(model.name)), "a+",1) as outfile:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        model.train().to(device)
        # Total training timer
        start_training_time = time.time()
        for epoch in range(start_epoch, max_epoch):
            scheduler.step()
            losses = SmoothedValue(len(data_loader))
            labeled_loss = SmoothedValue(len(data_loader))
            unlabeled_loss = SmoothedValue(len(data_loader))
            epoch = epoch + 1
            arguments["epoch"] = epoch
            
            # Epoch time
            start_epoch_time = time.time()

            for samples in tqdm(data_loader):
                # labeled
                images = samples[1].to(device)
                ground_truths = samples[2].to(device)
                unlabeled_images = samples[4].to(device)
                # labeled outputs
                outputs = model(images)
                unlabeled_outputs = model(unlabeled_images)
                # guessed unlabeled outputs
                unlabeled_ground_truths = guess_labels(unlabeled_images, model)
                #unlabeled_ground_truths = sharpen(unlabeled_ground_truths,0.5)
                #images, ground_truths, unlabeled_images, unlabeled_ground_truths = mix_up(0.75, images, ground_truths, unlabeled_images, unlabeled_ground_truths)
                ramp_up_factor = square_rampup(epoch,rampup_length=rampup_length)

                loss, ll, ul = criterion(outputs, ground_truths, unlabeled_outputs, unlabeled_ground_truths, ramp_up_factor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.update(loss)
                labeled_loss.update(ll)
                unlabeled_loss.update(ul)
                logger.debug("batch loss: {}".format(loss.item()))

            if epoch % checkpoint_period == 0:
                checkpointer.save("model_{:03d}".format(epoch), **arguments)

            if epoch == max_epoch:
                checkpointer.save("model_final", **arguments)

            epoch_time = time.time() - start_epoch_time


            eta_seconds = epoch_time * (max_epoch - epoch)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            outfile.write(("{epoch}, "
                        "{avg_loss:.6f}, "
                        "{median_loss:.6f}, "
                        "{median_labeled_loss},"
                        "{median_unlabeled_loss},"
                        "{lr:.6f}, "
                        "{memory:.0f}"
                        "\n"
                        ).format(
                    eta=eta_string,
                    epoch=epoch,
                    avg_loss=losses.avg,
                    median_loss=losses.median,
                    median_labeled_loss = labeled_loss.median,
                    median_unlabeled_loss = unlabeled_loss.median,
                    lr=optimizer.param_groups[0]["lr"],
                    memory = (torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) if torch.cuda.is_available() else .0,
                    )
                )  
            logger.info(("eta: {eta}, " 
                        "epoch: {epoch}, "
                        "avg. loss: {avg_loss:.6f}, "
                        "median loss: {median_loss:.6f}, "
                        "labeled loss: {median_labeled_loss}, "
                        "unlabeled loss: {median_unlabeled_loss}, "
                        "lr: {lr:.6f}, "
                        "max mem: {memory:.0f}"
                        ).format(
                    eta=eta_string,
                    epoch=epoch,
                    avg_loss=losses.avg,
                    median_loss=losses.median,
                    median_labeled_loss = labeled_loss.median,
                    median_unlabeled_loss = unlabeled_loss.median,
                    lr=optimizer.param_groups[0]["lr"],
                    memory = (torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) if torch.cuda.is_available() else .0
                    )
                )


        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / epoch)".format(
                total_time_str, total_training_time / (max_epoch)
            ))
        
    log_plot_file = os.path.join(output_folder,"{}_trainlog.pdf".format(model.name))
    logdf = pd.read_csv(os.path.join(output_folder,"{}_trainlog.csv".format(model.name)),header=None, names=["avg. loss", "median loss", "labeled loss", "unlabeled loss", "lr","max memory"])
    fig = loss_curve(logdf,output_folder)
    logger.info("saving {}".format(log_plot_file))
    fig.savefig(log_plot_file)
  