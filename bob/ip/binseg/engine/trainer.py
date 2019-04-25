#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
import datetime
from tqdm import tqdm
import torch
import os 
import pandas as pd

from bob.ip.binseg.utils.metric import SmoothedValue
from bob.ip.binseg.utils.plot import loss_curve

def do_train(
    model,
    data_loader,
    optimizer,
    criterion,
    scheduler,
    checkpointer,
    checkpoint_period,
    device,
    arguments,
    output_folder
):
    """ Trains the model """
    logger = logging.getLogger("bob.ip.binseg.engine.trainer")
    logger.info("Start training")
    start_epoch = arguments["epoch"]
    max_epoch = arguments["max_epoch"]

    # Logg to file
    with open (os.path.join(output_folder,"{}_trainlog.csv".format(model.name)), "a+") as outfile:
        
        model.train().to(device)
        # Total training timer
        start_training_time = time.time()

        for epoch in range(start_epoch, max_epoch):
            scheduler.step()
            losses = SmoothedValue(len(data_loader))
            epoch = epoch + 1
            arguments["epoch"] = epoch
            start_epoch_time = time.time()

            for images, ground_truths, masks, _ in tqdm(data_loader):

                images = images.to(device)
                ground_truths = ground_truths.to(device)
                #masks = masks.to(device) 

                outputs = model(images)
                loss = criterion(outputs, ground_truths)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.update(loss)
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
                        "{lr:.6f}, "
                        "{memory:.0f}"
                        "\n"
                        ).format(
                    eta=eta_string,
                    epoch=epoch,
                    avg_loss=losses.avg,
                    median_loss=losses.median,
                    lr=optimizer.param_groups[0]["lr"],
                    memory = (torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) if torch.cuda.is_available() else .0,
                    )
                )  
            logger.info(("eta: {eta}, " 
                        "epoch: {epoch}, "
                        "avg. loss: {avg_loss:.6f}, "
                        "median loss: {median_loss:.6f}, "
                        "lr: {lr:.6f}, "
                        "max mem: {memory:.0f}"
                        ).format(
                    eta=eta_string,
                    epoch=epoch,
                    avg_loss=losses.avg,
                    median_loss=losses.median,
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
    logdf = pd.read_csv(os.path.join(output_folder,"{}_trainlog.csv".format(model.name)),header=None, names=["avg. loss", "median loss","lr","max memory"])
    fig = loss_curve(logdf)
    logger.info("saving {}".format(log_plot_file))
    fig.savefig(log_plot_file)
