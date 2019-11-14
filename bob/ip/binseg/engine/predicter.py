#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import time
import datetime
import numpy as np
import torch
import torchvision.transforms.functional as VF
from tqdm import tqdm

from bob.ip.binseg.utils.summary import summary
from bob.ip.binseg.engine.inferencer import save_probability_images
from bob.ip.binseg.engine.inferencer import save_hdf


def do_predict(
    model,
    data_loader,
    device,
    output_folder = None
):

    """
    Run inference and calculate metrics

    Parameters
    ---------
    model : :py:class:`torch.nn.Module`
        neural network model (e.g. DRIU, HED, UNet)
    data_loader : py:class:`torch.torch.utils.data.DataLoader`
    device : str
        device to use ``'cpu'`` or ``'cuda'``
    output_folder : str
    """
    logger = logging.getLogger("bob.ip.binseg.engine.inference")
    logger.info("Start evaluation")
    logger.info("Output folder: {}, Device: {}".format(output_folder, device))
    results_subfolder = os.path.join(output_folder,'results')
    os.makedirs(results_subfolder,exist_ok=True)

    model.eval().to(device)
    # Sigmoid for probabilities
    sigmoid = torch.nn.Sigmoid()

    # Setup timers
    start_total_time = time.time()
    times = []

    for samples in tqdm(data_loader):
        names = samples[0]
        images = samples[1].to(device)
        with torch.no_grad():
            start_time = time.perf_counter()

            outputs = model(images)

            # necessary check for hed architecture that uses several outputs
            # for loss calculation instead of just the last concatfuse block
            if isinstance(outputs,list):
                outputs = outputs[-1]

            probabilities = sigmoid(outputs)

            batch_time = time.perf_counter() - start_time
            times.append(batch_time)
            logger.info("Batch time: {:.5f} s".format(batch_time))

            # Create probability images
            save_probability_images(probabilities, names, output_folder, logger)
            # Save hdf5
            save_hdf(probabilities, names, output_folder, logger)


    # Report times
    total_inference_time = str(datetime.timedelta(seconds=int(sum(times))))
    average_batch_inference_time = np.mean(times)
    total_evalution_time = str(datetime.timedelta(seconds=int(time.time() - start_total_time )))

    logger.info("Average batch inference time: {:.5f}s".format(average_batch_inference_time))

    times_file = "Times.txt"
    logger.info("saving {}".format(times_file))

    with open (os.path.join(results_subfolder,times_file), "w+") as outfile:
        date = datetime.datetime.now()
        outfile.write("Date: {} \n".format(date.strftime("%Y-%m-%d %H:%M:%S")))
        outfile.write("Total evaluation run-time: {} \n".format(total_evalution_time))
        outfile.write("Average batch inference time: {} \n".format(average_batch_inference_time))
        outfile.write("Total inference time: {} \n".format(total_inference_time))


