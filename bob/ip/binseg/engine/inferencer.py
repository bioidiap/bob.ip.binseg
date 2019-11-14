#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import time
import datetime
import numpy as np
import torch
import pandas as pd
import torchvision.transforms.functional as VF
from tqdm import tqdm

import bob.io.base

from bob.ip.binseg.utils.metric import SmoothedValue, base_metrics
from bob.ip.binseg.utils.plot import precision_recall_f1iso_confintval
from bob.ip.binseg.utils.summary import summary



def batch_metrics(predictions, ground_truths, names, output_folder, logger):
    """
    Calculates metrics on the batch and saves it to disc

    Parameters
    ----------
    predictions : :py:class:`torch.Tensor`
        tensor with pixel-wise probabilities
    ground_truths : :py:class:`torch.Tensor`
        tensor with binary ground-truth
    names : list
        list of file names
    output_folder : str
        output path
    logger : :py:class:`logging.Logger`
        python logger

    Returns
    -------
    list
        list containing batch metrics: ``[name, threshold, precision, recall, specificity, accuracy, jaccard, f1_score]``
    """
    step_size = 0.01
    batch_metrics = []

    for j in range(predictions.size()[0]):
        # ground truth byte
        gts = ground_truths[j].byte()

        file_name = "{}.csv".format(names[j])
        logger.info("saving {}".format(file_name))

        with open (os.path.join(output_folder,file_name), "w+") as outfile:

            outfile.write("threshold, precision, recall, specificity, accuracy, jaccard, f1_score\n")

            for threshold in np.arange(0.0,1.0,step_size):
                # threshold
                binary_pred = torch.gt(predictions[j], threshold).byte()

                # equals and not-equals
                equals = torch.eq(binary_pred, gts) # tensor
                notequals = torch.ne(binary_pred, gts) # tensor

                # true positives
                tp_tensor = (gts * binary_pred ) # tensor
                tp_count = torch.sum(tp_tensor).item() # scalar

                # false positives
                fp_tensor = torch.eq((binary_pred + tp_tensor), 1)
                fp_count = torch.sum(fp_tensor).item()

                # true negatives
                tn_tensor = equals - tp_tensor
                tn_count = torch.sum(tn_tensor).item()

                # false negatives
                fn_tensor = notequals - fp_tensor
                fn_count = torch.sum(fn_tensor).item()

                # calc metrics
                metrics = base_metrics(tp_count, fp_count, tn_count, fn_count)

                # write to disk
                outfile.write("{:.2f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f} \n".format(threshold, *metrics))

                batch_metrics.append([names[j],threshold, *metrics ])


    return batch_metrics


def save_probability_images(predictions, names, output_folder, logger):
    """
    Saves probability maps as image in the same format as the test image

    Parameters
    ----------
    predictions : :py:class:`torch.Tensor`
        tensor with pixel-wise probabilities
    names : list
        list of file names
    output_folder : str
        output path
    logger : :py:class:`logging.Logger`
        python logger
    """
    images_subfolder = os.path.join(output_folder,'images')
    for j in range(predictions.size()[0]):
        img = VF.to_pil_image(predictions.cpu().data[j])
        filename = '{}.png'.format(names[j].split(".")[0])
        fullpath = os.path.join(images_subfolder, filename)
        logger.info("saving {}".format(fullpath))
        fulldir = os.path.dirname(fullpath)
        if not os.path.exists(fulldir): os.makedirs(fulldir)
        img.save(fullpath)

def save_hdf(predictions, names, output_folder, logger):
    """
    Saves probability maps as image in the same format as the test image

    Parameters
    ----------
    predictions : :py:class:`torch.Tensor`
        tensor with pixel-wise probabilities
    names : list
        list of file names
    output_folder : str
        output path
    logger : :py:class:`logging.Logger`
        python logger
    """
    hdf5_subfolder = os.path.join(output_folder,'hdf5')
    if not os.path.exists(hdf5_subfolder): os.makedirs(hdf5_subfolder)
    for j in range(predictions.size()[0]):
        img = predictions.cpu().data[j].squeeze(0).numpy()
        filename = '{}.hdf5'.format(names[j].split(".")[0])
        fullpath = os.path.join(hdf5_subfolder, filename)
        logger.info("saving {}".format(filename))
        fulldir = os.path.dirname(fullpath)
        if not os.path.exists(fulldir): os.makedirs(fulldir)
        bob.io.base.save(img, fullpath)

def do_inference(
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

    # Collect overall metrics
    metrics = []

    for samples in tqdm(data_loader):
        names = samples[0]
        images = samples[1].to(device)
        ground_truths = samples[2].to(device)
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

            b_metrics = batch_metrics(probabilities, ground_truths, names,results_subfolder, logger)
            metrics.extend(b_metrics)

            # Create probability images
            save_probability_images(probabilities, names, output_folder, logger)
            # save hdf5
            save_hdf(probabilities, names, output_folder, logger)

    # DataFrame
    df_metrics = pd.DataFrame(metrics,columns= \
                           ["name",
                            "threshold",
                            "precision",
                            "recall",
                            "specificity",
                            "accuracy",
                            "jaccard",
                            "f1_score"])

    # Report and Averages
    metrics_file = "Metrics.csv".format(model.name)
    metrics_path = os.path.join(results_subfolder, metrics_file)
    logger.info("Saving average over all input images: {}".format(metrics_file))

    avg_metrics = df_metrics.groupby('threshold').mean()
    std_metrics = df_metrics.groupby('threshold').std()

    # Uncomment below for F1-score calculation based on average precision and metrics instead of
    # F1-scores of individual images. This method is in line with Maninis et. al. (2016)
    #avg_metrics["f1_score"] =  (2* avg_metrics["precision"]*avg_metrics["recall"])/ \
    #    (avg_metrics["precision"]+avg_metrics["recall"])

    avg_metrics["std_pr"] = std_metrics["precision"]
    avg_metrics["pr_upper"] = avg_metrics['precision'] + avg_metrics["std_pr"]
    avg_metrics["pr_lower"] = avg_metrics['precision'] - avg_metrics["std_pr"]
    avg_metrics["std_re"] = std_metrics["recall"]
    avg_metrics["re_upper"] = avg_metrics['recall'] + avg_metrics["std_re"]
    avg_metrics["re_lower"] = avg_metrics['recall'] - avg_metrics["std_re"]
    avg_metrics["std_f1"] = std_metrics["f1_score"]

    avg_metrics.to_csv(metrics_path)
    maxf1 = avg_metrics['f1_score'].max()
    optimal_f1_threshold = avg_metrics['f1_score'].idxmax()

    logger.info("Highest F1-score of {:.5f}, achieved at threshold {}".format(maxf1, optimal_f1_threshold))

    # Plotting
    np_avg_metrics = avg_metrics.to_numpy().T
    fig_name = "precision_recall.pdf"
    logger.info("saving {}".format(fig_name))
    fig = precision_recall_f1iso_confintval([np_avg_metrics[0]],[np_avg_metrics[1]],[np_avg_metrics[7]],[np_avg_metrics[8]],[np_avg_metrics[10]],[np_avg_metrics[11]], [model.name,None], title=output_folder)
    fig_filename = os.path.join(results_subfolder, fig_name)
    fig.savefig(fig_filename)

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

    # Save model summary
    summary_file = 'ModelSummary.txt'
    logger.info("saving {}".format(summary_file))

    with open (os.path.join(results_subfolder,summary_file), "w+") as outfile:
        summary(model,outfile)



