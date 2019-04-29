#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging
import time
import datetime
from tqdm import tqdm
import torch
import numpy as np
import pickle
import pandas as pd

from bob.ip.binseg.utils.metric import SmoothedValue, base_metrics
from bob.ip.binseg.utils.plot import precision_recall_f1iso

import torchvision.transforms.functional as VF

def batch_metrics(predictions, ground_truths, masks, names, output_folder, logger):
    """
    calculates metrics on the batch and saves it to disc

    Parameters
    ----------
    predictions: :py:class:torch.Tensor
    ground_truths : :py:class:torch.Tensor
    mask : :py:class:torch.Tensor
    names : list
    output_folder : str
    logger : logger

    Returns
    -------

    batch_metrics : list

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
                # TODO: Substract masks from True negatives?

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
    images_subfolder = os.path.join(output_folder,'images') 
    if not os.path.exists(images_subfolder): os.makedirs(images_subfolder)
    for j in range(predictions.size()[0]):
        img = VF.to_pil_image(predictions.cpu().data[j])
        filename = '{}_prob.gif'.format(names[j])
        logger.info("saving {}".format(filename))
        img.save(os.path.join(images_subfolder, filename))



def do_inference(
    model,
    data_loader,
    device,
    output_folder = None
):
    logger = logging.getLogger("bob.ip.binseg.engine.inference")
    logger.info("Start evaluation")
    logger.info("Split: {}, Output folder: {}, Device: {}".format(data_loader.dataset.split, output_folder, device))
    results_subfolder = os.path.join(output_folder,'results') 
    if not os.path.exists(results_subfolder): os.makedirs(results_subfolder)
    
    model.eval().to(device)
    # Sigmoid for probabilities 
    sigmoid = torch.nn.Sigmoid() 

    # Setup timers
    start_total_time = time.time()
    times = []

    # Collect overall metrics 
    metrics = []

    for images, ground_truths, masks, names in tqdm(data_loader):
        images = images.to(device)
        ground_truths = ground_truths.to(device)
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
            
            b_metrics = batch_metrics(probabilities, ground_truths, masks, names,results_subfolder, logger)
            metrics.extend(b_metrics)
            # Create probability images
            save_probability_images(probabilities, names, output_folder, logger)


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
    metrics_file = "Metrics_{}.csv".format(model.name)
    metrics_path = os.path.join(results_subfolder, metrics_file)
    logger.info("Saving average over all input images: {}".format(metrics_file))
    
    avg_metrics = df_metrics.groupby('threshold').mean()
    avg_metrics["model_name"] = model.name
    avg_metrics.to_csv(metrics_path)

    avg_metrics["f1_score"] =  2* avg_metrics["precision"]*avg_metrics["recall"]/ \
        (avg_metrics["precision"]+avg_metrics["recall"])
    
    maxf1 = avg_metrics['f1_score'].max()
    optimal_f1_threshold = avg_metrics['f1_score'].idxmax()
    
    logger.info("Highest F1-score of {:.5f}, achieved at threshold {}".format(maxf1, optimal_f1_threshold))
    
    # Plotting
    np_avg_metrics = avg_metrics.to_numpy().T
    fig_name = "precision_recall_{}.pdf".format(model.name)
    logger.info("saving {}".format(fig_name))
    fig = precision_recall_f1iso([np_avg_metrics[0]],[np_avg_metrics[1]], np_avg_metrics[-1])
    fig_filename = os.path.join(results_subfolder, fig_name)
    fig.savefig(fig_filename)
    
    # Report times
    total_inference_time = str(datetime.timedelta(seconds=int(sum(times))))
    average_batch_inference_time = np.mean(times)
    total_evalution_time = str(datetime.timedelta(seconds=int(time.time() - start_total_time )))

    logger.info("Average batch inference time: {:.5f}s".format(average_batch_inference_time))

    times_file = "Times_{}.txt".format(model.name)
    logger.info("saving {}".format(times_file))
        
    with open (os.path.join(results_subfolder,times_file), "w+") as outfile:
        date = datetime.datetime.now()
        outfile.write("Date: {} \n".format(date.strftime("%Y-%m-%d %H:%M:%S")))
        outfile.write("Total evaluation run-time: {} \n".format(total_evalution_time))
        outfile.write("Average batch inference time: {} \n".format(average_batch_inference_time))
        outfile.write("Total inference time: {} \n".format(total_inference_time))


