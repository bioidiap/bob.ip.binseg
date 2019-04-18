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

        single_metrics_file_path = os.path.join(output_folder, "{}.csv".format(names[j]))
        logger.info("saving {}".format(single_metrics_file_path))
        
        with open (single_metrics_file_path, "w+") as outfile:

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
    for j in range(predictions.size()[0]):
        img = VF.to_pil_image(predictions.cpu().data[j])
        filename = '{}_prob.gif'.format(names[j])
        logger.info("saving {}".format(filename))
        img.save(os.path.join(output_folder, filename))



def do_inference(
    model,
    data_loader,
    device,
    output_folder = None
):
    logger = logging.getLogger("bob.ip.binseg.engine.inference")
    logger.info("Start evaluation")
    logger.info("Output folder: {}, Device: {}".format(output_folder, device))
    model.eval()
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
            probabilities = sigmoid(outputs)
            
            batch_time = time.perf_counter() - start_time
            times.append(batch_time)
            logger.info("Batch time: {:.5f} s".format(batch_time))
            
            b_metrics = batch_metrics(probabilities, ground_truths, masks, names, output_folder, logger)
            metrics.extend(b_metrics)
            save_probability_images(probabilities, names, output_folder, logger)

    # NOTE: comment out for debugging
    #with open (os.path.join(output_folder, "metrics.pkl"), "wb+") as outfile:
    #    logger.debug("Saving metrics to {}".format(output_folder))
    #    pickle.dump(metrics, outfile)

    df_metrics = pd.DataFrame(metrics,columns= \
                           ["name",
                            "threshold",
                            "precision", 
                            "recall", 
                            "specificity", 
                            "accuracy", 
                            "jaccard", 
                            "f1_score"])

    
    # Save to disk
    metrics_path = os.path.join(output_folder, "Metrics.csv")
    logging.info("Saving average over all inputs: {}".format(metrics_path))
    
    # Report Averages
    avg_metrics = df_metrics.groupby('threshold').mean()
    avg_metrics.to_csv(metrics_path)

    avg_metrics["f1_score"] =  2* avg_metrics["precision"]*avg_metrics["recall"]/ \
        (avg_metrics["precision"]+avg_metrics["recall"])
    
    maxf1 = avg_metrics['f1_score'].max()
    optimal_f1_threshold = avg_metrics['f1_score'].idxmax()
    
    logging.info("Highest F1-score of {:.5f}, achieved at threshold {}".format(maxf1, optimal_f1_threshold))
    
    logging.info("Plotting Precision vs Recall")
    np_avg_metrics = avg_metrics.to_numpy().T
    fig = precision_recall_f1iso([np_avg_metrics[0]],[np_avg_metrics[1]],model.name)
    fig_filename = os.path.join(output_folder, 'simple-precision-recall.pdf')
    fig.savefig(fig_filename)
    
    # Report times
    total_inference_time = str(datetime.timedelta(seconds=int(sum(times))))
    average_batch_inference_time = np.mean(times)
    total_evalution_time = str(datetime.timedelta(seconds=int(time.time() - start_total_time )))

    # Logging 
    logger.info("Total evaluation run-time: {}".format(total_evalution_time))
    logger.info("Average batch inference time: {:.5f}s".format(average_batch_inference_time))
    logger.info("Total inference time: {}".format(total_inference_time))


