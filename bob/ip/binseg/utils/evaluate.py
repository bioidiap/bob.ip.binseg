#!/usr/bin/env python
# -*- coding: utf-8 -*-
# only use to evaluate 2nd human annotator
#  
import os 
import logging
import time
import datetime
import numpy as np
import torch
import pandas as pd
import torchvision.transforms.functional as VF
from tqdm import tqdm

from bob.ip.binseg.utils.metric import SmoothedValue, base_metrics
from bob.ip.binseg.utils.plot import precision_recall_f1iso, precision_recall_f1iso_confintval
from bob.ip.binseg.utils.summary import summary
from PIL import Image
from torchvision.transforms.functional import to_tensor


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



def do_eval(
    prediction_folder,
    data_loader,
    output_folder = None,
    title = '2nd human',
    legend = '2nd human'
):

    """
    Calculate metrics on saved prediction images (needs batch_size = 1 !)
    
    Parameters
    ---------
    model : :py:class:`torch.nn.Module`
        neural network model (e.g. DRIU, HED, UNet)
    data_loader : py:class:`torch.torch.utils.data.DataLoader`
    device : str
        device to use ``'cpu'`` or ``'cuda'``
    output_folder : str
    """
    logger = logging.getLogger("bob.ip.binseg.engine.evaluate")
    logger.info("Start evaluation")
    logger.info("Prediction folder {}".format(prediction_folder))
    results_subfolder = os.path.join(output_folder,'results') 
    os.makedirs(results_subfolder,exist_ok=True)
    
    
    # Collect overall metrics 
    metrics = []
    num_images = len(data_loader)
    for samples in tqdm(data_loader):
        names = samples[0]
        images = samples[1]
        ground_truths = samples[2]
      
    
        pred_file = os.path.join(prediction_folder,names[0])
        probabilities = Image.open(pred_file)    
        probabilities = probabilities.convert(mode='L')
        probabilities = to_tensor(probabilities)

            
        b_metrics = batch_metrics(probabilities, ground_truths, names,results_subfolder, logger)
        metrics.extend(b_metrics)
            


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
    metrics_file = "Metrics.csv"
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
    #print(avg_metrics)
    np_avg_metrics = avg_metrics.to_numpy().T
    fig_name = "precision_recall.pdf"
    logger.info("saving {}".format(fig_name))
    fig = precision_recall_f1iso_confintval([np_avg_metrics[0]],[np_avg_metrics[1]],[np_avg_metrics[7]],[np_avg_metrics[8]],[np_avg_metrics[10]],[np_avg_metrics[11]], [legend ,None], title=title)
    fig_filename = os.path.join(results_subfolder, fig_name)
    fig.savefig(fig_filename)



