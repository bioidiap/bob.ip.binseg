#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import csv
import pandas as pd
import PIL
from PIL import Image,ImageFont, ImageDraw
import torchvision.transforms.functional as VF
import torch

def precision_recall_f1iso(precision, recall, names, title=None):
    """
    Author: Andre Anjos (andre.anjos@idiap.ch).

    Creates a precision-recall plot of the given data.
    The plot will be annotated with F1-score iso-lines (in which the F1-score
    maintains the same value)

    Parameters
    ----------
    precision : :py:class:`numpy.ndarray` or :py:class:`list`
        A list of 1D np arrays containing the Y coordinates of the plot, or
        the precision, or a 2D np array in which the rows correspond to each
        of the system's precision coordinates.
    recall : :py:class:`numpy.ndarray` or :py:class:`list`
        A list of 1D np arrays containing the X coordinates of the plot, or
        the recall, or a 2D np array in which the rows correspond to each
        of the system's recall coordinates.
    names : :py:class:`list`
        An iterable over the names of each of the systems along the rows of
        ``precision`` and ``recall``
    title : :py:class:`str`, optional
        A title for the plot. If not set, omits the title

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure you can save or display
    """
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from itertools import cycle
    fig, ax1 = plt.subplots(1)
    lines = ["-","--","-.",":"]
    linecycler = cycle(lines)
    for p, r, n in zip(precision, recall, names):
        # Plots only from the point where recall reaches its maximum, otherwise, we
        # don't see a curve...
        i = r.argmax()
        pi = p[i:]
        ri = r[i:]
        valid = (pi+ri) > 0
        f1 = 2 * (pi[valid]*ri[valid]) / (pi[valid]+ri[valid])
        # optimal point along the curve
        argmax = f1.argmax()
        opi = pi[argmax]
        ori = ri[argmax]
        # Plot Recall/Precision as threshold changes
        ax1.plot(ri[pi>0], pi[pi>0], next(linecycler), label='[F={:.4f}] {}'.format(f1.max(), n),)
        ax1.plot(ori,opi, marker='o', linestyle=None, markersize=3, color='black')
    ax1.grid(linestyle='--', linewidth=1, color='gray', alpha=0.2)
    if len(names) > 1:
        plt.legend(loc='lower left', framealpha=0.5)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    if title is not None: ax1.set_title(title)
    # Annotates plot with F1-score iso-lines
    ax2 = ax1.twinx()
    f_scores = np.linspace(0.1, 0.9, num=9)
    tick_locs = []
    tick_labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='green', alpha=0.1)
        tick_locs.append(y[-1])
        tick_labels.append('%.1f' % f_score)
    ax2.tick_params(axis='y', which='both', pad=0, right=False, left=False)
    ax2.set_ylabel('iso-F', color='green', alpha=0.3)
    ax2.set_ylim([0.0, 1.0])
    ax2.yaxis.set_label_coords(1.015, 0.97)
    ax2.set_yticks(tick_locs) #notice these are invisible
    for k in ax2.set_yticklabels(tick_labels):
        k.set_color('green')
        k.set_alpha(0.3)
        k.set_size(8)
    # we should see some of axes 1 axes
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_position(('data', -0.015))
    ax1.spines['bottom'].set_position(('data', -0.015))
    # we shouldn't see any of axes 2 axes
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    plt.tight_layout()
    return fig

def precision_recall_f1iso_confintval(precision, recall, pr_upper, pr_lower, re_upper, re_lower, names, title=None):
    """
    Author: Andre Anjos (andre.anjos@idiap.ch).

    Creates a precision-recall plot of the given data.
    The plot will be annotated with F1-score iso-lines (in which the F1-score
    maintains the same value)

    Parameters
    ----------
    precision : :py:class:`numpy.ndarray` or :py:class:`list`
        A list of 1D np arrays containing the Y coordinates of the plot, or
        the precision, or a 2D np array in which the rows correspond to each
        of the system's precision coordinates.
    recall : :py:class:`numpy.ndarray` or :py:class:`list`
        A list of 1D np arrays containing the X coordinates of the plot, or
        the recall, or a 2D np array in which the rows correspond to each
        of the system's recall coordinates.
    names : :py:class:`list`
        An iterable over the names of each of the systems along the rows of
        ``precision`` and ``recall``
    title : :py:class:`str`, optional
        A title for the plot. If not set, omits the title

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure you can save or display
    """
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from itertools import cycle
    fig, ax1 = plt.subplots(1)
    lines = ["-","--","-.",":"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    colorcycler = cycle(colors)
    linecycler = cycle(lines)
    for p, r, pu, pl, ru, rl, n in zip(precision, recall, pr_upper, pr_lower, re_upper, re_lower, names):
        # Plots only from the point where recall reaches its maximum, otherwise, we
        # don't see a curve...
        i = r.argmax()
        pi = p[i:]
        ri = r[i:]
        pui = pu[i:]
        pli = pl[i:]
        rui = ru[i:]
        rli = rl[i:]
        valid = (pi+ri) > 0
        f1 = 2 * (pi[valid]*ri[valid]) / (pi[valid]+ri[valid])
        # optimal point along the curve
        argmax = f1.argmax()
        opi = pi[argmax]
        ori = ri[argmax]
        # Plot Recall/Precision as threshold changes
        ax1.plot(ri[pi>0], pi[pi>0], next(linecycler), label='[F={:.4f}] {}'.format(f1.max(), n),)
        ax1.plot(ori,opi, marker='o', linestyle=None, markersize=3, color='black')
        # Plot confidence
        # Upper bound
        #ax1.plot(r95ui[p95ui>0], p95ui[p95ui>0])
        # Lower bound
        #ax1.plot(r95li[p95li>0], p95li[p95li>0])
        # create the limiting polygon
        vert_x = np.concatenate((rui[pui>0], rli[pli>0][::-1]))
        vert_y = np.concatenate((pui[pui>0], pli[pli>0][::-1]))
        # hacky workaround to plot 2nd human
        if np.isclose(np.mean(rui), rui[1], rtol=1e-05):
            print('found human')
            p = plt.Polygon(np.column_stack((vert_x, vert_y)), facecolor='none', alpha=.2, edgecolor=next(colorcycler),lw=2)
        else:
            p = plt.Polygon(np.column_stack((vert_x, vert_y)), facecolor=next(colorcycler), alpha=.2, edgecolor='none',lw=.2)
        ax1.add_artist(p)

    ax1.grid(linestyle='--', linewidth=1, color='gray', alpha=0.2)
    if len(names) > 1:
        plt.legend(loc='lower left', framealpha=0.5)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    if title is not None: ax1.set_title(title)
    # Annotates plot with F1-score iso-lines
    ax2 = ax1.twinx()
    f_scores = np.linspace(0.1, 0.9, num=9)
    tick_locs = []
    tick_labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='green', alpha=0.1)
        tick_locs.append(y[-1])
        tick_labels.append('%.1f' % f_score)
    ax2.tick_params(axis='y', which='both', pad=0, right=False, left=False)
    ax2.set_ylabel('iso-F', color='green', alpha=0.3)
    ax2.set_ylim([0.0, 1.0])
    ax2.yaxis.set_label_coords(1.015, 0.97)
    ax2.set_yticks(tick_locs) #notice these are invisible
    for k in ax2.set_yticklabels(tick_labels):
        k.set_color('green')
        k.set_alpha(0.3)
        k.set_size(8)
    # we should see some of axes 1 axes
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_position(('data', -0.015))
    ax1.spines['bottom'].set_position(('data', -0.015))
    # we shouldn't see any of axes 2 axes
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    plt.tight_layout()
    return fig

def loss_curve(df, title):
    """ Creates a loss curve given a Dataframe with column names:

    ``['avg. loss', 'median loss','lr','max memory']``

    Parameters
    ----------
    df : :py:class:`pandas.DataFrame`

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    ax1 = df.plot(y="median loss", grid=True)
    ax1.set_title(title)
    ax1.set_ylabel('median loss')
    ax1.grid(linestyle='--', linewidth=1, color='gray', alpha=0.2)
    ax2 = df['lr'].plot(secondary_y=True,legend=True,grid=True,)
    ax2.set_ylabel('lr')
    ax1.set_xlabel('epoch')
    plt.tight_layout()
    fig = ax1.get_figure()
    return fig


def read_metricscsv(file):
    """
    Read precision and recall from csv file

    Parameters
    ----------
    file : str
        path to file

    Returns
    -------
    :py:class:`numpy.ndarray`
    :py:class:`numpy.ndarray`
    """
    with open (file, "r") as infile:
        metricsreader = csv.reader(infile)
        # skip header row
        next(metricsreader)
        precision = []
        recall = []
        pr_upper = []
        pr_lower = []
        re_upper = []
        re_lower = []
        for row in metricsreader:
            precision.append(float(row[1]))
            recall.append(float(row[2]))
            pr_upper.append(float(row[8]))
            pr_lower.append(float(row[9]))
            re_upper.append(float(row[11]))
            re_lower.append(float(row[12]))
    return np.array(precision), np.array(recall), np.array(pr_upper), np.array(pr_lower), np.array(re_upper), np.array(re_lower)


def plot_overview(outputfolders,title):
    """
    Plots comparison chart of all trained models

    Parameters
    ----------
    outputfolder : list
        list containing output paths of all evaluated models (e.g. ``['DRIVE/model1', 'DRIVE/model2']``)
    title : str
        title of plot
    Returns
    -------
    matplotlib.figure.Figure
    """
    precisions = []
    recalls = []
    pr_ups = []
    pr_lows = []
    re_ups = []
    re_lows = []
    names = []
    params = []
    for folder in outputfolders:
        # metrics
        metrics_path = os.path.join(folder,'results/Metrics.csv')
        pr, re, pr_upper, pr_lower, re_upper, re_lower = read_metricscsv(metrics_path)
        precisions.append(pr)
        recalls.append(re)
        pr_ups.append(pr_upper)
        pr_lows.append(pr_lower)
        re_ups.append(re_upper)
        re_lows.append(re_lower)
        modelname = folder.split('/')[-1]
        datasetname =  folder.split('/')[-2]
        # parameters
        summary_path = os.path.join(folder,'results/ModelSummary.txt')
        with open (summary_path, "r") as outfile:
          rows = outfile.readlines()
          lastrow = rows[-1]
          parameter = int(lastrow.split()[1].replace(',',''))
        #name = '[P={:.2f}M] {} {}'.format(parameter/100**3, modelname, "")
        name = '{} '.format(modelname)
        names.append(name)
    #title = folder.split('/')[-4]
    fig = precision_recall_f1iso_confintval(precisions,recalls, pr_ups, pr_lows, re_ups, re_lows, names,title)
    return fig

def metricsviz(dataset
                ,output_path
                ,tp_color= (0,255,0) # (128,128,128) Gray
                ,fp_color = (0, 0, 255) # (70, 240, 240) Cyan
                ,fn_color = (255, 0, 0) # (245, 130, 48) Orange
                ,overlayed=True):
    """ Visualizes true positives, false positives and false negatives
    Default colors TP: Gray, FP: Cyan, FN: Orange

    Parameters
    ----------
    dataset : :py:class:`torch.utils.data.Dataset`
    output_path : str
        path where results and probability output images are stored. E.g. ``'DRIVE/MODEL'``
    tp_color : tuple
        RGB values, by default (128,128,128)
    fp_color : tuple
        RGB values, by default (70, 240, 240)
    fn_color : tuple
        RGB values, by default (245, 130, 48)
    """

    for sample in dataset:
        # get sample
        name  = sample[0]
        img = VF.to_pil_image(sample[1]) # PIL Image
        gt = sample[2].byte() # byte tensor

        # read metrics
        metrics = pd.read_csv(os.path.join(output_path,'results','Metrics.csv'))
        optimal_threshold = metrics['threshold'][metrics['f1_score'].idxmax()]

        # read probability output
        pred = Image.open(os.path.join(output_path,'images',name))
        pred = pred.convert(mode='L')
        pred = VF.to_tensor(pred)
        binary_pred = torch.gt(pred, optimal_threshold).byte()

        # calc metrics
        # equals and not-equals
        equals = torch.eq(binary_pred, gt) # tensor
        notequals = torch.ne(binary_pred, gt) # tensor
        # true positives
        tp_tensor = (gt * binary_pred ) # tensor
        tp_pil = VF.to_pil_image(tp_tensor.float())
        tp_pil_colored = PIL.ImageOps.colorize(tp_pil, (0,0,0), tp_color)
        # false positives
        fp_tensor = torch.eq((binary_pred + tp_tensor), 1)
        fp_pil = VF.to_pil_image(fp_tensor.float())
        fp_pil_colored = PIL.ImageOps.colorize(fp_pil, (0,0,0), fp_color)
        # false negatives
        fn_tensor = notequals - fp_tensor
        fn_pil = VF.to_pil_image(fn_tensor.float())
        fn_pil_colored = PIL.ImageOps.colorize(fn_pil, (0,0,0), fn_color)

        # paste together
        tp_pil_colored.paste(fp_pil_colored,mask=fp_pil)
        tp_pil_colored.paste(fn_pil_colored,mask=fn_pil)

        if overlayed:
            tp_pil_colored = PIL.Image.blend(img, tp_pil_colored, 0.4)
            img_metrics = pd.read_csv(os.path.join(output_path,'results',name+'.csv'))
            f1 = img_metrics[' f1_score'].max()
            # add f1-score
            fnt_size = tp_pil_colored.size[1]//25
            draw = ImageDraw.Draw(tp_pil_colored)
            fnt = ImageFont.truetype('FreeMono.ttf', fnt_size)
            draw.text((0, 0),"F1: {:.4f}".format(f1),(255,255,255),font=fnt)

        # save to disk
        overlayed_path = os.path.join(output_path,'tpfnfpviz')
        fullpath = os.path.join(overlayed_path, name)
        fulldir = os.path.dirname(fullpath)
        if not os.path.exists(fulldir): os.makedirs(fulldir)
        tp_pil_colored.save(fullpath)


def overlay(dataset, output_path):
    """Overlays prediction probabilities vessel tree with original test image.

    Parameters
    ----------
    dataset : :py:class:`torch.utils.data.Dataset`
    output_path : str
        path where results and probability output images are stored. E.g. ``'DRIVE/MODEL'``
    """

    for sample in dataset:
        # get sample
        name  = sample[0]
        img = VF.to_pil_image(sample[1]) # PIL Image

        # read probability output
        pred = Image.open(os.path.join(output_path,'images',name)).convert(mode='L')
        # color and overlay
        pred_green = PIL.ImageOps.colorize(pred, (0,0,0), (0,255,0))
        overlayed = PIL.Image.blend(img, pred_green, 0.4)

        # add f1-score
        #fnt_size = overlayed.size[1]//25
        #draw = ImageDraw.Draw(overlayed)
        #fnt = ImageFont.truetype('FreeMono.ttf', fnt_size)
        #draw.text((0, 0),"F1: {:.4f}".format(f1),(255,255,255),font=fnt)
        # save to disk
        overlayed_path = os.path.join(output_path,'overlayed')
        fullpath = os.path.join(overlayed_path, name)
        fulldir = os.path.dirname(fullpath)
        if not os.path.exists(fulldir): os.makedirs(fulldir)
        overlayed.save(fullpath)


def savetransformedtest(dataset, output_path):
    """Save the test images as they are fed into the neural network.
    Makes it easier to create overlay animations (e.g. slide)

    Parameters
    ----------
    dataset : :py:class:`torch.utils.data.Dataset`
    output_path : str
        path where results and probability output images are stored. E.g. ``'DRIVE/MODEL'``
    """

    for sample in dataset:
        # get sample
        name  = sample[0]
        img = VF.to_pil_image(sample[1]) # PIL Image

        # save to disk
        testimg_path = os.path.join(output_path,'transformedtestimages')
        fullpath = os.path.join(testimg_path, name)
        fulldir = os.path.dirname(fullpath)
        if not os.path.exists(fulldir): os.makedirs(fulldir)
        img.save(fullpath)
