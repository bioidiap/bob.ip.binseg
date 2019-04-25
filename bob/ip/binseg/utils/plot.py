#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author='Andre Anjos',
# author_email='andre.anjos@idiap.ch',

import numpy as np

def precision_recall_f1iso(precision, recall, names, title=None, human_perf_bsds500=False):
    '''Creates a precision-recall plot of the given data.   
    The plot will be annotated with F1-score iso-lines (in which the F1-score
    maintains the same value)   
    Parameters
    ----------  
      precision : :py:class:`np.ndarray` or :py:class:`list`
        A list of 1D np arrays containing the Y coordinates of the plot, or
        the precision, or a 2D np array in which the rows correspond to each
        of the system's precision coordinates.  
      recall : :py:class:`np.ndarray` or :py:class:`list`
        A list of 1D np arrays containing the X coordinates of the plot, or
        the recall, or a 2D np array in which the rows correspond to each
        of the system's recall coordinates. 
      names : :py:class:`list`
        An iterable over the names of each of the systems along the rows of
        ``precision`` and ``recall``    
      title : :py:class:`str`, optional
        A title for the plot. If not set, omits the title   
      human_perf_bsds500 : :py:class:`bool`, optional
        Whether to display the human performance on the BSDS-500 dataset - it is
        a fixed point on precision=0.897659 and recall=0.700762.    
    Returns
    ------- 
      figure : matplotlib.figure.Figure
        A matplotlib figure you can save or display 
    ''' 
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt 
    fig, ax1 = plt.subplots(1)  
    for p, r, n in zip(precision, recall, names):   
        # Plots only from the point where recall reaches its maximum, otherwise, we
        # don't see a curve...
        i = r.argmax()
        pi = p[i:]
        ri = r[i:]    
        valid = (pi+ri) > 0
        f1 = 2 * (pi[valid]*ri[valid]) / (pi[valid]+ri[valid])    
        # Plot Recall/Precision as threshold changes
        ax1.plot(ri[pi>0], pi[pi>0], label='[F=%.3f] %s' % (f1.max(), n)) 
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
        if human_perf_bsds500:
            plt.plot(0.700762, 0.897659, 'go', markersize=5, label='[F=0.800] Human')
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



def loss_curve(df):
    ''' Creates a loss curve
    Dataframe with column names:
    ["avg. loss", "median loss","lr","max memory"]
    Arguments
    ---------
    df : :py:class.`pandas.DataFrame`
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    ''' 
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt 
    ax1 = df.plot(y="median loss", grid=True)
    ax1.set_ylabel('median loss')
    ax1.grid(linestyle='--', linewidth=1, color='gray', alpha=0.2)
    ax2 = df['lr'].plot(secondary_y=True,legend=True,grid=True,)
    ax2.set_ylabel('lr')
    ax1.set_xlabel('epoch')
    plt.tight_layout()  
    fig = ax1.get_figure()
    return fig