.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.experiment:

==============================
 Running complete experiments
==============================

We provide an :ref:`aggregator command called "experiment"
<bob.ip.binseg.cli.experiment>` that runs training, followed by prediction,
evaluation and comparison.  After running, you
will be able to find results from model fitting, prediction, evaluation and
comparison under a single output directory.

For example, to train a Mobile V2 U-Net architecture on the STARE dataset,
evaluate both train and test set performances, output prediction maps and
overlay analysis, together with a performance curve, run the following:

.. code-block:: sh

   $ bob binseg experiment -vv m2unet stare --batch-size=16 --overlayed
   # check results in the "results" folder
