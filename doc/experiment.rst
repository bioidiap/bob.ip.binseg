.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.experiment:

==============================
 Running complete experiments
==============================

We provide an :ref:`aggregator command called "experiment"
<bob.ip.binseg.cli.experiment>` that runs training, followed by prediction,
evaluation and comparison.  After running, you will be able to find results
from model fitting, prediction, evaluation and comparison under a single output
directory.

For example, to train a Mobile V2 U-Net architecture on the STARE dataset
(optic vessel segmentation), evaluate both train and test set performances,
output prediction maps and overlay analysis, together with a performance curve,
run the following:

.. code-block:: sh

   $ bob binseg experiment -vv m2unet stare --batch-size=16 --overlayed
   # check results in the "results" folder

You may run the system on a GPU by using the ``--device=cuda:0`` option.


Using your own dataset
======================

To use your own dataset, we recommend you read our instructions at
:py:mod:`bob.ip.binseg.configs.datasets.csv`, and setup one or more CSV file
describing input data and ground-truth (segmentation maps), and potential test
data.  Then, prepare a configuration file by copying our configuration example
and edit it to apply the required transforms to your input data.  Once you are
happy with the result, use it in place of one of our datasets:

.. code-block:: sh

   $ bob binseg config copy csv-dataset-example mydataset.py
   # edit mydataset following instructions
   $ bob binseg experiment ... mydataset.py ...


Changing defaults
=================

We provide a large set of preset configurations to build models from known
datasets.  You can :ref:`copy any of the existing configuration resources
<bob.ip.binseg.cli.config.copy>` and edit to build your own customized version.
Once you're happy, you may use the newly created files directly on your command
line.  For example, suppose you wanted to slightly change the DRIVE
pre-processing pipeline.  You could do the following:

   .. code-block:: bash

      $ bob binseg config copy drive my_drive_remix.py
      # edit my_drive_remix.py to your needs
      $ bob binseg train -vv <model> ./my_drive_remix.py


.. _bob.ip.binseg.gridtk-tip:

Running at Idiap's SGE grid
===========================

If you are at Idiap, you may install the package ``gridtk`` (``conda install
gridtk``) on your environment, and submit the job like this:

.. code-block:: sh

   $ jman submit --queue=gpu --memory=24G --name=myjob -- bob binseg train --device='cuda:0' ... #paste the rest of the command-line

:download:`This bash-script function <scripts/functions.sh>` can be of help
when switching between local and SGE-based running.  Just copy and source this
file, then call the function ``run`` as many times as required to benchmark
your task.

  .. literalinclude:: scripts/functions.sh
     :language: bash
