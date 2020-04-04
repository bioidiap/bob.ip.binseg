.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.eval:

============
 Evaluation
============

To evaluate trained models use our CLI interface. ``bob binseg evaluate``
followed by the model and the dataset configuration, and the path to the
pretrained model via the argument ``--weight``.

Alternatively point to the output folder used during training via the
``--output-path`` argument.   The Checkpointer will load the model as indicated
in the file: ``last_checkpoint``.

Use ``bob binseg evaluate --help`` for more information.

E.g. run inference on model M2U-Net on the DRIVE test set:

.. code-block:: bash

    # Point directly to saved model via -w argument:
    bob binseg evaluate m2unet drive-test -o /outputfolder/for/results -w /direct/path/to/weight/model_final.pth

    # Use training output path (requries last_checkpoint file to be present)
    # The evaluation results will be stored in the same folder
    bob binseg test m2unet drive-test -o /outputfolder/for/results


Outputs
========
The inference run generates the following output files:

.. code-block:: text

    .
    ├── images  # the predicted probabilities as grayscale images in .png format
    ├── hdf5    # the predicted probabilties in hdf5 format
    ├── last_checkpoint  # text file that keeps track of the last checkpoint
    ├── trainlog.csv # training log
    ├── trainlog.pdf # training log plot
    ├── model_*.pth # model checkpoints
    └── results
        ├── image*.jpg.csv # evaluation metrics for each image
        ├── Metrics.csv # average evaluation metrics
        ├── ModelSummary.txt # model summary and parameter count
        ├── precision_recall.pdf # precision vs recall plot
        └── Times.txt # inference times


Inference Only Mode
====================

If you wish to run inference only on a folder containing images, use the
``predict`` function in combination with a
:py:mod:`bob.ip.binseg.configs.datasets.imagefolderinference` config. E.g.:

.. code-block:: bash

    bob binseg predict M2UNet /path/to/myinferencedatasetconfig.py -b 1 -d cpu -o /my/output/path -w /path/to/pretrained/weight/model_final.pth -vv



To run evaluation of pretrained models pass url as ``-w`` argument. E.g.:

.. code-block:: bash

    bob binseg test DRIU DRIVETEST -o Evaluation_DRIU_DRIVE -w https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/DRIU_DRIVE.pth
    bob binseg test M2UNet DRIVETEST -o Evaluation_M2UNet_DRIVE -w https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_DRIVE.pth



.. include:: links.rst
