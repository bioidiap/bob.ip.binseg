.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.eval:

==========================
 Inference and Evaluation
==========================


Inference
---------

You may use one of your trained models (or :ref:`one of ours
<bob.ip.binseg.models>` to run inference on existing datasets or your own
dataset.


Inference on an existing datasets
=================================

To run inference, use the sub-command :ref:`predict
<bob.ip.binseg.cli.predict>` to run prediction on an existing dataset:

.. code-block:: sh

   $ bob binseg predict -vv <model> -w <path/to/model.pth> <dataset>


Replace ``<model>`` and ``<dataset>`` by the appropriate :ref:`configuration
files <bob.ip.binseg.configs>`.  Replace ``<path/to/model.pth>`` to a path
leading to the pre-trained model, or URL pointing to a pre-trained model (e.g.
:ref:`one of ours <bob.ip.binseg.models>`).


Inference on a custom dataset
=============================

If you would like to test your own data against one of the pre-trained models,
you need to instantiate one of:

* :py:mod:`A CSV-based configuration <bob.ip.binseg.configs.datasets.csv>`
* :py:mod:`A folder-based configuration <bob.ip.binseg.configs.datasets.folder>`

Read the appropriate module documentation for details.


.. code-block:: bash

   $ bob binseg config copy folder-dataset-example mydataset.py
   # or
   $ bob binseg config copy csv-dataset-example mydataset.py
   # edit mydataset.py to your liking
   $ bob binseg predict -vv <model> -w <path/to/model.pth> ./mydataset.py


Evaluation
----------

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


To run evaluation of pretrained models pass url as ``-w`` argument. E.g.:

.. code-block:: bash

    bob binseg test DRIU DRIVETEST -o Evaluation_DRIU_DRIVE -w https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/DRIU_DRIVE.pth
    bob binseg test M2UNet DRIVETEST -o Evaluation_M2UNet_DRIVE -w https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_DRIVE.pth



.. include:: links.rst
