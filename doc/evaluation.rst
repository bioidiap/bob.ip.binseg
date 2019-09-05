.. -*- coding: utf-8 -*-
.. _bob.ip.binseg.evaluation:

==========
Evaluation
==========

To evaluate trained models use use ``bob binseg test`` followed by
the model config, the dataset config and the path to the pretrained
model via the argument ``-w``.

Alternatively point to the output folder used during training via
the ``-o`` argument. The Checkpointer will load the model as indicated
in the file: ``last_checkpoint``.

Use ``bob binseg test --help`` for more information.

E.g. run inference on model M2U-Net on the DRIVE test set:

.. code-block:: bash

    # Point directly to saved model via -w argument:
    bob binseg test M2UNet DRIVETEST -o /outputfolder/for/results -w /direct/path/to/weight/model_final.pth

    # Use training output path (requries last_checkpoint file to be present)
    # The evaluation results will be stored in the same folder
    bob binseg test M2UNet DRIVETEST -o /DRIVE/M2UNet/output

Outputs
========
The inference run generates the following output files:

.. code-block:: bash

    .
    ├── images  # the predicted probabilities as grayscale images in .png format 
    ├── hdf5    # the predicted probabilties in hdf5 format
    ├── last_checkpoint  # text file that keeps track of the last checkpoint 
    ├── M2UNet_trainlog.csv # training log 
    ├── M2UNet_trainlog.pdf # training log plot
    ├── model_*.pth # model checkpoints
    └── results
        ├── image*.jpg.csv # evaluation metrics for each image
        ├── Metrics.csv # average evaluation metrics
        ├── ModelSummary.txt # model summary and parameter count
        ├── precision_recall.pdf # precision vs recall plot
        └── Times.txt # inference times

Inference Only Mode
====================

If you wish to run inference only on a folder containing images, use the ``predict`` function in combination with a :ref:`bob.ip.binseg.configs.datasets.imagefolderinference` config. E.g.:

.. code-block:: bash

    bob binseg predict M2UNet /path/to/myinferencedatasetconfig.py -b 1 -d cpu -o /my/output/path -w /path/to/pretrained/weight/model_final.pth -vv

Pretrained Models
=================

Due to storage limitations we only provide weights of a subset
of all evaluated models:



+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+
|                    | DRIU               | M2UNet                                                                                                                         |
+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+
| DRIVE              | `DRIU_DRIVE.pth`_  | `M2UNet_DRIVE.pth <m2unet_drive.pth_>`_                                                                                        |
+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+
| COVD-DRIVE         |                    | `M2UNet_COVD-DRIVE.pth <https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_COVD-DRIVE.pth>`_               |
+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+
| COVD-DRIVE SSL     |                    | `M2UNet_COVD-DRIVE_SSL.pth <https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_COVD-DRIVE_SSL.pth>`_       |
+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+
| STARE              | DRIU_STARE.pth_    | `M2UNet_STARE.pth <https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_STARE.pth>`_                         |
+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+
| COVD-STARE         |                    | `M2UNet_COVD-STARE.pth <https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_COVD-STARE.pth>`_               |
+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+
| COVD-STARE SSL     |                    | `M2UNet_COVD-STARE_SSL.pth <https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_COVD-STARE_SSL.pth>`_       |
+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+
| CHASE_DB1          | DRIU_CHASEDB1.pth_ | `M2UNet_CHASEDB1.pth <https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_CHASEDB1.pth>`_                   |
+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+
| COVD-CHASE_DB1     |                    | `M2UNet_COVD-CHASEDB1.pth <https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_COVD-CHASEDB1.pth>`_         |
+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+
| COVD-CHASE_DB1 SSL |                    | `M2UNet_COVD-CHASEDB1_SSL.pth <https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_COVD-CHASEDB1_SSL.pth>`_ |
+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+
| IOSTARVESSEL       | DRIU_IOSTAR.pth_   | `M2UNet_IOSTARVESSEL.pth <https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_IOSTARVESSEL.pth>`_           |
+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+
| COVD-IOSTAR        |                    | `M2UNet_COVD-IOSTAR.pth <https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_COVD-IOSTAR.pth>`_             |
+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+
| COVD-IOSTAR SSL    |                    | `M2UNet_COVD-IOSTAR_SSL.pth <https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_COVD-IOSTAR_SSL.pth>`_     |
+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+
| HRF                | DRIU_HRF.pth_      | `M2UNet_HRF1168.pth <https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_HRF1168.pth>`_                     |
+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+
| COVD-HRF           |                    | `M2UNet_COVD-HRF.pth <https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_COVD-HRF.pth>`_                   |
+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+
| COVD-HRF SSL       |                    | `M2UNet_COVD-HRF_SSL.pth <https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_COVD-HRF_SSL.pth>`_           |
+--------------------+--------------------+--------------------------------------------------------------------------------------------------------------------------------+



To run evaluation of pretrained models pass url as ``-w`` argument. E.g.:

.. code-block:: bash

    bob binseg test DRIU DRIVETEST -o Evaluation_DRIU_DRIVE -w https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/DRIU_DRIVE.pth
    bob binseg test M2UNet DRIVETEST -o Evaluation_M2UNet_DRIVE -w https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/M2UNet_DRIVE.pth



.. include:: links.rst
