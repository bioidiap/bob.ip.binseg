.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.training:

==========
 Training
==========

To train a new FCN, use the command-line interface (CLI) application ``bob
binseg train``, available on your prompt.  To use this CLI, you must define
the input dataset that will be used to train the FCN, as well as the type of
model that will be trained.  You may issue ``bob binseg train --help`` for a
help message containing more detailed instructions.

To replicate our results, use our main application ``bob binseg train``
followed by the model configuration, and dataset configuration files, and/or
command-line options.  Use ``bob binseg train --help`` for more information.

.. tip::

   We strongly advice training with a GPU (using ``--device="cuda:0"``).
   Depending on the available GPU memory you might have to adjust your batch
   size (``--batch``).


Baseline Benchmarks
===================

The following table describes recommended batch sizes for 24Gb of RAM GPU
card, for supervised training of baselines.  Use it like this:

.. code-block:: sh

   # change <model> and <dataset> by one of items bellow
   $ bob binseg train -vv <model> <dataset> --batch-size=<see-table> --device="cuda:0"
   # check results in the "results" folder

.. list-table::

  * - **Models / Datasets**
    - :py:mod:`drive <bob.ip.binseg.configs.datasets.drive>`
    - :py:mod:`stare <bob.ip.binseg.configs.datasets.stare>`
    - :py:mod:`chasedb1 <bob.ip.binseg.configs.datasets.chasedb1>`
    - :py:mod:`iostar-vessel <bob.ip.binseg.configs.datasets.iostarvessel>`
    - :py:mod:`hrf <bob.ip.binseg.configs.datasets.hrf1168>`
  * - :py:mod:`unet <bob.ip.binseg.configs.models.unet>`
    - 4
    - 2
    - 2
    - 2
    - 1
  * - :py:mod:`hed <bob.ip.binseg.configs.models.hed>`
    - 8
    - 4
    - 4
    - 4
    - 1
  * - :py:mod:`driu <bob.ip.binseg.configs.models.driu>` / :py:mod:`driu-bn <bob.ip.binseg.configs.models.driubn>`
    - 8
    - 5
    - 4
    - 4
    - 1
  * - :py:mod:`m2unet <bob.ip.binseg.configs.models.m2unet>`
    - 16
    - 6
    - 6
    - 6
    - 1


.. tip::

   Instead of the default configurations, you can pass the full path of your
   customized dataset and model files.  You may :ref:`copy any of the existing
   configuration resources <bob.ip.binseg.cli.config.copy>` and change them
   locally.  Once you're happy, you may use the newly created files directly on
   your training command line.  For example, suppose you wanted to slightly
   change the drive pre-processing pipeline.  You could do the following:

   .. code-block:: bash

      $ bob binseg config copy drive my_drive_remix.py
      # edit my_drive_remix.py to your needs
      $ bob binseg train -vv <model> ./my_drive_remix.py --batch-size=<see-table> --device="cuda:0"


.. _bob.ip.binseg.gridtk-tip:

.. tip::

   If you are at Idiap, you may install the package ``gridtk`` (``conda install
   gridtk``) on your environment, and submit the job like this:

   .. code-block:: sh

      $ jman submit --queue=gpu --memory=24G --name=m2unet-drive -- bob binseg train --device='cuda:0' ... #paste the rest of the command-line


Combined Vessel Dataset (COVD)
==============================

The following table describes recommended batch sizes for 24Gb of RAM GPU
card, for supervised training of COVD- systems.  Use it like this:

.. code-block:: sh

   # change <model> and <dataset> by one of items bellow
   $ bob binseg train -vv <model> <dataset> --batch-size=<see-table> --device="cuda:0"

.. list-table::

  * - **Models / Datasets**
    - :py:mod:`covd-drive <bob.ip.binseg.configs.datasets.starechasedb1iostarhrf544>`
    - :py:mod:`covd-stare <bob.ip.binseg.configs.datasets.drivechasedb1iostarhrf608>`
    - :py:mod:`covd-chasedb1 <bob.ip.binseg.configs.datasets.drivestareiostarhrf960>`
    - :py:mod:`covd-iostar-vessel <bob.ip.binseg.configs.datasets.drivestarechasedb1hrf1024>`
    - :py:mod:`covd-hrf <bob.ip.binseg.configs.datasets.drivestarechasedb1iostar1168>`
  * - :py:mod:`driu <bob.ip.binseg.configs.models.driu>` / :py:mod:`driu-bn <bob.ip.binseg.configs.models.driubn>`
    - 4
    - 4
    - 2
    - 2
    - 2
  * - :py:mod:`m2unet <bob.ip.binseg.configs.models.m2unet>`
    - 8
    - 4
    - 4
    - 4
    - 4


Combined Vessel Dataset (COVD) and Semi-Supervised Learning (SSL)
=================================================================

The following table describes recommended batch sizes for 24Gb of RAM GPU
card, for semi-supervised learning of COVD- systems.  Use it like this:

.. code-block:: sh

   # change <model> and <dataset> by one of items bellow
   $ bob binseg train -vv --ssl <model> <dataset> --batch-size=<see-table> --device="cuda:0"

.. list-table::

  * - **Models / Datasets**
    - :py:mod:`covd-drive-ssl <bob.ip.binseg.configs.datasets.starechasedb1iostarhrf544ssldrive>`
    - :py:mod:`covd-stare-ssl <bob.ip.binseg.configs.datasets.drivechasedb1iostarhrf608sslstare>`
    - :py:mod:`covd-chasedb1-ssl <bob.ip.binseg.configs.datasets.drivestareiostarhrf960sslchase>`
    - :py:mod:`covd-iostar-vessel-ssl <bob.ip.binseg.configs.datasets.drivestarechasedb1hrf1024ssliostar>`
    - :py:mod:`covd-hrf-ssl <bob.ip.binseg.configs.datasets.drivestarechasedb1iostar1168sslhrf>`
  * - :py:mod:`driu-ssl <bob.ip.binseg.configs.models.driussl>` / :py:mod:`driu-bn-ssl <bob.ip.binseg.configs.models.driubnssl>`
    - 4
    - 4
    - 2
    - 1
    - 1
  * - :py:mod:`m2unet-ssl <bob.ip.binseg.configs.models.m2unetssl>`
    - 4
    - 4
    - 2
    - 2
    - 2


Using your own dataset
======================

To use your own dataset, we recommend you read our instructions at
:py:mod:`bob.ip.binseg.configs.datasets.csv`, and setup a CSV file describing
input data and ground-truth (segmentation maps).  Then, prepare a configuration
file by copying our configuration example and edit it to apply the required
transforms to your input data.  Once you are happy with the result, use it in
place of one of our datasets:

.. code-block:: sh

   $ bob binseg config copy csv-dataset-example mydataset.py
   # edit mydataset following instructions
   $ bob binseg train ... mydataset.py ...
