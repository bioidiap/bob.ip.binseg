.. -*- coding: utf-8 -*-
.. _bob.ip.binseg.datasets:

==================
Supported Datasets
==================

+-----+---------------+-------------+--------+-------+------+------+--------+-----+-----+----------------------------+
|  #  |     Name      |    H x W    | # imgs | Train | Test | Mask | Vessel | OD  | Cup | Train-Test split reference |
+=====+===============+=============+========+=======+======+======+========+=====+=====+============================+
| 1   | Drive_        | 584 x 565   | 40     | 20    | 20   | x    | x      |     |     | `Staal et al. (2004)`_     |
+-----+---------------+-------------+--------+-------+------+------+--------+-----+-----+----------------------------+
| 2   | STARE_        | 605 x 700   | 20     | 10    | 10   |      | x      |     |     | `Maninis et al. (2016)`_   |
+-----+---------------+-------------+--------+-------+------+------+--------+-----+-----+----------------------------+
| 3   | CHASEDB1_     | 960 x 999   | 28     | 8     | 20   |      | x      |     |     | `Fraz et al. (2012)`_      |
+-----+---------------+-------------+--------+-------+------+------+--------+-----+-----+----------------------------+
| 4   | HRF_          | 2336 x 3504 | 45     | 15    | 30   | x    | x      |     |     | `Orlando et al. (2016)`_   |
+-----+---------------+-------------+--------+-------+------+------+--------+-----+-----+----------------------------+
| 5   | IOSTAR_       | 1024 x 1024 | 30     | 20    | 10   | x    | x      | x   |     | `Meyer et al. (2017)`_     |
+-----+---------------+-------------+--------+-------+------+------+--------+-----+-----+----------------------------+
| 6   | DRIONS-DB_    | 400 x 600   | 110    | 60    | 50   |      |        | x   |     | `Maninis et al. (2016)`_   |
+-----+---------------+-------------+--------+-------+------+------+--------+-----+-----+----------------------------+
| 7   | RIM-ONEr3_    | 1424 x 1072 | 159    | 99    | 60   |      |        | x   | x   | `Maninis et al. (2016)`_   |
+-----+---------------+-------------+--------+-------+------+------+--------+-----+-----+----------------------------+
| 8   | Drishti-GS1_  | varying     | 101    | 50    | 51   |      |        | x   | x   | `Sivaswamy et al. (2014)`_ |
+-----+---------------+-------------+--------+-------+------+------+--------+-----+-----+----------------------------+
| 9   | REFUGE_ train | 2056 x 2124 | 400    | 400   |      |      |        | x   | x   | REFUGE_                    |
+-----+---------------+-------------+--------+-------+------+------+--------+-----+-----+----------------------------+
| 9   | REFUGE_ val   | 1634 x 1634 | 400    |       | 400  |      |        | x   | x   | REFUGE_                    |
+-----+---------------+-------------+--------+-------+------+------+--------+-----+-----+----------------------------+


Add-on: Folder-based Dataset
============================

For quick experimentation we also provide a PyTorch class that works with the following 
dataset folder structure for images and ground-truth (gt):

.. code-block:: bash

    root
       |- images
       |- gt 

the file names should have the same stem. Currently all image formats that can be read via PIL are supported. Additionally we support hdf5 binary files.

For training a new dataset config needs to be created. You can copy the template :ref:`bob.ip.binseg.configs.datasets.imagefolder` and amend accordingly, 
e.g. the full path of the dataset and if necessary any preprocessing steps such as resizing, cropping, padding etc..

Training can then be started with

.. code-block:: bash

    bob binseg train M2UNet /path/to/myimagefolderconfig.py -b 4 -d cuda -o /my/output/path -vv

Similary for testing, a test dataset config needs to be created. You can copy the template :ref:`bob.ip.binseg.configs.datasets.imagefoldertest` and amend accordingly.

Testing can then be started with 

.. code-block:: bash

    bob binseg test M2UNet /path/to/myimagefoldertestconfig.py -b 2 -d cuda -o /my/output/path -vv

.. include:: links.rst
