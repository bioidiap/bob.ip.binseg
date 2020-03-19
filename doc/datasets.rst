.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.datasets:

====================
 Supported Datasets
====================

Here is a list of currently support datasets in this package, alongside notable
properties.  Each dataset name is linked to the current location where raw data
can be downloaded.  We include the reference of the data split protocols used
to generate iterators for training and testing.


+-----------------+--------------------+-----------------------+-------------+---------+------+--------+-----+-----+--------------------+-------+------+
|   Dataset       |   Reference        | ``bob.db`` package    |    H x W    | Samples | Mask | Vessel | OD  | Cup | Split Reference    | Train | Test |
+-----------------+--------------------+-----------------------+-------------+---------+------+--------+-----+-----+--------------------+-------+------+
| DRIVE_          | [DRIVE-2004]_      | ``bob.db.drive``      | 584 x 565   | 40      | x    | x      |     |     | [DRIVE-2004]_      | 20    | 20   |
+-----------------+--------------------+-----------------------+-------------+---------+------+--------+-----+-----+--------------------+-------+------+
| STARE_          | [STARE-2000]_      | ``bob.db.stare``      | 605 x 700   | 20      |      | x      |     |     | [MANINIS-2016]_    | 10    | 10   |
+-----------------+--------------------+-----------------------+-------------+---------+------+--------+-----+-----+--------------------+-------+------+
| CHASE-DB1_      | [CHASEDB1-2012]_   | ``bob.db.chasedb``    | 960 x 999   | 28      |      | x      |     |     | [CHASEDB1-2012]_   | 8     | 20   |
+-----------------+--------------------+-----------------------+-------------+---------+------+--------+-----+-----+--------------------+-------+------+
| HRF_            | [HRF-2013]_        | ``bob.db.hrf``        | 2336 x 3504 | 45      | x    | x      |     |     | [ORLANDO-2017]_    | 15    | 30   |
+-----------------+--------------------+-----------------------+-------------+---------+------+--------+-----+-----+--------------------+-------+------+
| IOSTAR_         | [IOSTAR-2016]_     | ``bob.db.iostar``     | 1024 x 1024 | 30      | x    | x      | x   |     | [MEYER-2017]_      | 20    | 10   |
+-----------------+--------------------+-----------------------+-------------+---------+------+--------+-----+-----+--------------------+-------+------+
| DRIONS-DB_      | [DRIONSDB-2008]_   | ``bob.db.drionsdb``   | 400 x 600   | 110     |      |        | x   |     | [MANINIS-2016]_    | 60    | 50   |
+-----------------+--------------------+-----------------------+-------------+---------+------+--------+-----+-----+--------------------+-------+------+
| `RIM-ONE r3`_   | [RIMONER3-2015]_   | ``bob.db.rimoner3``   | 1424 x 1072 | 159     |      |        | x   | x   | [MANINIS-2016]_    | 99    | 60   |
+-----------------+-------------------+------------------------+-------------+---------+------+--------+-----+-----+--------------------+-------+------+
| Drishti-GS1_    | [DRISHTIGS1-2014]_ | ``bob.db.drishtigs1`` | varying     | 101     |      |        | x   | x   | [DRISHTIGS1-2014]_ | 50    | 51   |
+-----------------+--------------------+-----------------------+-------------+---------+------+--------+-----+-----+--------------------+-------+------+
| REFUGE_ (train) | [REFUGE-2018]_     | ``bob.db.refuge``     | 2056 x 2124 | 400     |      |        | x   | x   | [REFUGE-2018]_     | 400   |      |
+-----------------+--------------------+-----------------------+-------------+---------+------+--------+-----+-----+--------------------+-------+------+
| REFUGE_ (val)   | [REFUGE-2018]_     | ``bob.db.refuge``     | 1634 x 1634 | 400     |      |        | x   | x   | [REFUGE-2018]_     |       | 400  |
+-----------------+--------------------+-----------------------+-------------+---------+------+--------+-----+-----+--------------------+-------+------+


Folder-based Dataset
--------------------

For quick experimentation, we also provide a PyTorch_ class that works with the
following dataset folder structure for images and ground-truth (gt):

.. code-block:: text

   root
      |- images
      |- gt


The file names should have the same stem. Currently, all image formats that can
be read via PIL are supported.  Additionally, we also support HDF5 binary
files.

For training, a new dataset configuration needs to be created. You can copy the
template :ref:`bob.ip.binseg.configs.datasets.imagefolder` and amend it
accordingly, e.g. to point to the the full path of the dataset and if necessary
any preprocessing steps such as resizing, cropping, padding etc.

Training can then be started with, e.g.:

.. code-block:: sh

   bob binseg train M2UNet /path/to/myimagefolderconfig.py -b 4 -d cuda -o /my/output/path -vv

Similary for testing, a test dataset config needs to be created. You can copy
the template :ref:`bob.ip.binseg.configs.datasets.imagefoldertest` and amend it
accordingly.

Testing can then be started with, e.g.:

.. code-block:: bash

   bob binseg test M2UNet /path/to/myimagefoldertestconfig.py -b 2 -d cuda -o /my/output/path -vv


.. include:: links.rst
