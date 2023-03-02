.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.install:

==============
 Installation
==============

We support two installation modes, through pip_, or mamba_ (conda).


.. tab:: pip

   Stable, from PyPI:

   .. code:: sh

      pip install deepdraw

   Latest beta, from GitLab package registry:

   .. code:: sh

      pip install --pre --index-url https://gitlab.idiap.ch/api/v4/groups/software/-/packages/pypi/simple --extra-index-url https://pypi.org/simple deepdraw

   .. tip::

      To avoid long command-lines you may configure pip to define the indexes and
      package search priorities as you like.


.. tab:: mamba/conda

   Stable:

   .. code:: sh

      mamba install -c https://www.idiap.ch/software/biosignal/conda -c conda-forge deepdraw

   Latest beta:

   .. code:: sh

      mamba install -c https://www.idiap.ch/software/biosignal/conda/label/beta -c conda-forge deepdraw


.. _deepdraw.setup:

Setup
-----

A configuration file may be useful to setup global options that should be often
reused.  The location of the configuration file depends on the value of the
environment variable ``$XDG_CONFIG_HOME``, but defaults to
``~/.config/deepdraw.toml``.  You may edit this file using your preferred
editor.

Here is an example configuration file that may be useful as a starting point:

.. code:: toml

   [datadir]
   indian = "/Users/myself/dbs/tbxpredict"
   montgomery = "/Users/myself/dbs/montgomery-xrayset"
   shenzhen = "/Users/myself/dbs/shenzhen"
   nih_cxr14_re = "/Users/myself/dbs/nih-cxr14-re"

   [nih_cxr14_re]
   idiap_folder_structure = false  # set to `true` if at Idiap


.. tip::

   To get a list of valid data directories that can be configured, execute:

   .. code:: sh

      binseg dataset list


   You must procure and download datasets by yourself.  The raw data is not
   included in this package as we are not authorised to redistribute it.

   To check whether the downloaded version is consistent with the structure
   that is expected by this package, run:

   .. code:: sh

      binseg dataset check montgomery


.. _deepdraw.setup.datasets:

====================
 Supported Datasets
====================

Here is a list of currently support datasets in this package, alongside notable
properties.  Each dataset name is linked to the current location where raw data
can be downloaded.  We include the reference of the data split protocols used
to generate iterators for training and testing.


Retinography
------------


.. list-table:: Supported Retinography Datasets (``*``: provided within this package)

   * - Dataset
     - Reference
     - H x W
     - Samples
     - Mask
     - Vessel
     - OD
     - Cup
     - Split Reference
     - Train
     - Test
   * - DRIVE_
     - [DRIVE-2004]_
     - 584 x 565
     - 40
     - ``x``
     - ``x``
     -
     -
     - [DRIVE-2004]_
     - 20
     - 20
   * - STARE_
     - [STARE-2000]_
     - 605 x 700
     - 20
     - ``*``
     - ``x``
     -
     -
     - [MANINIS-2016]_
     - 10
     - 10
   * - CHASE-DB1_
     - [CHASEDB1-2012]_
     - 960 x 999
     - 28
     - ``*``
     - ``x``
     -
     -
     - [CHASEDB1-2012]_
     - 8
     - 20
   * - HRF_
     - [HRF-2013]_
     - 2336 x 3504
     - 45
     - ``x``
     - ``x``
     -
     -
     - [ORLANDO-2017]_
     - 15
     - 30
   * - IOSTAR_
     - [IOSTAR-2016]_
     - 1024 x 1024
     - 30
     - ``x``
     - ``x``
     - ``x``
     -
     - [MEYER-2017]_
     - 20
     - 10
   * - DRIONS-DB_
     - [DRIONSDB-2008]_
     - 400 x 600
     - 110
     -
     -
     - ``x``
     -
     - [MANINIS-2016]_
     - 60
     - 50
   * - `RIM-ONE r3`_
     - [RIMONER3-2015]_
     - 1424 x 1072
     - 159
     -
     -
     - ``x``
     - ``x``
     - [MANINIS-2016]_
     - 99
     - 60
   * - Drishti-GS1_
     - [DRISHTIGS1-2014]_
     - varying
     - 101
     -
     -
     - ``x``
     - ``x``
     - [DRISHTIGS1-2014]_
     - 50
     - 51
   * - REFUGE_
     - [REFUGE-2018]_
     - 2056 x 2124 (1634 x 1634)
     - 1200
     -
     -
     - ``x``
     - ``x``
     - [REFUGE-2018]_
     - 400 (+400)
     - 400
   * - DRHAGIS_
     - [DRHAGIS-2017]_
     - Varying
     - 39
     - ``x``
     - ``x``
     -
     -
     - [DRHAGIS-2017]_
     - 19
     - 20

.. warning:: **REFUGE Dataset Support**

  The original directory ``Training400/AMD`` in REFUGE is considered to be
  replaced by an updated version provided by the `AMD Grand-Challenge`_ (with
  matching names).

  The changes concerns images ``A0012.jpg``, which was corrupted in REFUGE, and
  ``A0013.jpg``, which only exists in the AMD Grand-Challenge version.


X-Ray
-----

.. list-table:: Supported X-Ray Datasets

   * - Dataset
     - Reference
     - H x W
     - Radiography Type
     - Samples
     - Mask
     - Split Reference
     - Train
     - Test
   * - `Montgomery County`_
     - [MC-2014]_
     - 4020 x 4892, or 4892 x 4020
     - Digital Radiography (DR)
     - 138
     - ``*``
     - [GAAL-2020]_
     - 96 (+14)
     - 28
   * - JSRT_
     - [JSRT-2000]_
     - 2048 x 2048
     - Digitized Radiography (laser digitizer)
     - 247
     - ``*``
     - [GAAL-2020]_
     - 172 (+25)
     - 50
   * - Shenzhen_
     - [SHENZHEN-2014]_
     - Varying
     - Computed Radiography (CR)
     - 662
     - ``*``
     - [GAAL-2020]_
     - 396 (+56)
     - 114
   * - CXR8_
     - [CXR8-2017]_
     - 1024 x 1024
     - Digital Radiography
     - 112120
     - ``x``
     - [GAAL-2020]_
     - 78484 (+11212)
     - 22424

.. warning:: **SHENZHEN/JSRT/CXR8 Dataset Support**

  For some datasets (in which the annotations/masks are downloaded separately
  from the dataset with the original images), both the original images and
  annotations must be downloaded and placed inside the same directory, to match
  the dataset reference dictionary's path.

  * The Shenzhen_ root directory should then contain at least these two
    subdirectories:

    - ``CXR_png/`` (directory containing the CXR images)
    - ``mask/`` (contains masks downloaded from `Shenzhen Annotations`_)

  * The CXR8_ root directory:

    - ``images/`` (directory containing the CXR images)
    - ``segmentations/`` (contains masks downloaded from `CXR8 Annotations`_)

  * The JSRT_ root directory:

    - ``All247images/`` (directory containing the CXR images, in raw format)
    - ``scratch/`` (contains masks downloaded from `JSRT Annotations`_)


.. include:: links.rst
