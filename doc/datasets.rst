.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.datasets:

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
     - Samples
     - Mask
     - Split Reference
     - Train
     - Test
   * - `Montgomery County`_
     - [MC-2014]_
     - 4020 x 4892, or 4892 x 4020
     - 138
     - ``x``
     - [GAAL-2020]_
     - 96 (+14)
     - 28
   * - JSRT_
     - [JSRT-2000]_
     - 2048 x 2048
     - 247
     - ``x``
     - [GAAL-2020]_
     - 172 (+25)
     - 50
   * - Shenzhen_
     - [SHENZHEN-2014]_
     - Varying
     - 662
     - ``x``
     - [GAAL-2020]_
     - 396 (+56)
     - 114

.. include:: links.rst
