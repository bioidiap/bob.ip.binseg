.. -*- coding: utf-8 -*-
.. _bob.ip.binseg.setup:

=======
 Setup
=======

Complete Bob's `installation`_ instructions. Then, to install this package, do
this:

.. code-block:: sh

   $ conda activate <myenv>
   (<myenv>) $ conda install bob.ip.binseg

.. note::

   The value ``<myenv>`` should correspond to the name of the environment where
   you initially installed your Bob packages.


Datasets
--------

The package supports a range of retina fundus datasets, but does not install the
`bob.db` iterator APIs by default, or includes the raw data itself, which you
must procure.

To setup a dataset, do the following:

1. Download the dataset from the authors website (see below for all download
   links)
2. Install the corresponding bob.db package via ``conda install
   bob.db.<database>``.  E.g. to install the DRIVE API run ``conda install
   bob.db.drive``
3. :ref:`datasetpathsetup`
4. :ref:`dsconsistency`

+------------+----------------------------------------------------------------------+---------------------+
| Dataset    | Website                                                              | `bob.db` package    |
+------------+----------------------------------------------------------------------+---------------------+
| STARE      | http://cecas.clemson.edu/~ahoover/stare/                             | `bob.db.stare`      |
+------------+----------------------------------------------------------------------+---------------------+
| DRIVE      | https://www.isi.uu.nl/Research/Databases/DRIVE/                      | `bob.db.drive`      |
+------------+----------------------------------------------------------------------+---------------------+
| DRIONS     | http://www.ia.uned.es/~ejcarmona/DRIONS-DB.html                      | `bob.db.drionsdb`   |
+------------+----------------------------------------------------------------------+---------------------+
| RIM-ONE    | http://medimrg.webs.ull.es/research/downloads/                       | `bob.db.rimoner3`   |
+------------+----------------------------------------------------------------------+---------------------+
| CHASE-DB1  | https://blogs.kingston.ac.uk/retinal/chasedb1/                       | `bob.db.chasedb`    |
+------------+----------------------------------------------------------------------+---------------------+
| HRF        | https://www5.cs.fau.de/research/data/fundus-images/                  | `bob.db.hrf`        |
+------------+----------------------------------------------------------------------+---------------------+
| Drishti-GS | http://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php | `bob.db.drishtigs1` |
+------------+----------------------------------------------------------------------+---------------------+
| IOSTAR     | http://www.retinacheck.org/datasets                                  | `bob.db.iostar`     |
+------------+----------------------------------------------------------------------+---------------------+
| REFUGE     | https://refuge.grand-challenge.org/Details/                          | `bob.db.refuge`     |
+------------+----------------------------------------------------------------------+---------------------+


.. _datasetpathsetup:

Set up dataset paths
====================

.. warning::

   Our dataset connectors expect you provide "root" paths of raw datasets as
   you unpack them in their **pristine** state.  Changing the location of files
   within a dataset distribution will likely cause execution errors.

For each dataset that you are planning to use, set the ``datadir`` to the root
path where it is stored.  E.g.:

.. code-block:: sh

   (<myenv>) $ bob config set bob.db.drive.datadir "/path/to/drivedataset/"

To check your current setup, do the following:

.. code-block:: sh

   (<myenv>) $ bob config show
   {
       "bob.db.chasedb1.datadir": "/idiap/resource/database/CHASE-DB11/",
       "bob.db.drionsdb.datadir": "/idiap/resource/database/DRIONS",
       "bob.db.drishtigs1.datadir": "/idiap/resource/database/Drishti-GS1/",
       "bob.db.drive.datadir": "/idiap/resource/database/DRIVE",
       "bob.db.hrf.datadir": "/idiap/resource/database/HRF",
       "bob.db.iostar.datadir": "/idiap/resource/database/IOSTAR/IOSTAR Vessel Segmentation Dataset/",
       "bob.db.refuge.datadir": "/idiap/resource/database/REFUGE",
       "bob.db.rimoner3.datadir": "/idiap/resource/database/RIM-ONE/RIM-ONE r3",
       "bob.db.stare.datadir": "/idiap/resource/database/STARE"
   }


.. _dsconsistency:

Test dataset consitency
=======================

To check whether the downloaded version is consistent with the structure that
is expected by our ``bob.db`` packages, run ``bob_dbmanage.py datasettocheck
checkfiles`` E.g.:

.. code-block:: sh

   (<myenv>) $ bob_dbmanage.py drive checkfiles
   > checkfiles completed sucessfully

If there are problems on the current file organisation, this procedure should
detect and highlight which files are missing.


.. include:: links.rst
