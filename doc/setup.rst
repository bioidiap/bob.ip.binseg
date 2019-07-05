.. -*- coding: utf-8 -*-
.. _bob.ip.binseg.setup:

=========
Setup
=========

Bob.ip.binseg
=============

Complete bob's `installation`_ instructions. Then, to install this
package

.. code-block:: bash

    conda install bob.ip.binseg

Dataset Links
=============

+------------+----------------------------------------------------------------------+
| Dataset    | Website                                                              |
+------------+----------------------------------------------------------------------+
| STARE      | http://cecas.clemson.edu/~ahoover/stare/                             |
+------------+----------------------------------------------------------------------+
| DRIVE      | https://www.isi.uu.nl/Research/Databases/DRIVE/                      |
+------------+----------------------------------------------------------------------+
| DRIONS     | http://www.ia.uned.es/~ejcarmona/DRIONS-DB.html                      |
+------------+----------------------------------------------------------------------+
| RIM-ONE    | http://medimrg.webs.ull.es/research/downloads/                       |
+------------+----------------------------------------------------------------------+
| CHASE-DB1  | https://blogs.kingston.ac.uk/retinal/chasedb1/                       |
+------------+----------------------------------------------------------------------+
| HRF        | https://www5.cs.fau.de/research/data/fundus-images/                  |
+------------+----------------------------------------------------------------------+
| Drishti-GS | http://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php |
+------------+----------------------------------------------------------------------+
| IOSTAR     | http://www.retinacheck.org/datasets                                  |
+------------+----------------------------------------------------------------------+
| REFUGE     | https://refuge.grand-challenge.org/Details/                          |
+------------+----------------------------------------------------------------------+

Setting up dataset paths
========================

For each dataset that you are planning to use, set the datadir to
the path where it is stored. E.g.:

.. code-block:: bash

    bob config set bob.db.drive.datadir "/path/to/drivedataset/"

To check your current setup

.. code-block:: bash

    bob config show

This should result in an output similar to the following:

.. code-block:: bash

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



Testing dataset consitency
==========================

To check whether the downloaded version is consistent with
the structure that is expected by our ``bob.db`` packages
run ``bob_dbmanage.py datasettocheck checkfiles``
E.g.:

.. code-block:: sh

    conda activate your-conda-env-with-bob.ip.binseg
    bob_dbmanage.py drive checkfiles
    > checkfiles completed sucessfully

.. include:: links.rst
