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

The package supports a range of retina fundus datasets, but does not include
the raw data itself, which you must procure.

To setup a dataset, do the following:

1. Download the dataset from the authors website (see
   :ref:`bob.ip.binseg.datasets` for download links and details), unpack it and
   store the directory leading to the uncompressed directory structure.

   .. warning::

      Our dataset connectors expect you provide "root" paths of raw datasets as
      you unpack them in their **pristine** state.  Changing the location of
      files within a dataset distribution will likely cause execution errors.

3.  For each dataset that you are planning to use, set the ``datadir`` to the
    root path where it is stored.  E.g.:

    .. code-block:: sh

       (<myenv>) $ bob config set bob.db.drive.datadir "/path/to/drivedataset/"

    To check your current setup, do the following:

    .. code-block:: sh

       (<myenv>) $ bob config show
       {
           "bob.db.chasedb1.datadir": "/path/to/chasedb1/",
           "bob.db.drionsdb.datadir": "/path/to/drionsdb",
           "bob.db.drive.datadir": "/path/to/drive",
           "bob.db.hrf.datadir": "/path/to/hrf",
       }

    This command will show the set location for each configured dataset.  These
    paths are automatically used by the dataset iterators provided by the
    ``bob.db`` packages to find the raw datafiles.

4. To check whether the downloaded version is consistent with the structure
   that is expected by our ``bob.db`` packages, run ``bob_dbmanage.py
   <dataset> checkfiles``, where ``<dataset>`` should be replaced by the
   dataset programmatic name. E.g., to check DRIVE files, use:

   .. code-block:: sh

      (<myenv>) $ bob_dbmanage.py drive checkfiles
      > checkfiles completed sucessfully

   If there are problems on the current file organisation, this procedure
   should detect and highlight which files are missing.

   .. tip::

      The programmatic name of datasets follow the ``bob.db.<dataset>``
      nomenclature.  For example, the programmatic name of CHASE-DB1 is
      ``chasedb1``, because the package name implementing iterators to its
      files is ``bob.db.chasedb1``.


.. include:: links.rst
