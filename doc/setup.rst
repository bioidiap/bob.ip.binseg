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

2.  For each dataset that you are planning to use, set the ``datadir`` to the
    root path where it is stored.  E.g.:

    .. code-block:: sh

       (<myenv>) $ bob config set bob.ip.binseg.drive.datadir "/path/to/drive"

    To check supported raw datasets and your current setup, do the following:

    .. code-block:: sh

       (<myenv>) $ bob binseg dataset list
       Supported datasets:
       - drive: bob.ip.binseg.drive.datadir = "/Users/andre/work/bob/dbs/drive"
       * stare: bob.ip.binseg.stare.datadir (not set)

    This command will show the set location for each configured dataset, and
    the variable names for each supported dataset which has not yet been setup.

3. To check whether the downloaded version is consistent with the structure
   that is expected by this package, run ``bob binseg dataset check
   <dataset>``, where ``<dataset>`` should be replaced by the
   dataset programmatic name. E.g., to check DRIVE files, use:

   .. code-block:: sh

      (<myenv>) $ bob binseg dataset check drive
      ...

   If there are problems on the current file organisation, this procedure
   should detect and highlight which files are missing (cannot be loaded).

.. include:: links.rst
