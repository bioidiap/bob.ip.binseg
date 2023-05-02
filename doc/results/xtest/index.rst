.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.results.xtest:

==========================
 Cross-Database (X-)Tests
==========================

* Models are trained and tested on the same dataset (numbers in parenthesis
  indicate number of parameters per model), and then evaluated across the test
  sets of other databases.  X-tested datasets therefore represent *unseen*
  data and can be a good proxy for generalisation analysis.
* Each table row indicates a base trained model and each column the databases
  the model was tested against.  The native performance (intra-database) is
  marked **in bold**.  Thresholds are chosen *a priori* on the training set of
  the database used to generate the model being cross-tested.  Hence, the
  threshold used for all experiments in a same row is always the same.
* You can cross check the analysis numbers provided in this table by
  downloading this software package, the raw data, and running ``bob binseg
  analyze`` providing the model URL as ``--weight`` parameter, and then the
  ``-xtest`` resource variant of the dataset the model was trained on.  For
  example, to run cross-evaluation tests for the DRIVE dataset, use the
  configuration resource :py:mod:`drive-xtest
  <deepdraw.configs.datasets.drive.xtest>`.
* For each row, the peak performance is always obtained in an intra-database
  test (training and testing on the same database).  Conversely, we observe a
  performance degradation (albeit not catastrophic in most cases) for all other
  datasets in the cross test.
* We only show results for select systems in :ref:`baseline analysis
  <deepdraw.results.baselines>`.  You may run analysis on the other models
  by downloading them from our website (via the ``--weight`` parameter on the
  :ref:`analyze script <deepdraw.cli>`).


Models on Specific Tasks
------------------------

.. toctree::
   :maxdepth: 2

   vessel/driu
   vessel/m2unet
   vessel/unet
   disc/hed
   disc/m2unet
   disc/unet
   cup/hed
   cup/m2unet
   cup/unet
   lung/lwnet
   lung/m2unet
   lung/unet


.. include:: ../../links.rst
