.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.results.baselines:

===================
 Baseline Results
===================

* Benchmark results for models: DRIU, HED, M2U-Net, U-Net, and Little W-Net.
* Models are trained and tested on the same dataset (**numbers in bold**
  indicate approximate number of parameters per model). DRIU, HED, M2U-Net and
  U-Net Models are trained for a fixed number of 1000 epochs, with a learning
  rate of 0.001 until epoch 900 and then 0.0001 until the end of the training,
  after being initialized with a VGG-16 backend.  Little W-Net models are
  trained using a cosine anneling strategy (see [GALDRAN-2020]_ and
  [SMITH-2017]_) for 2000 epochs.
* During the training session, an unaugmented copy of the training set is used
  as validation set.  We keep checkpoints for the best performing networks
  based on such validation set.  The best performing network during training is
  used for evaluation.
* Image masks are used during the evaluation, errors are only assessed within
  the masked region.
* Database and model resource configuration links (table top row and left
  column) are linked to the originating configuration files used to obtain
  these results.
* Check `our paper`_ for details on the calculation of the F1 Score and standard
  deviations (in parentheses).
* Single performance numbers correspond to *a priori* performance indicators,
  where the threshold is previously selected on the training set
* You can cross check the analysis numbers provided in this table by
  downloading this software package, the raw data, and running ``bob binseg
  analyze`` providing the model URL as ``--weight`` parameter.
* For comparison purposes, we provide "second-annotator" performances on the
  same test set, where available.


Tasks
-----

.. toctree::
   :maxdepth: 1

   vessel
   lung
   od_oc


.. include:: ../../links.rst
