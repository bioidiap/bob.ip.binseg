.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.results.cod:

================================
 Combined Dataset (COD) Results
================================

* Models are trained on a COD **excluding** the target dataset, and tested on
  the target dataset (**numbers in bold** indicate number of parameters per
  model).  Models are trained for a fixed number of 1000 epochs, with a
  learning rate of 0.001 until epoch 900 and then 0.0001 until the end of the
  training.
* Database and model resource configuration links (table top row and left
  column) are linked to the originating configuration files used to obtain
  these results.
* Single performance numbers correspond to *a priori* performance indicators,
  where the threshold is previously selected on the training set (COD
  excluding the target dataset)
* You can cross check the analysis numbers provided in this table by
  downloading this software package, the raw data, and running ``bob binseg
  analyze`` providing the model URL as ``--weight`` parameter.
* For comparison purposes, we provide "second-annotator" performances on the
  same test set, where available.


.. toctree::
   :maxdepth: 1

   vessel


.. include:: ../../links.rst
