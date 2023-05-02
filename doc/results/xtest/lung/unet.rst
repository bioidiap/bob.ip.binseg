.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.results.xtest.lung.unet:

============================
 UNet on Lung Segmentation
============================


.. list-table::
   :header-rows: 2

   * -
     - montgomery
     - jsrt
     - shenzhen
   * - Model / W x H
     - 256 x 256
     - 256 x 256
     - 256 x 256
   * - :py:mod:`montgomery <deepdraw.configs.datasets.montgomery.default>`
     - **0.982**
     - 0.970
     - 0.963
   * - :py:mod:`jsrt <deepdraw.configs.datasets.jsrt.default>`
     - 0.973
     - **0.982**
     - 0.963
   * - :py:mod:`shenzhen <deepdraw.configs.datasets.shenzhen.default>`
     - 0.942
     - 0.945
     - **0.952**


.. include:: ../../../links.rst
