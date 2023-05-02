.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.results.xtest.lung.lwnet:

===================================
 Little W-Net on Lung Segmentation
===================================


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
   * - :py:mod:`montgomery <deepdraw.configs.datasets.montgomery.default>` (`model <baselines_lwnet_montgomery_>`_)
     - **0.978**
     - 0.969
     - 0.964
   * - :py:mod:`jsrt <deepdraw.configs.datasets.jsrt.default>` (`model <baselines_lwnet_jsrt_>`_)
     - 0.967
     - **0.979**
     - 0.963
   * - :py:mod:`shenzhen <deepdraw.configs.datasets.shenzhen.default>` (`model <baselines_lwnet_shenzhen_>`_)
     - 0.920
     - 0.939
     - **0.950**


.. include:: ../../../links.rst
