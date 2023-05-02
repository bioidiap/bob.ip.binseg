.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.results.xtest.cup.hed:

================================
 HED on Optic-cup Segmentation
================================


.. list-table::
   :header-rows: 2

   * -
     - drishtigs1-cup
     - refuge-cup
     - rimoner3-cup
   * - Model / W x H
     - 768 x 768
     - 768 x 768
     - 768 x 768
   * - :py:mod:`drishtigs1-cup <deepdraw.configs.datasets.drishtigs1.cup_all_768>`
     - **0.904**
     - 0.789
     - 0.727
   * - :py:mod:`refuge-cup <deepdraw.configs.datasets.refuge.cup_768>`
     - 0.818
     - **0.852**
     - 0.690
   * - :py:mod:`rimoner3-cup <deepdraw.configs.datasets.rimoner3.cup_exp1_512>`
     - 0.720
     - 0.673
     - **0.804**

.. include:: ../../../links.rst
