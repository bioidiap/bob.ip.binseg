.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.results.xtest.disc.unet:

=================================
 UNET on Optic-disc Segmentation
=================================

Datasets on 768x768

.. list-table::
   :header-rows: 2

   * -
     - drionsdb
     - drishtigs1-disc
     - iostar-disc
     - refuge-disc
     - rimoner3-disc
   * - Model / W x H
     - 768 x 768
     - 768 x 768
     - 768 x 768
     - 768 x 768
     - 768 x 768
   * - :py:mod:`drionsdb <deepdraw.configs.datasets.drionsdb.expert1_768>`
     - **0.960**
     - 0.957
     - 0.057
     - 0.886
     - 0.888
   * - :py:mod:`drishtigs1-disc <deepdraw.configs.datasets.drishtigs1.disc_all_768>`
     - 0.932
     - **0.976**
     - 0.057
     - 0.896
     - 0.860
   * - :py:mod:`iostar-disc <deepdraw.configs.datasets.iostar.optic_disc_768>`
     - 0.127
     - 0.342
     - **0.920**
     - 0.143
     - 0.265
   * - :py:mod:`refuge-disc <deepdraw.configs.datasets.refuge.disc_768>`
     - 0.817
     - 0.937
     - 0.057
     - **0.938**
     - 0.838
   * - :py:mod:`rimoner3-disc <deepdraw.configs.datasets.rimoner3.disc_exp1_768>`
     - 0.754
     - 0.772
     - 0.057
     - 0.481
     - **0.956**

Datasets on 512x512

.. list-table::
   :header-rows: 2

   * -
     - drionsdb
     - drishtigs1-disc
     - iostar-disc
     - refuge-disc
     - rimoner3-disc
   * - Model / W x H
     - 512 x 512
     - 512 x 512
     - 512 x 512
     - 512 x 512
     - 512 x 512
   * - :py:mod:`drionsdb <deepdraw.configs.datasets.drionsdb.expert1_512>`
     - **0.961**
     - 0.966
     - 0.320
     - 0.891
     - 0.864
   * - :py:mod:`drishtigs1-disc <deepdraw.configs.datasets.drishtigs1.disc_all_512>`
     - 0.953
     - **0.975**
     - 0.057
     - 0.923
     - 0.884
   * - :py:mod:`iostar-disc <deepdraw.configs.datasets.iostar.optic_disc_512>`
     - 0.050
     - 0.086
     - **0.921**
     - 0.074
     - 0.115
   * - :py:mod:`refuge-disc <deepdraw.configs.datasets.refuge.disc_512>`
     - 0.885
     - 0.943
     - 0.057
     - **0.945**
     - 0.879
   * - :py:mod:`rimoner3-disc <deepdraw.configs.datasets.rimoner3.disc_exp1_512>`
     - 0.864
     - 0.890
     - 0.064
     - 0.734
     - **0.956**



.. include:: ../../../links.rst
