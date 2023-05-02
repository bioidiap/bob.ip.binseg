.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.results.xtest.vessel.m2unet:

================================
 M2U-Net on Vessel Segmentation
================================


.. list-table::
   :header-rows: 2

   * -
     - drive
     - stare
     - chasedb1
     - hrf
     - iostar-vessel
   * - Model / W x H
     - 544 x 544
     - 704 x 608
     - 960 x 960
     - 1648 x 1168
     - 1024 x 1024
   * - :py:mod:`drive <deepdraw.configs.datasets.drive.default>` (`model <baselines_m2unet_drive_>`_)
     - **0.804 (0.014)**
     - 0.736 (0.144)
     - 0.548 (0.055)
     - 0.744 (0.058)
     - 0.722 (0.036)
   * - :py:mod:`stare <deepdraw.configs.datasets.stare.ah>` (`model <baselines_m2unet_stare_>`_)
     - 0.715 (0.031)
     - **0.811 (0.039)**
     - 0.632 (0.033)
     - 0.765 (0.049)
     - 0.673 (0.033)
   * - :py:mod:`chasedb1 <deepdraw.configs.datasets.chasedb1.first_annotator>` (`model <baselines_m2unet_chase_>`_)
     - 0.677 (0.027)
     - 0.695 (0.099)
     - **0.801 (0.018)**
     - 0.763 (0.040)
     - 0.761 (0.018)
   * - :py:mod:`hrf <deepdraw.configs.datasets.hrf.default>` (`model <baselines_m2unet_hrf_>`_)
     - 0.591 (0.071)
     - 0.460 (0.230)
     - 0.332 (0.108)
     - **0.796 (0.043)**
     - 0.419 (0.088)
   * - :py:mod:`iostar-vessel <deepdraw.configs.datasets.iostar.vessel>` (`model <baselines_m2unet_iostar_>`_)
     - 0.743 (0.019)
     - 0.745 (0.076)
     - 0.771 (0.030)
     - 0.749 (0.052)
     - **0.817 (0.021)**


.. include:: ../../../links.rst
