.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.results.xtest.vessel.driu:

=============================
 DRIU on Vessel Segmentation
=============================


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
   * - :py:mod:`drive <deepdraw.configs.datasets.drive.default>` (`model <baselines_driu_drive_>`_)
     - **0.819 (0.016)**
     - 0.759 (0.151)
     - 0.321 (0.068)
     - 0.711 (0.067)
     - 0.493 (0.049)
   * - :py:mod:`stare <deepdraw.configs.datasets.stare.ah>` (`model <baselines_driu_stare_>`_)
     - 0.733 (0.037)
     - **0.824 (0.037)**
     - 0.491 (0.094)
     - 0.773 (0.051)
     - 0.469 (0.055)
   * - :py:mod:`chasedb1 <deepdraw.configs.datasets.chasedb1.first_annotator>` (`model <baselines_driu_chase_>`_)
     - 0.730 (0.023)
     - 0.730 (0.101)
     - **0.811 (0.018)**
     - 0.779 (0.043)
     - 0.774 (0.019)
   * - :py:mod:`hrf <deepdraw.configs.datasets.hrf.default>` (`model <baselines_driu_hrf_>`_)
     - 0.702 (0.038)
     - 0.641 (0.160)
     - 0.600 (0.072)
     - **0.802 (0.039)**
     - 0.546  (0.078)
   * - :py:mod:`iostar-vessel <deepdraw.configs.datasets.iostar.vessel>` (`model <baselines_driu_iostar_>`_)
     - 0.758 (0.019)
     - 0.724 (0.115)
     - 0.777 (0.032)
     - 0.727 (0.059)
     - **0.825 (0.021)**


.. include:: ../../../links.rst
