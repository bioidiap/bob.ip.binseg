.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.results.xtest.vessel.unet:

=============================
 UNET on Vessel Segmentation
=============================

768X768 résolution

.. list-table::
   :header-rows: 2

   * -
     - drive
     - stare
     - chasedb1
     - hrf
     - iostar-vessel
   * - Model / W x H
     - 768 x 768
     - 768 x 768
     - 768 x 768
     - 768 x 768
     - 768 x 768
   * - :py:mod:`drive <deepdraw.configs.datasets.drive.default_768>`
     - **0.814**
     - 0.807
     - 0.752
     - 0.736
     - 0.739
   * - :py:mod:`stare <deepdraw.configs.datasets.stare.ah_768>`
     - 0.767
     - **0.829**
     - 0.752
     - 0.755
     - 0.739
   * - :py:mod:`chasedb1 <deepdraw.configs.datasets.chasedb1.first_annotator_768>`
     - 0.774
     - 0.800
     - **0.803**
     - 0.730
     - 0.771
   * - :py:mod:`hrf <deepdraw.configs.datasets.hrf.default_768>`
     - 0.712
     - 0.769
     - 0.648
     - **0.804**
     - 0.700
   * - :py:mod:`iostar-vessel <deepdraw.configs.datasets.iostar.vessel_768>`
     - 0.768
     - 0.783
     - 0.773
     - 0.744
     - **0.820**


1024x1024 résolution

.. list-table::
   :header-rows: 2

   * -
     - drive
     - stare
     - chasedb1
     - hrf
     - iostar-vessel
   * - Model / W x H
     - 1024 x 1024
     - 1024 x 1024
     - 1024 x 1024
     - 1024 x 1024
     - 1024 x 1024
   * - :py:mod:`drive <deepdraw.configs.datasets.drive.default_1024>`
     - **0.815**
     - 0.812
     - 0.732
     - 0.742
     - 0.780
   * - :py:mod:`stare <deepdraw.configs.datasets.stare.ah_1024>`
     - 0.772
     - **0.824**
     - 0.767
     - 0.758
     - 0.766
   * - :py:mod:`chasedb1 <deepdraw.configs.datasets.chasedb1.first_annotator_1024>`
     - 0.774
     - 0.762
     - **0.806**
     - 0.729
     - 0.779
   * - :py:mod:`hrf <deepdraw.configs.datasets.hrf.default_1024>`
     - 0.710
     - 0.762
     - 0.569
     - **0.807**
     - 0.672
   * - :py:mod:`iostar-vessel <deepdraw.configs.datasets.iostar.vessel>`
     - 0.764
     - 0.775
     - 0.776
     - 0.741
     - **0.825**




.. include:: ../../../links.rst
