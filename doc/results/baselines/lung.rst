.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.results.baselines.lung:

=============================================
 Lung Segmentation from Frontal Chest X-Rays
=============================================

.. list-table::
   :header-rows: 2

   * -
     - :py:mod:`unet <deepdraw.configs.models.unet>`
     - :py:mod:`m2unet <deepdraw.configs.models.m2unet>`
     - :py:mod:`lwnet <deepdraw.configs.models.lwnet>`
   * - Dataset
     - 25.8M
     - 550k
     - 68k
   * - :py:mod:`montgomery <deepdraw.configs.datasets.montgomery.default>`
     -   0.982
     -  `0.980 <baselines_m2unet_montgomery_>`_
     -  `0.978 <baselines_lwnet_montgomery_>`_
   * - :py:mod:`jsrt <deepdraw.configs.datasets.jsrt.default>`
     -   0.982
     -  `0.982 <baselines_m2unet_jsrt_>`_
     -  `0.979 <baselines_lwnet_jsrt_>`_
   * - :py:mod:`shenzhen <deepdraw.configs.datasets.shenzhen.default>`
     -   0.952
     -  `0.955 <baselines_m2unet_shenzhen_>`_
     -  `0.950 <baselines_lwnet_shenzhen_>`_


Notes
-----

* The following table describes recommended batch sizes for 5Gb of RAM GPU
  card:

  .. list-table::

    * - **Models / Datasets**
      - :py:mod:`montgomery <deepdraw.configs.datasets.montgomery.default>`
      - :py:mod:`jsrt <deepdraw.configs.datasets.jsrt.default>`
      - :py:mod:`shenzhen <deepdraw.configs.datasets.shenzhen.default>`
    * - :py:mod:`unet <deepdraw.configs.models.unet>`
      - 8
      - 8
      - 8
    * - :py:mod:`m2unet <deepdraw.configs.models.m2unet>`
      - 8
      - 8
      - 8
    * - :py:mod:`lwnet <deepdraw.configs.models.lwnet>`
      - 8
      - 8
      - 8


.. include:: ../../links.rst
