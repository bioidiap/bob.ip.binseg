.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.results.baselines.lung:

=============================================
 Lung Segmentation from Frontal Chest X-Rays
=============================================

.. list-table::
   :header-rows: 2

   * -
     - :py:mod:`m2unet <bob.ip.binseg.configs.models.m2unet>`
     - :py:mod:`lwnet <bob.ip.binseg.configs.models.lwnet>`
   * - Dataset
     - 550k
     - 68k
   * - :py:mod:`montgomery <bob.ip.binseg.configs.datasets.montgomery.default>`
     -  `0.982 <baselines_m2unet_montgomery_>`_
     -  `0.975 <baselines_lwnet_montgomery_>`_
   * - :py:mod:`jsrt <bob.ip.binseg.configs.datasets.jsrt.default>`
     -  `0.982 <baselines_m2unet_jsrt_>`_
     -  `0.977 <baselines_lwnet_jsrt_>`_
   * - :py:mod:`shenzhen <bob.ip.binseg.configs.datasets.shenzhen.default>`
     -  `0.955 <baselines_m2unet_shenzhen_>`_
     -  `0.952 <baselines_lwnet_shenzhen_>`_


Notes
-----

* The following table describes recommended batch sizes for 5Gb of RAM GPU
  card:

  .. list-table::

    * - **Models / Datasets**
      - :py:mod:`montgomery <bob.ip.binseg.configs.datasets.montgomery.default>`
      - :py:mod:`jsrt <bob.ip.binseg.configs.datasets.jsrt.default>`
      - :py:mod:`shenzhen <bob.ip.binseg.configs.datasets.shenzhen.default>`
    * - :py:mod:`m2unet <bob.ip.binseg.configs.models.m2unet>`
      - 8
      - 8
      - 8
    * - :py:mod:`lwnet <bob.ip.binseg.configs.models.lwnet>`
      - 8
      - 8
      - 8


.. include:: ../../links.rst
