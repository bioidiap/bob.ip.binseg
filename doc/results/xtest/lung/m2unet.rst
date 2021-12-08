.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.results.xtest.lung.m2unet:

================================
 M2U-Net on Lung Segmentation
================================


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
   * - :py:mod:`montgomery <bob.ip.binseg.configs.datasets.montgomery.default>` (`model <baselines_m2unet_montgomery_>`_)
     - **0.980**
     - 0.970
     - 0.962
   * - :py:mod:`jsrt <bob.ip.binseg.configs.datasets.jsrt.default>` (`model <baselines_m2unet_jsrt_>`_)
     - 0.971
     - **0.982**
     - 0.967
   * - :py:mod:`shenzhen <bob.ip.binseg.configs.datasets.shenzhen.default>` (`model <baselines_m2unet_shenzhen_>`_)
     - 0.942
     - 0.945
     - **0.952**


.. include:: ../../../links.rst
