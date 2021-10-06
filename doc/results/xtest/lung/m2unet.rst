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
     - 512 x 512
     - 512 x 512
     - 512 x 512
   * - :py:mod:`montgomery <bob.ip.binseg.configs.datasets.montgomery.default>` (`model <baselines_m2unet_montgomery_>`_)
     - **0.982**
     - 0.970
     - 0.959
   * - :py:mod:`jsrt <bob.ip.binseg.configs.datasets.jsrt.default>` (`model <baselines_m2unet_jsrt_>`_)
     - 0.973
     - **0.982**
     - 0.961
   * - :py:mod:`shenzhen <bob.ip.binseg.configs.datasets.shenzhen.default>` (`model <baselines_m2unet_shenzhen_>`_)
     - 0.935
     - 0.944
     - **0.955**


.. include:: ../../../links.rst
