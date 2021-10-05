.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.results.xtest.lung.lwnet:

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
     - 512 x 512
     - 512 x 512
     - 512 x 512
   * - :py:mod:`montgomery <bob.ip.binseg.configs.datasets.montgomery.default>` (`model <baselines_lwnet_montgomery_>`_)
     - **0.977**
     - 0.946
     - 0.919
   * - :py:mod:`jsrt <bob.ip.binseg.configs.datasets.jsrt.default>` (`model <baselines_lwnet_jsrt_>`_)
     - 0.969
     - **0.980**
     - 0.934
   * - :py:mod:`shenzhen <bob.ip.binseg.configs.datasets.shenzhen.default>` (`model <baselines_lwnet_shenzhen_>`_)
     - 0.946
     - 0.916
     - **0.955**


.. include:: ../../../links.rst
