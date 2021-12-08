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
     - 256 x 256
     - 256 x 256
     - 256 x 256
   * - :py:mod:`montgomery <bob.ip.binseg.configs.datasets.montgomery.default>` (`model <baselines_lwnet_montgomery_>`_)
     - **0.978**
     - 0.969
     - 0.964
   * - :py:mod:`jsrt <bob.ip.binseg.configs.datasets.jsrt.default>` (`model <baselines_lwnet_jsrt_>`_)
     - 0.967
     - **0.979**
     - 0.963
   * - :py:mod:`shenzhen <bob.ip.binseg.configs.datasets.shenzhen.default>` (`model <baselines_lwnet_shenzhen_>`_)
     - 0.920
     - 0.939
     - **0.950**


.. include:: ../../../links.rst
