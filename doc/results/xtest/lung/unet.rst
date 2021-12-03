.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.results.xtest.lung.unet:

============================
 UNet on Lung Segmentation
============================


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
   * - :py:mod:`montgomery <bob.ip.binseg.configs.datasets.montgomery.default>`
     - **0.982**
     - 0.970
     - 0.963
   * - :py:mod:`jsrt <bob.ip.binseg.configs.datasets.jsrt.default>`
     - 0.973
     - **0.982**
     - 0.963
   * - :py:mod:`shenzhen <bob.ip.binseg.configs.datasets.shenzhen.default>`
     - 0.942
     - 0.945
     - **0.952**


.. include:: ../../../links.rst
