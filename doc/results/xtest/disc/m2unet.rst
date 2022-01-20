.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.results.xtest.disc.m2unet:

===================================
 M2UNET on Optic-disc Segmentation
===================================


.. list-table::
   :header-rows: 2

   * -
     - drionsdb
     - drishtigs1-disc
     - iostar-disc
     - refuge-disc
     - rimoner3-disc
   * - Model / W x H
     - 768 x 768
     - 768 x 768
     - 768 x 768
     - 768 x 768
     - 768 x 768
   * - :py:mod:`drionsdb <bob.ip.binseg.configs.datasets.drionsdb.expert1_768>`
     - **0.959**
     - 0.951
     - 0.206
     - 0.819
     - 0.904
   * - :py:mod:`drishtigs1-disc <bob.ip.binseg.configs.datasets.drishtigs1.disc_all_768>`
     - 0.859
     - **0.975**
     - 0.177
     - 0.914
     - 0.837
   * - :py:mod:`iostar-disc <bob.ip.binseg.configs.datasets.iostar.optic_disc_768>`
     - 0.202
     - 0.134
     - **0.917**
     - 0.094
     - 0.126
   * - :py:mod:`refuge-disc <bob.ip.binseg.configs.datasets.refuge.disc_768>`
     - 0.712
     - 0.950
     - 0.057
     - **0.936**
     - 0.825
   * - :py:mod:`rimoner3-disc <bob.ip.binseg.configs.datasets.rimoner3.disc_exp1_768>`
     - 0.782
     - 0.883
     - 0.057
     - 0.724
     - **0.955**


.. include:: ../../../links.rst
