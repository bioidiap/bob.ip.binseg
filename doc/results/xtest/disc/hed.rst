.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.results.xtest.disc.hed:

================================
 HED on Optic-disc Segmentation
================================


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
     - **0.917**
     - 0.937
     - 0.067
     - 0.621
     - 0.806
   * - :py:mod:`drishtigs1-disc <bob.ip.binseg.configs.datasets.drishtigs1.disc_all_768>`
     - 0.873
     - **0.975**
     - 0.209
     - 0.944
     - 0.831
   * - :py:mod:`iostar-disc <bob.ip.binseg.configs.datasets.iostar.optic_disc_768>`
     - 0.082
     - 0.059
     - **0.922**
     - 0.114
     - 0.058
   * - :py:mod:`refuge-disc <bob.ip.binseg.configs.datasets.refuge.disc_768>`
     - 0.299
     - 0.935
     - 0.048
     - **0.924**
     - 0.752
   * - :py:mod:`rimoner3-disc <bob.ip.binseg.configs.datasets.rimoner3.disc_exp1_768>`
     - 0.818
     - 0.837
     - 0.020
     - 0.696
     - **0.954**


.. include:: ../../../links.rst
