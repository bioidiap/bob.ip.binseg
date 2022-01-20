.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.results.xtest.cup.unet:

================================
 UNET on Optic-cup Segmentation
================================

Dataset on 768x768

.. list-table::
   :header-rows: 2

   * -
     - drishtigs1-cup
     - refuge-cup
     - rimoner3-cup
   * - Model / W x H
     - 768 x 768
     - 768 x 768
     - 768 x 768
   * - :py:mod:`drishtigs1-cup <bob.ip.binseg.configs.datasets.drishtigs1.cup_all_768>`
     - **0.913**
     - 0.817
     - 0.766
   * - :py:mod:`refuge-cup <bob.ip.binseg.configs.datasets.refuge.cup_768>`
     - 0.835
     - **0.828**
     - 0.691
   * - :py:mod:`rimoner3-cup <bob.ip.binseg.configs.datasets.rimoner3.cup_exp1_768>`
     - 0.759
     - 0.591
     - **0.809**

Datasets on 512x512

.. list-table::
   :header-rows: 2

   * -
     - drishtigs1-cup
     - refuge-cup
     - rimoner3-cup
   * - Model / W x H
     - 512 x 512
     - 512 x 512
     - 512 x 512
   * - :py:mod:`drishtigs1-cup <bob.ip.binseg.configs.datasets.drishtigs1.cup_all_512>`
     - **0.913**
     - 0.772
     - 0.763
   * - :py:mod:`refuge-cup <bob.ip.binseg.configs.datasets.refuge.cup_512>`
     - 0.864
     - **0.853**
     - 0.707
   * - :py:mod:`rimoner3-cup <bob.ip.binseg.configs.datasets.rimoner3.cup_exp1_512>`
     - 0.747
     - 0.625
     - **0.819**


.. include:: ../../../links.rst
