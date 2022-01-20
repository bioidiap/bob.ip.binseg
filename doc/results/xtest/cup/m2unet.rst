.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.results.xtest.cup.m2unet:

==================================
 M2UNET on Optic-cup Segmentation
==================================


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
     - **0.918**
     - 0.729
     - 0.735
   * - :py:mod:`refuge-cup <bob.ip.binseg.configs.datasets.refuge.cup_768>`
     - 0.837
     - **0.787**
     - 0.693
   * - :py:mod:`rimoner3-cup <bob.ip.binseg.configs.datasets.rimoner3.cup_exp1_512>`
     - 0.601
     - 0.441
     - **0.824**

.. include:: ../../../links.rst
