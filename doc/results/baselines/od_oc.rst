.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.results.baselines.od_oc:

========================================================
 Optic disc and Optic cup Segmentation for Retinography
========================================================

Optic Disc

.. list-table::
   :header-rows: 2

   * -
     - :py:mod:`driu <bob.ip.binseg.configs.models.driu>`
     - :py:mod:`hed <bob.ip.binseg.configs.models.hed>`
     - :py:mod:`m2unet <bob.ip.binseg.configs.models.m2unet>`
     - :py:mod:`unet <bob.ip.binseg.configs.models.unet>`
     - :py:mod:`lwnet <bob.ip.binseg.configs.models.lwnet>`
     - :py:mod:`driu-od <bob.ip.binseg.configs.models.driu_od>`
   * - Dataset
     - 15M
     - 14.7M
     - 550k
     - 25.8M
     - 68k
   * - :py:mod:`drionsdb <bob.ip.binseg.configs.datasets.drionsdb.expert1>`
     - 0.944
     - 0.960
     - 0.960
     - 0.960
     - 0.888
     - 0.960
   * - :py:mod:`drishtigs1-disc <bob.ip.binseg.configs.datasets.drishtigs1.disc_all>`
     - 0.952
     - 0.971
     - 0.967
     - 0.970
     - 0.652
     - 0.897
   * - :py:mod:`iostar-disc <bob.ip.binseg.configs.datasets.iostar.optic_disc>`
     - 0.913
     - 0.910
     - 0.878
     - 0.904
     - 0.875
     - 0.908
   * - :py:mod:`refuge-disc <bob.ip.binseg.configs.datasets.refuge.disc>`
     - 0.872
     - 0.864
     - 0.840
     -
     - 0.754
     -
   * - :py:mod:`rimoner3-disc <bob.ip.binseg.configs.datasets.rimoner3.disc_exp1>`
     - 0.935
     - 0.953
     - 0.950
     - 0.947
     - 0.720
     - 0.949

Notes
-----

* The following table describes recommended batch sizes for 24Gb of RAM GPU
  card:


.. list-table::
   :header-rows: 2

   * - **Models / Datasets**
     - :py:mod:`driu <bob.ip.binseg.configs.models.driu>`
     - :py:mod:`hed <bob.ip.binseg.configs.models.hed>`
     - :py:mod:`m2unet <bob.ip.binseg.configs.models.m2unet>`
     - :py:mod:`unet <bob.ip.binseg.configs.models.unet>`
     - :py:mod:`lwnet <bob.ip.binseg.configs.models.lwnet>`
     - :py:mod:`driu-od <bob.ip.binseg.configs.models.driu_od>`
   * - :py:mod:`drionsdb <bob.ip.binseg.configs.datasets.drionsdb.expert1>`
     - 4
     - 4
     - 6
     - 2
     - 8
     - 4
   * - :py:mod:`drishtigs1-disc <bob.ip.binseg.configs.datasets.drishtigs1.disc_all>`
     - 1
     - 1
     - 2
     - 1
     - 2
     - 1
   * - :py:mod:`iostar-disc <bob.ip.binseg.configs.datasets.iostar.optic_disc>`
     - 1
     - 1
     - 1
     - 1
     - 4
     - 1
   * - :py:mod:`refuge-disc <bob.ip.binseg.configs.datasets.refuge.disc>`
     - 1
     - 2
     - 1
     - 1
     - 4
     - 1
   * - :py:mod:`rimoner3-disc <bob.ip.binseg.configs.datasets.rimoner3.disc_exp1>`
     - 1
     - 1
     - 1
     - 1
     - 4
     - 1
