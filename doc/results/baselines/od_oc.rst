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
     - 15.2M
   * - :py:mod:`drionsdb <bob.ip.binseg.configs.datasets.drionsdb.expert1>`
     - 0.958
     - 0.961
     - 0.960
     - 0.961
     - 0.922
     - 0.960
   * - :py:mod:`drishtigs1-disc <bob.ip.binseg.configs.datasets.drishtigs1.disc_all>`
     - 0.973
     - 0.975
     - 0.974
     - 0.975
     - 0.965
     - 0.972
   * - :py:mod:`iostar-disc <bob.ip.binseg.configs.datasets.iostar.optic_disc>`
     - 0.894
     - 0.922
     - 0.913
     - 0.921
     - 0.893
     - 0.921
   * - :py:mod:`refuge-disc <bob.ip.binseg.configs.datasets.refuge.disc>`
     - 0.921
     - 0.939
     - 0.942
     - 0.945
     - 0.894
     - 0.941
   * - :py:mod:`rimoner3-disc <bob.ip.binseg.configs.datasets.rimoner3.disc_exp1>`
     - 0.950
     - 0.955
     - 0.953
     - 0.956
     - 0.939
     - 0.954

Notes
-----

* The following table describes recommended batch sizes for 24Gb of RAM GPU
  card:


.. list-table::
   :header-rows: 1

   * -
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
     - 6
     - 4
   * - :py:mod:`drishtigs1-disc <bob.ip.binseg.configs.datasets.drishtigs1.disc_all>`
     - 4
     - 4
     - 5
     - 2
     - 5
     - 4
   * - :py:mod:`iostar-disc <bob.ip.binseg.configs.datasets.iostar.optic_disc>`
     - 4
     - 4
     - 6
     - 4
     - 6
     - 4
   * - :py:mod:`refuge-disc <bob.ip.binseg.configs.datasets.refuge.disc>`
     - 5
     - 5
     - 10
     - 5
     - 20
     - 5
   * - :py:mod:`rimoner3-disc <bob.ip.binseg.configs.datasets.rimoner3.disc_exp1>`
     - 4
     - 4
     - 5
     - 2
     - 5
     - 4

Optic Cup

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
     - 15.2M
   * - :py:mod:`drishtigs1-cup <bob.ip.binseg.configs.datasets.drishtigs1.cup_all>`
     - 0.903
     - 0.910
     - 0.912
     - 0.913
     - 0.877
     - 0.913
   * - :py:mod:`refuge-cup <bob.ip.binseg.configs.datasets.refuge.cup>`
     - 0.861
     -
     - 0.831
     - 0.863
     - 0.700
     - 0.854
   * - :py:mod:`rimoner3-cup <bob.ip.binseg.configs.datasets.rimoner3.cup_exp1>`
     - 0.799
     - 0.819
     - 0.829
     - 0.819
     - 0.736
     - 0.822

Notes
-----

* The following table describes recommended batch sizes for 24Gb of RAM GPU
  card:


.. list-table::
   :header-rows: 1

   * -
     - :py:mod:`driu <bob.ip.binseg.configs.models.driu>`
     - :py:mod:`hed <bob.ip.binseg.configs.models.hed>`
     - :py:mod:`m2unet <bob.ip.binseg.configs.models.m2unet>`
     - :py:mod:`unet <bob.ip.binseg.configs.models.unet>`
     - :py:mod:`lwnet <bob.ip.binseg.configs.models.lwnet>`
     - :py:mod:`driu-od <bob.ip.binseg.configs.models.driu_od>`
   * - :py:mod:`drishtigs1-cup <bob.ip.binseg.configs.datasets.drishtigs1.cup_all>`
     - 4
     - 4
     - 5
     - 2
     - 5
     - 4
   * - :py:mod:`refuge-cup <bob.ip.binseg.configs.datasets.refuge.cup>`
     - 5
     - 5
     - 10
     - 5
     - 20
     - 5
   * - :py:mod:`rimoner3-cup <bob.ip.binseg.configs.datasets.rimoner3.cup_exp1>`
     - 4
     - 4
     - 5
     - 2
     - 5
     - 4

.. include:: ../../links.rst
