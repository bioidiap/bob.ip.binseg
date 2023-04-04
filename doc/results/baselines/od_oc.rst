.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.results.baselines.od_oc:

========================================================
 Optic disc and Optic cup Segmentation for Retinography
========================================================

**Optic Disc 512x512**

.. list-table::
   :header-rows: 2

   * -
     - :py:mod:`driu <deepdraw.configs.models.driu>`
     - :py:mod:`hed <deepdraw.configs.models.hed>`
     - :py:mod:`m2unet <deepdraw.configs.models.m2unet>`
     - :py:mod:`unet <deepdraw.configs.models.unet>`
     - :py:mod:`lwnet <deepdraw.configs.models.lwnet>`
     - :py:mod:`driu-od <deepdraw.configs.models.driu_od>`
   * - Dataset
     - 15M
     - 14.7M
     - 550k
     - 25.8M
     - 68k
     - 15.2M
   * - :py:mod:`drionsdb <deepdraw.configs.datasets.drionsdb.expert1_512>`
     - 0.958
     - 0.961
     - 0.960
     - 0.961
     - 0.922
     - 0.960
   * - :py:mod:`drishtigs1-disc <deepdraw.configs.datasets.drishtigs1.disc_all_512>`
     - 0.973
     - 0.975
     - 0.974
     - 0.975
     - 0.965
     - 0.972
   * - :py:mod:`iostar-disc <deepdraw.configs.datasets.iostar.optic_disc_512>`
     - 0.894
     - 0.922
     - 0.913
     - 0.921
     - 0.893
     - 0.921
   * - :py:mod:`refuge-disc <deepdraw.configs.datasets.refuge.disc_512>`
     - 0.921
     - 0.939
     - 0.942
     - 0.945
     - 0.894
     - 0.941
   * - :py:mod:`rimoner3-disc <deepdraw.configs.datasets.rimoner3.disc_exp1_512>`
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
     - :py:mod:`driu <deepdraw.configs.models.driu>`
     - :py:mod:`hed <deepdraw.configs.models.hed>`
     - :py:mod:`m2unet <deepdraw.configs.models.m2unet>`
     - :py:mod:`unet <deepdraw.configs.models.unet>`
     - :py:mod:`lwnet <deepdraw.configs.models.lwnet>`
     - :py:mod:`driu-od <deepdraw.configs.models.driu_od>`
   * - :py:mod:`drionsdb <deepdraw.configs.datasets.drionsdb.expert1_512>`
     - 4
     - 4
     - 6
     - 2
     - 6
     - 4
   * - :py:mod:`drishtigs1-disc <deepdraw.configs.datasets.drishtigs1.disc_all_512>`
     - 4
     - 4
     - 5
     - 2
     - 5
     - 4
   * - :py:mod:`iostar-disc <deepdraw.configs.datasets.iostar.optic_disc_512>`
     - 4
     - 4
     - 6
     - 4
     - 6
     - 4
   * - :py:mod:`refuge-disc <deepdraw.configs.datasets.refuge.disc_512>`
     - 5
     - 5
     - 10
     - 5
     - 20
     - 5
   * - :py:mod:`rimoner3-disc <deepdraw.configs.datasets.rimoner3.disc_exp1_512>`
     - 4
     - 4
     - 5
     - 2
     - 5
     - 4

**Optic Disc 768x768**

.. list-table::
   :header-rows: 2

   * -
     - :py:mod:`driu <deepdraw.configs.models.driu>`
     - :py:mod:`hed <deepdraw.configs.models.hed>`
     - :py:mod:`m2unet <deepdraw.configs.models.m2unet>`
     - :py:mod:`unet <deepdraw.configs.models.unet>`
     - :py:mod:`lwnet <deepdraw.configs.models.lwnet>`
     - :py:mod:`driu-od <deepdraw.configs.models.driu_od>`
   * - Dataset
     - 15M
     - 14.7M
     - 550k
     - 25.8M
     - 68k
     - 15.2M
   * - :py:mod:`drionsdb <deepdraw.configs.datasets.drionsdb.expert1_768>`
     - 0.945
     - 0.917
     - 0.959
     - 0.960
     - 0.875
     - 0.949
   * - :py:mod:`drishtigs1-disc <deepdraw.configs.datasets.drishtigs1.disc_all_768>`
     - 0.971
     - 0.975
     - 0.975
     - 0.976
     - 0.959
     - 0.970
   * - :py:mod:`iostar-disc <deepdraw.configs.datasets.iostar.optic_disc_768>`
     - 0.908
     - 0.922
     - 0.917
     - 0.920
     - 0.898
     - 0.911
   * - :py:mod:`refuge-disc <deepdraw.configs.datasets.refuge.disc_768>`
     - 0.921
     - 0.924
     - 0.936
     - 0.938
     - 0.837
     - 0.929
   * - :py:mod:`rimoner3-disc <deepdraw.configs.datasets.rimoner3.disc_exp1_768>`
     - 0.950
     - 0.954
     - 0.955
     - 0.956
     - 0.925
     - 0.954
   * - Combined datasets
     - 0.947
     - 0.958
     - 0.955
     - 0.958
     - 0.682
     - 0.956

Notes
-----

* The following table describes recommended batch sizes for 24Gb of RAM GPU
  card:


.. list-table::
   :header-rows: 1

   * -
     - :py:mod:`driu <deepdraw.configs.models.driu>`
     - :py:mod:`hed <deepdraw.configs.models.hed>`
     - :py:mod:`m2unet <deepdraw.configs.models.m2unet>`
     - :py:mod:`unet <deepdraw.configs.models.unet>`
     - :py:mod:`lwnet <deepdraw.configs.models.lwnet>`
     - :py:mod:`driu-od <deepdraw.configs.models.driu_od>`
   * - :py:mod:`drionsdb <deepdraw.configs.datasets.drionsdb.expert1_512>`
     - 4
     - 4
     - 6
     - 2
     - 6
     - 4
   * - :py:mod:`drishtigs1-disc <deepdraw.configs.datasets.drishtigs1.disc_all_768>`
     - 4
     - 4
     - 5
     - 2
     - 5
     - 4
   * - :py:mod:`iostar-disc <deepdraw.configs.datasets.iostar.optic_disc_768>`
     - 4
     - 4
     - 6
     - 4
     - 6
     - 4
   * - :py:mod:`refuge-disc <deepdraw.configs.datasets.refuge.disc_768>`
     - 5
     - 5
     - 10
     - 5
     - 20
     - 5
   * - :py:mod:`rimoner3-disc <deepdraw.configs.datasets.rimoner3.disc_exp1_768>`
     - 4
     - 4
     - 5
     - 2
     - 5
     - 4




**Optic Cup 512x512**

.. list-table::
   :header-rows: 2

   * -
     - :py:mod:`driu <deepdraw.configs.models.driu>`
     - :py:mod:`hed <deepdraw.configs.models.hed>`
     - :py:mod:`m2unet <deepdraw.configs.models.m2unet>`
     - :py:mod:`unet <deepdraw.configs.models.unet>`
     - :py:mod:`lwnet <deepdraw.configs.models.lwnet>`
     - :py:mod:`driu-od <deepdraw.configs.models.driu_od>`
   * - Dataset
     - 15M
     - 14.7M
     - 550k
     - 25.8M
     - 68k
     - 15.2M
   * - :py:mod:`drishtigs1-cup <deepdraw.configs.datasets.drishtigs1.cup_all_512>`
     - 0.903
     - 0.910
     - 0.912
     - 0.913
     - 0.877
     - 0.913
   * - :py:mod:`refuge-cup <deepdraw.configs.datasets.refuge.cup_512>`
     - 0.861
     - 0.853
     - 0.831
     - 0.863
     - 0.700
     - 0.854
   * - :py:mod:`rimoner3-cup <deepdraw.configs.datasets.rimoner3.cup_exp1_512>`
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
     - :py:mod:`driu <deepdraw.configs.models.driu>`
     - :py:mod:`hed <deepdraw.configs.models.hed>`
     - :py:mod:`m2unet <deepdraw.configs.models.m2unet>`
     - :py:mod:`unet <deepdraw.configs.models.unet>`
     - :py:mod:`lwnet <deepdraw.configs.models.lwnet>`
     - :py:mod:`driu-od <deepdraw.configs.models.driu_od>`
   * - :py:mod:`drishtigs1-cup <deepdraw.configs.datasets.drishtigs1.cup_all_512>`
     - 4
     - 4
     - 5
     - 2
     - 5
     - 4
   * - :py:mod:`refuge-cup <deepdraw.configs.datasets.refuge.cup_512>`
     - 5
     - 5
     - 10
     - 5
     - 20
     - 5
   * - :py:mod:`rimoner3-cup <deepdraw.configs.datasets.rimoner3.cup_exp1_512>`
     - 4
     - 4
     - 5
     - 2
     - 5
     - 4

**Optic Cup 768x768**

.. list-table::
   :header-rows: 2

   * -
     - :py:mod:`driu <deepdraw.configs.models.driu>`
     - :py:mod:`hed <deepdraw.configs.models.hed>`
     - :py:mod:`m2unet <deepdraw.configs.models.m2unet>`
     - :py:mod:`unet <deepdraw.configs.models.unet>`
     - :py:mod:`lwnet <deepdraw.configs.models.lwnet>`
     - :py:mod:`driu-od <deepdraw.configs.models.driu_od>`
   * - Dataset
     - 15M
     - 14.7M
     - 550k
     - 25.8M
     - 68k
     - 15.2M
   * - :py:mod:`drishtigs1-cup <deepdraw.configs.datasets.drishtigs1.cup_all_768>`
     - 0.899
     - 0.904
     - 0.918
     - 0.913
     - 0.861
     - 0.913
   * - :py:mod:`refuge-cup <deepdraw.configs.datasets.refuge.cup_768>`
     - 0.830
     - 0.852
     - 0.787
     - 0.828
     - 0.590
     - 0.838
   * - :py:mod:`rimoner3-cup <deepdraw.configs.datasets.rimoner3.cup_exp1_768>`
     - 0.769
     - 0.804
     - 0.824
     - 0.809
     - 0.748
     - 0.813
   * - Combined datasets
     - 0.854
     - 0.847
     - 0.865
     - 0.864
     - 0.668
     - 0.860

Notes
-----

* The following table describes recommended batch sizes for 24Gb of RAM GPU
  card:


.. list-table::
   :header-rows: 1

   * -
     - :py:mod:`driu <deepdraw.configs.models.driu>`
     - :py:mod:`hed <deepdraw.configs.models.hed>`
     - :py:mod:`m2unet <deepdraw.configs.models.m2unet>`
     - :py:mod:`unet <deepdraw.configs.models.unet>`
     - :py:mod:`lwnet <deepdraw.configs.models.lwnet>`
     - :py:mod:`driu-od <deepdraw.configs.models.driu_od>`
   * - :py:mod:`drishtigs1-cup <deepdraw.configs.datasets.drishtigs1.cup_all_768>`
     - 4
     - 4
     - 5
     - 2
     - 5
     - 4
   * - :py:mod:`refuge-cup <deepdraw.configs.datasets.refuge.cup_768>`
     - 5
     - 5
     - 10
     - 5
     - 20
     - 5
   * - :py:mod:`rimoner3-cup <deepdraw.configs.datasets.rimoner3.cup_exp1_768>`
     - 4
     - 4
     - 5
     - 2
     - 5
     - 4


.. include:: ../../links.rst
