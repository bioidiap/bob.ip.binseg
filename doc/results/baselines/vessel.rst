.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.results.baselines.vessel:

==============================================
 Retinal Vessel Segmentation for Retinography
==============================================


.. list-table::
   :header-rows: 2

   * -
     -
     - :py:mod:`driu <deepdraw.configs.models.driu>`
     - :py:mod:`hed <deepdraw.configs.models.hed>`
     - :py:mod:`m2unet <deepdraw.configs.models.m2unet>`
     - :py:mod:`unet <deepdraw.configs.models.unet>`
     - :py:mod:`lwnet <deepdraw.configs.models.lwnet>`
   * - Dataset
     - 2nd. Annot.
     - 15M
     - 14.7M
     - 550k
     - 25.8M
     - 68k
   * - :py:mod:`drive <deepdraw.configs.datasets.drive.default>`
     - 0.788  (0.021)
     - `0.821 (0.014) <baselines_driu_drive_>`_
     - `0.813 (0.016) <baselines_hed_drive_>`_
     - `0.802 (0.014) <baselines_m2unet_drive_>`_
     - `0.825 (0.015) <baselines_unet_drive_>`_
     -  0.828
   * - :py:mod:`stare <deepdraw.configs.datasets.stare.ah>`
     - 0.759 (0.028)
     - `0.828 (0.039) <baselines_driu_stare_>`_
     - `0.815 (0.047) <baselines_hed_stare_>`_
     - `0.818 (0.035) <baselines_m2unet_stare_>`_
     - `0.828 (0.050) <baselines_unet_stare_>`_
     -  0.839
   * - :py:mod:`chasedb1 <deepdraw.configs.datasets.chasedb1.first_annotator>`
     - 0.768 (0.023)
     - `0.812 (0.018) <baselines_driu_chase_>`_
     - `0.806 (0.020) <baselines_hed_chase_>`_
     - `0.798 (0.018) <baselines_m2unet_chase_>`_
     - `0.807 (0.017) <baselines_unet_chase_>`_
     -  0.820
   * - :py:mod:`hrf <deepdraw.configs.datasets.hrf.default>` (1168x1648)
     -
     - `0.808 (0.038) <baselines_driu_hrf_>`_
     - `0.803 (0.040) <baselines_hed_hrf_>`_
     - `0.796 (0.048) <baselines_m2unet_hrf_>`_
     - `0.811 (0.039) <baselines_unet_hrf_>`_
     -  0.814
   * - :py:mod:`hrf <deepdraw.configs.datasets.hrf.default>` (2336x3296)
     -
     - `0.722 (0.073) <baselines_driu_hrf_>`_
     - `0.703 (0.090) <baselines_hed_hrf_>`_
     - `0.713 (0.143) <baselines_m2unet_hrf_>`_
     - `0.756 (0.051) <baselines_unet_hrf_>`_
     -  0.744
   * - :py:mod:`iostar-vessel <deepdraw.configs.datasets.iostar.vessel>`
     -
     - `0.825 (0.020) <baselines_driu_iostar_>`_
     - `0.827 (0.020) <baselines_hed_iostar_>`_
     - `0.820 (0.018) <baselines_m2unet_iostar_>`_
     - `0.818 (0.020) <baselines_unet_iostar_>`_
     -  0.832


Notes
-----

* HRF models were trained using half the full resolution (1168x1648)
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
   * - :py:mod:`drive <deepdraw.configs.datasets.drive.default>`
     - 8
     - 8
     - 16
     - 4
     - 4
   * - :py:mod:`stare <deepdraw.configs.datasets.stare.ah>`
     - 5
     - 4
     - 6
     - 2
     - 4
   * - :py:mod:`chasedb1 <deepdraw.configs.datasets.chasedb1.first_annotator>`
     - 4
     - 4
     - 6
     - 2
     - 4
   * - :py:mod:`hrf <deepdraw.configs.datasets.hrf.default>`
     - 1
     - 1
     - 1
     - 1
     - 4
   * - :py:mod:`iostar-vessel <deepdraw.configs.datasets.iostar.vessel>`
     - 4
     - 4
     - 6
     - 2
     - 4

Results for datasets with (768x768 resolution)

.. list-table::
   :header-rows: 2

   * -
     -
     - :py:mod:`driu <deepdraw.configs.models.driu>`
     - :py:mod:`hed <deepdraw.configs.models.hed>`
     - :py:mod:`m2unet <deepdraw.configs.models.m2unet>`
     - :py:mod:`unet <deepdraw.configs.models.unet>`
     - :py:mod:`lwnet <deepdraw.configs.models.lwnet>`
   * - Dataset
     - 2nd. Annot.
     - 15M
     - 14.7M
     - 550k
     - 25.8M
     - 68k
   * - :py:mod:`drive <deepdraw.configs.datasets.drive.default_768>`
     -
     - 0.812
     - 0.806
     - 0.800
     - 0.814
     - 0.807
   * - :py:mod:`stare <deepdraw.configs.datasets.stare.ah_768>`
     -
     - 0.819
     - 0.812
     - 0.793
     - 0.829
     - 0.817
   * - :py:mod:`chasedb1 <deepdraw.configs.datasets.chasedb1.first_annotator_768>`
     -
     - 0.809
     - 0.790
     - 0.793
     - 0.803
     - 0.797
   * - :py:mod:`hrf <deepdraw.configs.datasets.hrf.default_768>`
     -
     - 0.799
     - 0.774
     - 0.773
     - 0.804
     - 0.800
   * - :py:mod:`iostar-vessel <deepdraw.configs.datasets.iostar.vessel_768>`
     -
     - 0.825
     - 0.818
     - 0.813
     - 0.820
     - 0.820
   * - Combined datasets
     -
     - 0.811
     - 0.798
     - 0.798
     - 0.813
     - 0.804

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
   * - :py:mod:`drive <deepdraw.configs.datasets.drive.default_768>`
     - 8
     - 8
     - 8
     - 4
     - 8
   * - :py:mod:`stare <deepdraw.configs.datasets.stare.ah_768>`
     - 8
     - 8
     - 8
     - 4
     - 8
   * - :py:mod:`chasedb1 <deepdraw.configs.datasets.chasedb1.first_annotator_768>`
     - 8
     - 8
     - 8
     - 4
     - 8
   * - :py:mod:`hrf <deepdraw.configs.datasets.hrf.default_768>`
     - 8
     - 8
     - 8
     - 4
     - 8
   * - :py:mod:`iostar-vessel <deepdraw.configs.datasets.iostar.vessel_768>`
     - 8
     - 8
     - 8
     - 4
     - 8

Results for datasets with (1024x1024 resolution)

.. list-table::
   :header-rows: 2

   * -
     -
     - :py:mod:`driu <deepdraw.configs.models.driu>`
     - :py:mod:`hed <deepdraw.configs.models.hed>`
     - :py:mod:`m2unet <deepdraw.configs.models.m2unet>`
     - :py:mod:`unet <deepdraw.configs.models.unet>`
     - :py:mod:`lwnet <deepdraw.configs.models.lwnet>`
   * - Dataset
     - 2nd. Annot.
     - 15M
     - 14.7M
     - 550k
     - 25.8M
     - 68k
   * - :py:mod:`drive <deepdraw.configs.datasets.drive.default_1024>`
     -
     - 0.813
     - 0.806
     - 0.804
     - 0.815
     - 0.809
   * - :py:mod:`stare <deepdraw.configs.datasets.stare.ah_1024>`
     -
     - 0.821
     - 0.812
     - 0.816
     - 0.820
     - 0.814
   * - :py:mod:`chasedb1 <deepdraw.configs.datasets.chasedb1.first_annotator_1024>`
     -
     - 0.806
     - 0.806
     - 0.790
     - 0.806
     - 0.793
   * - :py:mod:`hrf <deepdraw.configs.datasets.hrf.default_1024>`
     -
     - 0.805
     - 0.793
     - 0.786
     - 0.807
     - 0.805
   * - :py:mod:`iostar-vessel <deepdraw.configs.datasets.iostar.vessel>`
     -
     - 0.829
     - 0.825
     - 0.817
     - 0.825
     - 0.824


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
   * - :py:mod:`drive <deepdraw.configs.datasets.drive.default_1024>`
     - 8
     - 8
     - 8
     - 4
     - 8
   * - :py:mod:`stare <deepdraw.configs.datasets.stare.ah_1024>`
     - 8
     - 8
     - 8
     - 4
     - 8
   * - :py:mod:`chasedb1 <deepdraw.configs.datasets.chasedb1.first_annotator_1024>`
     - 8
     - 8
     - 8
     - 4
     - 8
   * - :py:mod:`hrf <deepdraw.configs.datasets.hrf.default_1024>`
     - 8
     - 8
     - 8
     - 4
     - 8
   * - :py:mod:`iostar-vessel <deepdraw.configs.datasets.iostar.vessel>`
     - 8
     - 8
     - 8
     - 4
     - 8



.. include:: ../../links.rst
