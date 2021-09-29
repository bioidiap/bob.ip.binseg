.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.results.baselines.vessel:

==============================================
 Retinal Vessel Segmentation for Retinography
==============================================


.. list-table::
   :header-rows: 2

   * -
     -
     - :py:mod:`driu <bob.ip.binseg.configs.models.driu>`
     - :py:mod:`hed <bob.ip.binseg.configs.models.hed>`
     - :py:mod:`m2unet <bob.ip.binseg.configs.models.m2unet>`
     - :py:mod:`unet <bob.ip.binseg.configs.models.unet>`
     - :py:mod:`lwnet <bob.ip.binseg.configs.models.lwnet>`
   * - Dataset
     - 2nd. Annot.
     - 15M
     - 14.7M
     - 550k
     - 25.8M
     - 68k
   * - :py:mod:`drive <bob.ip.binseg.configs.datasets.drive.default>`
     - 0.788  (0.021)
     - `0.821 (0.014) <baselines_driu_drive_>`_
     - `0.813 (0.016) <baselines_hed_drive_>`_
     - `0.802 (0.014) <baselines_m2unet_drive_>`_
     - `0.825 (0.015) <baselines_unet_drive_>`_
     -  0.828
   * - :py:mod:`stare <bob.ip.binseg.configs.datasets.stare.ah>`
     - 0.759 (0.028)
     - `0.828 (0.039) <baselines_driu_stare_>`_
     - `0.815 (0.047) <baselines_hed_stare_>`_
     - `0.818 (0.035) <baselines_m2unet_stare_>`_
     - `0.828 (0.050) <baselines_unet_stare_>`_
     -  0.839
   * - :py:mod:`chasedb1 <bob.ip.binseg.configs.datasets.chasedb1.first_annotator>`
     - 0.768 (0.023)
     - `0.812 (0.018) <baselines_driu_chase_>`_
     - `0.806 (0.020) <baselines_hed_chase_>`_
     - `0.798 (0.018) <baselines_m2unet_chase_>`_
     - `0.807 (0.017) <baselines_unet_chase_>`_
     -  0.820
   * - :py:mod:`hrf <bob.ip.binseg.configs.datasets.hrf.default>` (1168x1648)
     -
     - `0.808 (0.038) <baselines_driu_hrf_>`_
     - `0.803 (0.040) <baselines_hed_hrf_>`_
     - `0.796 (0.048) <baselines_m2unet_hrf_>`_
     - `0.811 (0.039) <baselines_unet_hrf_>`_
     -  0.814
   * - :py:mod:`hrf <bob.ip.binseg.configs.datasets.hrf.default>` (2336x3296)
     -
     - `0.722 (0.073) <baselines_driu_hrf_>`_
     - `0.703 (0.090) <baselines_hed_hrf_>`_
     - `0.713 (0.143) <baselines_m2unet_hrf_>`_
     - `0.756 (0.051) <baselines_unet_hrf_>`_
     -  0.744
   * - :py:mod:`iostar-vessel <bob.ip.binseg.configs.datasets.iostar.vessel>`
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

    * - **Models / Datasets**
      - :py:mod:`drive <bob.ip.binseg.configs.datasets.drive.default>`
      - :py:mod:`stare <bob.ip.binseg.configs.datasets.stare.ah>`
      - :py:mod:`chasedb1 <bob.ip.binseg.configs.datasets.chasedb1.first_annotator>`
      - :py:mod:`iostar-vessel <bob.ip.binseg.configs.datasets.iostar.vessel>`
      - :py:mod:`hrf <bob.ip.binseg.configs.datasets.hrf.default>`
    * - :py:mod:`unet <bob.ip.binseg.configs.models.unet>`
      - 4
      - 2
      - 2
      - 2
      - 1
    * - :py:mod:`hed <bob.ip.binseg.configs.models.hed>`
      - 8
      - 4
      - 4
      - 4
      - 1
    * - :py:mod:`driu <bob.ip.binseg.configs.models.driu>` / :py:mod:`driu-bn <bob.ip.binseg.configs.models.driu_bn>`
      - 8
      - 5
      - 4
      - 4
      - 1
    * - :py:mod:`m2unet <bob.ip.binseg.configs.models.m2unet>`
      - 16
      - 6
      - 6
      - 6
      - 1


.. include:: ../../links.rst
