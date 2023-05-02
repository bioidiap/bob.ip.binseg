.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.results.cod.vessel:

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
   * - Dataset
     - 2nd. Annot.
     - 15M
     - 14.7M
     - 0.55M
     - 25.8M
   * - :py:mod:`drive <deepdraw.configs.datasets.drive.covd>`
     - 0.788 (0.021)
     - `0.768 (0.031) <covd_driu_drive_>`_
     - `0.750 (0.036) <covd_hed_drive_>`_
     - `0.771 (0.027) <covd_m2unet_drive_>`_
     - `0.775 (0.029) <covd_unet_drive_>`_
   * - :py:mod:`stare <deepdraw.configs.datasets.stare.covd>`
     - 0.759 (0.028)
     - `0.786 (0.100) <covd_driu_stare_>`_
     - `0.738 (0.193) <covd_hed_stare_>`_
     - `0.800 (0.080) <covd_m2unet_stare_>`_
     - `0.806 (0.072) <covd_unet_stare_>`_
   * - :py:mod:`chasedb1 <deepdraw.configs.datasets.chasedb1.covd>`
     - 0.768 (0.023)
     - `0.778 (0.031) <covd_driu_chase_>`_
     - `0.777 (0.028) <covd_hed_chase_>`_
     - `0.776 (0.031) <covd_m2unet_chase_>`_
     - `0.779 (0.028) <covd_unet_chase_>`_
   * - :py:mod:`hrf <deepdraw.configs.datasets.hrf.covd>`
     -
     - `0.742 (0.049) <covd_driu_hrf_>`_
     - `0.719 (0.047) <covd_hed_hrf_>`_
     - `0.735 (0.045) <covd_m2unet_hrf_>`_
     - `0.746 (0.046) <covd_unet_hrf_>`_
   * - :py:mod:`iostar-vessel <deepdraw.configs.datasets.iostar.covd>`
     -
     - `0.790 (0.023) <covd_driu_iostar_>`_
     - `0.792 (0.020) <covd_hed_iostar_>`_
     - `0.788 (0.021) <covd_m2unet_iostar_>`_
     - `0.783 (0.019) <covd_unet_iostar_>`_


Notes
-----

* The following table describes recommended batch sizes for 24Gb of RAM GPU
  card, for supervised training of COD-systems:

  .. code-block:: sh

     # change <model> and <dataset> by one of items bellow
     $ bob binseg experiment -vv <model> <dataset> --batch-size=<see-table> --device="cuda:0"

  .. list-table::

    * - **Models / Datasets**
      - :py:mod:`drive-covd <deepdraw.configs.datasets.drive.covd>`
      - :py:mod:`stare-covd <deepdraw.configs.datasets.stare.covd>`
      - :py:mod:`chasedb1-covd <deepdraw.configs.datasets.chasedb1.covd>`
      - :py:mod:`iostar-vessel-covd <deepdraw.configs.datasets.iostar.covd>`
      - :py:mod:`hrf-covd <deepdraw.configs.datasets.hrf.covd>`
    * - :py:mod:`driu <deepdraw.configs.models.driu>` / :py:mod:`driu-bn <deepdraw.configs.models.driu_bn>`
      - 4
      - 4
      - 2
      - 2
      - 2
    * - :py:mod:`m2unet <deepdraw.configs.models.m2unet>`
      - 8
      - 4
      - 4
      - 4
      - 4


.. include:: ../../links.rst
