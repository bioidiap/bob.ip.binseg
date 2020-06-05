.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.results.covd:

========================================
 Combined Vessel Dataset (COVD) Results
========================================


F1 Scores (micro-level)
-----------------------

* Benchmark results for models: DRIU, HED, M2U-Net and U-Net.
* Models are trained on a COVD **excluding** the target dataset, and tested on
  the target dataset (**numbers in bold** indicate number of parameters per
  model).  Models are trained for a fixed number of 1000 epochs, with a
  learning rate of 0.001 until epoch 900 and then 0.0001 until the end of the
  training.
* Database and model resource configuration links (table top row and left
  column) are linked to the originating configuration files used to obtain
  these results.
* Check `our paper`_ for details on the calculation of the F1 Score and standard
  deviations (in parentheses).
* Single performance numbers correspond to *a priori* performance indicators,
  where the threshold is previously selected on the training set (COVD
  excluding the target dataset)
* You can cross check the analysis numbers provided in this table by
  downloading this software package, the raw data, and running ``bob binseg
  analyze`` providing the model URL as ``--weight`` parameter.
* For comparison purposes, we provide "second-annotator" performances on the
  same test set, where available.


.. list-table::
   :header-rows: 2

   * -
     -
     - :py:mod:`driu <bob.ip.binseg.configs.models.driu>`
     - :py:mod:`hed <bob.ip.binseg.configs.models.hed>`
     - :py:mod:`m2unet <bob.ip.binseg.configs.models.m2unet>`
     - :py:mod:`unet <bob.ip.binseg.configs.models.unet>`
   * - Dataset
     - 2nd. Annot.
     - 15M
     - 14.7M
     - 0.55M
     - 25.8M
   * - :py:mod:`drive <bob.ip.binseg.configs.datasets.drive.covd>`
     - 0.788 (0.021)
     - `0.768 (0.031) <covd_driu_drive_>`_
     - `0.750 (0.036) <covd_hed_drive_>`_
     - `0.771 (0.027) <covd_m2unet_drive_>`_
     - `0.775 (0.029) <covd_unet_drive_>`_
   * - :py:mod:`stare <bob.ip.binseg.configs.datasets.stare.covd>`
     - 0.759 (0.028)
     - `0.786 (0.100) <covd_driu_stare_>`_
     - `0.738 (0.193) <covd_hed_stare_>`_
     - `0.800 (0.080) <covd_m2unet_stare_>`_
     - `0.806 (0.072) <covd_unet_stare_>`_
   * - :py:mod:`chasedb1 <bob.ip.binseg.configs.datasets.chasedb1.covd>`
     - 0.768 (0.023)
     - `0.778 (0.031) <covd_driu_chase_>`_
     - `0.777 (0.028) <covd_hed_chase_>`_
     - `0.776 (0.031) <covd_m2unet_chase_>`_
     - `0.779 (0.028) <covd_unet_chase_>`_
   * - :py:mod:`hrf <bob.ip.binseg.configs.datasets.hrf.covd>`
     -
     - `0.742 (0.049) <covd_driu_hrf_>`_
     - `0.719 (0.047) <covd_hed_hrf_>`_
     - `0.735 (0.045) <covd_m2unet_hrf_>`_
     - `0.746 (0.046) <covd_unet_hrf_>`_
   * - :py:mod:`iostar-vessel <bob.ip.binseg.configs.datasets.iostar.covd>`
     -
     - `0.790 (0.023) <covd_driu_iostar_>`_
     - `0.792 (0.020) <covd_hed_iostar_>`_
     - `0.788 (0.021) <covd_m2unet_iostar_>`_
     - `0.783 (0.019) <covd_unet_iostar_>`_


.. include:: ../../links.rst
