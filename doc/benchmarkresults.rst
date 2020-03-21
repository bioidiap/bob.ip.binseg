.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.benchmarkresults:

===================
 Benchmark Results
===================

F1 Scores (micro-level)
-----------------------

* Benchmark results for models: DRIU, HED, M2U-Net and U-Net.
* Models are trained and tested on the same dataset using the
  train-test split as indicated in :ref:`bob.ip.binseg.configs.datasets` (i.e.,
  these are *intra*-datasets tests)
* Standard-deviations across all test images are indicated in brakets
* Database and Model links (table top row and left column) are linked to the
  originating configuration files used to obtain these results.
* For some results, the actual deep neural network models are provided (by
  clicking on the associated F1 Score).
* Check `our paper`_ for details on the calculation of the F1 Score and standard
  deviations.

.. list-table::
   :header-rows: 1

   * - F1 (std)
     - :py:mod:`DRIU <bob.ip.binseg.configs.models.driu>`
     - :py:mod:`HED <bob.ip.binseg.configs.models.hed>`
     - :py:mod:`M2U-Net <bob.ip.binseg.configs.models.m2unet>`
     - :py:mod:`U-Net <bob.ip.binseg.configs.models.unet>`
   * - :py:mod:`CHASE-DB1 <bob.ip.binseg.configs.datasets.chasedb1>`
     - `0.810 (0.021) <driu_chasedb1.pth_>`_
     - 0.810 (0.022)
     - `0.802 (0.019) <m2unet_chasedb1.pth_>`_
     - 0.812 (0.020)
   * - :py:mod:`DRIVE <bob.ip.binseg.configs.datasets.drive>`
     - `0.820 (0.014) <driu_drive.pth_>`_
     - 0.817 (0.013)
     - `0.803 (0.014) <m2unet_drive.pth_>`_
     - 0.822 (0.015)
   * - :py:mod:`HRF <bob.ip.binseg.configs.datasets.hrf1168>`
     - `0.783 (0.055) <driu_hrf.pth_>`_
     - 0.783 (0.058)
     - `0.780 (0.057) <m2unet_hrf.pth_>`_
     - 0.788 (0.051)
   * - :py:mod:`IOSTAR (vessel) <bob.ip.binseg.configs.datasets.iostarvessel>`
     - `0.825 (0.020) <driu_iostar.pth_>`_
     - 0.825 (0.020)
     - `0.817 (0.020) <m2unet_iostar.pth_>`_
     - 0.818 (0.019)
   * - :py:mod:`STARE <bob.ip.binseg.configs.datasets.stare>`
     - `0.827 (0.037) <driu_stare.pth_>`_
     - 0.823 (0.037)
     - `0.815 (0.041) <m2unet_stare.pth_>`_
     - 0.829 (0.042)


.. include:: links.rst
