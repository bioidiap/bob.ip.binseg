.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.results.baselines:

===================
 Baseline Results
===================

F1 Scores (micro-level)
-----------------------

* Benchmark results for models: DRIU, HED, M2U-Net and U-Net.
* Models are trained and tested on the same dataset (**numbers in bold**
  indicate number of parameters per model).  Models are trained for a fixed
  number of 1000 epochs, with a learning rate of 0.001 until epoch 900 and then
  0.0001 until the end of the training.
* Database and model resource configuration links (table top row and left
  column) are linked to the originating configuration files used to obtain
  these results.
* Check `our paper`_ for details on the calculation of the F1 Score and standard
  deviations (in parentheses).
* Single performance numbers correspond to *a priori* performance indicators,
  where the threshold is previously selected on the training set
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
   * - :py:mod:`drive <bob.ip.binseg.configs.datasets.drive.default>`
     - 0.788  (0.021)
     - `0.819 (0.017) <baselines_driu_drive_>`_
     - `0.806 (0.017) <baselines_hed_drive_>`_
     - `0.803 (0.017) <baselines_m2unet_drive_>`_
     - `0.823 (0.016) <baselines_unet_drive_>`_
   * - :py:mod:`stare <bob.ip.binseg.configs.datasets.stare.ah>`
     - 0.759 (0.028)
     - `0.822 (0.037) <baselines_driu_stare_>`_
     - `0.808 (0.046) <baselines_hed_stare_>`_
     - `0.811 (0.039) <baselines_m2unet_stare_>`_
     - `0.827 (0.041) <baselines_unet_stare_>`_
   * - :py:mod:`chasedb1 <bob.ip.binseg.configs.datasets.chasedb1.first_annotator>`
     - 0.768 (0.023)
     - `0.810 (0.017) <baselines_driu_chase_>`_
     - `0.806 (0.021) <baselines_hed_chase_>`_
     - `0.798 (0.017) <baselines_m2unet_chase_>`_
     - `0.803 (0.015) <baselines_unet_chase_>`_
   * - :py:mod:`hrf <bob.ip.binseg.configs.datasets.hrf.default>`
     -
     - `0.802 (0.039) <baselines_driu_hrf_>`_
     - `0.793 (0.041) <baselines_hed_hrf_>`_
     - `0.785 (0.041) <baselines_m2unet_hrf_>`_
     - `0.797 (0.038) <baselines_unet_hrf_>`_
   * - :py:mod:`iostar-vessel <bob.ip.binseg.configs.datasets.iostar.vessel>`
     -
     - `0.823 (0.021) <baselines_driu_iostar_>`_
     - `0.821 (0.022) <baselines_hed_iostar_>`_
     - `0.816 (0.021) <baselines_m2unet_iostar_>`_
     - `0.818 (0.019) <baselines_unet_iostar_>`_


Precision-Recall (PR) Curves
----------------------------

Next, you will find the PR plots showing confidence intervals, for the various
models explored, on a per dataset arrangement.  All curves correspond to test
set performances.  Single performance figures (F1-micro scores) correspond to
its average value across all test set images, for a fixed threshold set to
``0.5``.

.. list-table::

    * - .. figure:: drive.png
           :align: center
           :scale: 50%
           :alt: Model comparisons for drive datasets

           :py:mod:`drive <bob.ip.binseg.configs.datasets.drive.default>`: PR curve and F1 scores at T=0.5 (:download:`pdf <drive.pdf>`)
      - .. figure:: stare.png
           :align: center
           :scale: 50%
           :alt: Model comparisons for stare datasets

           :py:mod:`stare <bob.ip.binseg.configs.datasets.stare.ah>`: PR curve and F1 scores at T=0.5 (:download:`pdf <stare.pdf>`)
    * - .. figure:: chasedb1.png
           :align: center
           :scale: 50%
           :alt: Model comparisons for chasedb1 datasets

           :py:mod:`chasedb1 <bob.ip.binseg.configs.datasets.chasedb1.first_annotator>`: PR curve and F1 scores at T=0.5 (:download:`pdf <chasedb1.pdf>`)
      - .. figure:: hrf.png
           :align: center
           :scale: 50%
           :alt: Model comparisons for hrf datasets

           :py:mod:`hrf <bob.ip.binseg.configs.datasets.hrf.default>`: PR curve and F1 scores at T=0.5 (:download:`pdf <hrf.pdf>`)
    * - .. figure:: iostar-vessel.png
           :align: center
           :scale: 50%
           :alt: Model comparisons for iostar-vessel datasets

           :py:mod:`iostar-vessel <bob.ip.binseg.configs.datasets.iostar.vessel>`: PR curve and F1 scores at T=0.5 (:download:`pdf <iostar-vessel.pdf>`)
      -


Remarks
-------

* There seems to be no clear winner as confidence intervals based on the
  standard deviation overlap substantially between the different models, and
  across different datasets.
* There seems to be almost no effect on the number of parameters on
  performance.  U-Net, the largest model, is not a clear winner through all
  baseline benchmarks
* Where second annotator labels exist, model performance and variability seems
  on par with such annotations.  One possible exception is for CHASE-DB1, where
  models show consistently less variability than the second annotator.
  Unfortunately, this cannot be conclusive.

.. include:: ../../links.rst
