.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.results.baselines:

===================
 Baseline Results
===================

F1 Scores (micro-level)
-----------------------

* Benchmark results for models: DRIU, HED, M2U-Net, U-Net, and Little W-Net.
* Models are trained and tested on the same dataset (**numbers in bold**
  indicate approximate number of parameters per model). DRIU, HED, M2U-Net and
  U-Net Models are trained for a fixed number of 1000 epochs, with a learning
  rate of 0.001 until epoch 900 and then 0.0001 until the end of the training,
  after being initialized with a VGG-16 backend.  Little W-Net models are
  trained using a cosine anneling strategy (see [GALDRAN-2020]_ and
  [SMITH-2017]_) for 2000 epochs.
* During the training session, an unaugmented copy of the training set is used
  as validation set.  We keep checkpoints for the best performing networks
  based on such validation set.  The best performing network during training is
  used for evaluation.
* Image masks are used during the evaluation, errors are only assessed within
  the masked region.
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
* :ref:`Our baseline script <bob.ip.binseg.baseline-script>` was used to
  generate the results displayed here.
* HRF models were trained using half the full resolution (1168x1648)


.. list-table::
   :header-rows: 2

   * -
     -
     - :py:mod:`driu <bob.ip.binseg.configs.models.driu>`
     - :py:mod:`hed <bob.ip.binseg.configs.models.hed>`
     - :py:mod:`m2unet <bob.ip.binseg.configs.models.m2unet>`
     - :py:mod:`unet <bob.ip.binseg.configs.models.unet>`
     - lwnet?
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
     -
   * - :py:mod:`stare <bob.ip.binseg.configs.datasets.stare.ah>`
     - 0.759 (0.028)
     - `0.828 (0.039) <baselines_driu_stare_>`_
     - `0.815 (0.047) <baselines_hed_stare_>`_
     - `0.818 (0.035) <baselines_m2unet_stare_>`_
     - `0.828 (0.050) <baselines_unet_stare_>`_
     -
   * - :py:mod:`chasedb1 <bob.ip.binseg.configs.datasets.chasedb1.first_annotator>`
     - 0.768 (0.023)
     - `0.812 (0.018) <baselines_driu_chase_>`_
     - `0.806 (0.020) <baselines_hed_chase_>`_
     - `0.798 (0.018) <baselines_m2unet_chase_>`_
     - `0.807 (0.017) <baselines_unet_chase_>`_
     -
   * - :py:mod:`hrf <bob.ip.binseg.configs.datasets.hrf.default>` (1168x1648)
     -
     - `0.808 (0.038) <baselines_driu_hrf_>`_
     - `0.803 (0.040) <baselines_hed_hrf_>`_
     - `0.796 (0.048) <baselines_m2unet_hrf_>`_
     - `0.811 (0.039) <baselines_unet_hrf_>`_
     -
   * - :py:mod:`hrf <bob.ip.binseg.configs.datasets.hrf.default>` (2336x3296)
     -
     - `0.722 (0.073) <baselines_driu_hrf_>`_
     - `0.703 (0.090) <baselines_hed_hrf_>`_
     - `0.713 (0.143) <baselines_m2unet_hrf_>`_
     - `0.756 (0.051) <baselines_unet_hrf_>`_
     -
   * - :py:mod:`iostar-vessel <bob.ip.binseg.configs.datasets.iostar.vessel>`
     -
     - `0.825 (0.020) <baselines_driu_iostar_>`_
     - `0.827 (0.020) <baselines_hed_iostar_>`_
     - `0.820 (0.018) <baselines_m2unet_iostar_>`_
     - `0.818 (0.020) <baselines_unet_iostar_>`_
     -

Precision-Recall (PR) Curves
----------------------------

Next, you will find the PR plots showing confidence intervals, for the various
models explored, on a per dataset arrangement.  All curves correspond to test
set performances.  Single performance figures (F1-micro scores) correspond to
its average value across all test set images, for a fixed threshold set to
``0.5``, and using 1000 points for curve calculation.

.. tip:: **Curve Intepretation**

   PR curves behave differently than traditional ROC curves (using Specificity
   versus Sensitivity) with respect to the overall shape.  You may have a look
   at [DAVIS-2006]_ for details on the relationship between PR and ROC curves.
   For example, PR curves are not guaranteed to be monotonically increasing or
   decreasing with the scanned thresholds.

   Each evaluated threshold in a combination of trained models and datasets is
   represented by a point in each curve.  Points are linearly interpolated to
   created a line.  For each evaluated threshold and every trained model and
   dataset, we assume that the standard deviation on both precision and recall
   estimation represent good proxies for the uncertainty around that point.  We
   therefore plot a transparent ellipse centered around each evaluated point in
   which the width corresponds to twice the recall standard deviation and the
   height, twice the precision standard deviation.


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
           :alt: Model comparisons for hrf datasets (matching training resolution: 1168x1648)

           :py:mod:`hrf <bob.ip.binseg.configs.datasets.hrf.default>` (1168x1648): PR curve and F1 scores at T=0.5 (:download:`pdf <hrf.pdf>`)
    * - .. figure:: iostar-vessel.png
           :align: center
           :scale: 50%
           :alt: Model comparisons for iostar-vessel datasets

           :py:mod:`iostar-vessel <bob.ip.binseg.configs.datasets.iostar.vessel>`: PR curve and F1 scores at T=0.5 (:download:`pdf <iostar-vessel.pdf>`)
      - .. figure:: hrf-fullres.png
           :align: center
           :scale: 50%
           :alt: Model comparisons for hrf datasets (double training resolution: 2336x3296)

           :py:mod:`hrf <bob.ip.binseg.configs.datasets.hrf.default>` (2336x3296): PR curve and F1 scores at T=0.5 (:download:`pdf <hrf-fullres.pdf>`)


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
  Unfortunately, this is not conclusive.
* Training at half resolution for HRF shows a small loss in performance (10 to
  15%) when the high-resolution version is used as evaluation set.


.. include:: ../../links.rst
