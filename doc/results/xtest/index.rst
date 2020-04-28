.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.results.xtest:

======================
 Cross-Database Tests
======================

F1 Scores (micro-level)
-----------------------

* Benchmark results for models: DRIU, HED, M2U-Net and U-Net.
* Models are trained and tested on the same dataset (numbers in parenthesis
  indicate number of parameters per model), and then evaluated across the test
  sets of other datasets.
* You can cross check the analysis numbers provided in this table by
  downloading this software package, the raw data, and running ``bob binseg
  analyze`` providing the model URL as ``--weight`` parameter, and then the
  ``-xtest`` resource variant of the dataset the model was trained on.  For
  example, to run cross-evaluation tests for the DRIVE dataset, use the
  configuration resource :py:mod:`drive-xtest
  <bob.ip.binseg.configs.datasets.drive.xtest>`.  Otherwise, we
  also provide `CSV files
  <https://www.idiap.ch/software/bob/data/bob/bob.ip.binseg/master/xtest/>`_
  with the estimated performance per threshold (100 steps) per subset.
* For comparison purposes, we provide "second-annotator" performances on the
  same test set, where available.
* We only show results for DRIU (~15.4 million parameters) and M2U-Net (~550
  thousand parameters) as these models seem to represent the performance
  extremes according to our :ref:`baseline analysis
  <bob.ip.binseg.results.baselines>`.  You may run analysis on the other models
  by downloading them from our website (via the ``--weight`` parameter on the
  :ref:`analyze script <bob.ip.binseg.cli.analyze>`).  This script may help you
  in this task, provided you created a directory structure as suggested by
  :ref:`our baseline script <bob.ip.binseg.baseline-script>`:

  .. literalinclude:: ../../scripts/xtest.sh
     :language: bash


DRIU
====


.. list-table::
   :header-rows: 1

   * - Model / X-Test
     - :py:mod:`drive <bob.ip.binseg.configs.datasets.drive.xtest>`
     - :py:mod:`stare <bob.ip.binseg.configs.datasets.stare.xtest>`
     - :py:mod:`chasedb1 <bob.ip.binseg.configs.datasets.chasedb1.xtest>`
     - :py:mod:`hrf <bob.ip.binseg.configs.datasets.hrf.xtest>`
     - :py:mod:`iostar-vessel <bob.ip.binseg.configs.datasets.iostar.vessel_xtest>`
   * - `drive <baselines_driu_drive_>`_
     -
     -
     -
     -
     -
   * - `stare <baselines_driu_stare_>`_
     -
     -
     -
     -
     -
   * - `chasedb1 <baselines_driu_chase_>`_
     -
     -
     -
     -
     -
   * - `hrf <baselines_driu_hrf_>`_
     -
     -
     -
     -
     -
   * - `iostar-vessel <baselines_driu_iostar_>`_
     -
     -
     -
     -
     -


Precision-Recall (PR) Curves
----------------------------


.. include:: ../../links.rst
