.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.eval:


==========================
 Inference and Evaluation
==========================

This guides explains how to run inference or a complete evaluation using
command-line tools.  Inference produces probability of TB presence for input
images, while evaluation will analyze such output against existing annotations
and produce performance figures.


Inference
---------

In inference (or prediction) mode, we input data, the trained model, and output
a CSV file containing the prediction outputs for every input image.

To run inference, use the sub-command :ref:`predict <deepdraw.cli>` to run
prediction on an existing dataset:

.. code:: sh

   binseg predict -vv <model> -w <path/to/model.pth> <dataset>


Replace ``<model>`` and ``<dataset>`` by the appropriate :ref:`configuration
files <deepdraw.config>`.  Replace ``<path/to/model.pth>`` to a path leading to
the pre-trained model.


Evaluation
----------

In evaluation, we input a dataset and predictions to generate performance
summaries that help analysis of a trained model.  Evaluation is done using the
:ref:`evaluate command <deepdraw.cli>` followed by the model and the annotated
dataset configuration, and the path to the pretrained weights via the
``--weight`` argument.

Use ``binseg evaluate --help`` for more information.

E.g. run evaluation on predictions from the Montgomery set, do the following:

.. code:: sh

   binseg evaluate -vv montgomery -p /predictions/folder -o /eval/results/folder


Comparing Systems
-----------------

To compare multiple systems together and generate combined plots and tables,
use the :ref:`compare command <deepdraw.cli>`.  Use ``--help`` for a quick
guide.

.. code:: sh

   binseg compare -vv A A/metrics.csv B B/metrics.csv --output-figure=plot.pdf --output-table=table.txt --threshold=0.5


.. include:: ../links.rst
