.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.cli:

==============================
 Command-Line Interface (CLI)
==============================

This package provides a single entry point for all of its applications using
:ref:`Bob's unified CLI mechanism <bob.extension.cli>`.  A list of available
applications can be retrieved using:

.. command-output:: bob binseg --help


Setup
-----

A CLI application to list and check installed (raw) datasets.

.. _bob.ip.binseg.cli.dataset:

.. command-output:: bob binseg dataset --help


List available datasets
=======================

Lists supported and configured raw datasets.

.. _bob.ip.binseg.cli.dataset.list:

.. command-output:: bob binseg dataset list --help


Check available datasets
========================

Checks if we can load all files listed for a given dataset (all subsets in all
protocols).

.. _bob.ip.binseg.cli.dataset.check:

.. command-output:: bob binseg dataset check --help


Preset Configuration Resources
------------------------------

A CLI application allows one to list, inspect and copy available configuration
resources exported by this package.

.. _bob.ip.binseg.cli.config:

.. command-output:: bob binseg config --help


.. _bob.ip.binseg.cli.config.list:

Listing Resources
=================

.. command-output:: bob binseg config list --help


.. _bob.ip.binseg.cli.config.list.all:

Available Resources
===================

Here is a list of all resources currently exported.

.. command-output:: bob binseg config list -v


.. _bob.ip.binseg.cli.config.describe:

Describing a Resource
=====================

.. command-output:: bob binseg config describe --help


.. _bob.ip.binseg.cli.config.copy:

Copying a Resource
==================

You may use this command to locally copy a resource file so you can change it.

.. command-output:: bob binseg config copy --help


.. _bob.ip.binseg.cli.combined:

Running and Analyzing Experiments
---------------------------------

These applications run a combined set of steps in one go.  They work well with
our preset :ref:`configuration resources <bob.ip.binseg.cli.config.list.all>`.


.. _bob.ip.binseg.cli.experiment:

Running a Full Experiment Cycle
===============================

This command can run training, prediction, evaluation and comparison from a
single, multi-step application.

.. command-output:: bob binseg experiment --help


.. _bob.ip.binseg.cli.analyze:

Running Complete Experiment Analysis
====================================

This command can run prediction, evaluation and comparison from a
single, multi-step application.

.. command-output:: bob binseg analyze --help


.. _bob.ip.binseg.cli.single:

Single-Step Applications
------------------------

These applications allow finer control over the experiment cycle.  They also
work well with our preset :ref:`configuration resources
<bob.ip.binseg.cli.config.list.all>`, but allow finer control on the input
datasets.


.. _bob.ip.binseg.cli.train:

Training FCNs
=============

Training creates of a new PyTorch_ model.  This model can be used for
evaluation tests or for inference.

.. command-output:: bob binseg train --help


.. _bob.ip.binseg.cli.predict:

Prediction with FCNs
====================

Inference takes as input a PyTorch_ model and generates output probabilities as
HDF5 files.  The probability map has the same size as the input and indicates,
from 0 to 1 (floating-point number), the probability of a vessel in that pixel,
from less probable (0.0) to more probable (1.0).

.. command-output:: bob binseg predict --help


.. _bob.ip.binseg.cli.evaluate:

FCN Performance Evaluation
==========================

Evaluation takes inference results and compares it to ground-truth, generating
a series of analysis figures which are useful to understand model performance.

.. command-output:: bob binseg evaluate --help


.. _bob.ip.binseg.cli.compare:

Performance Comparison
======================

Performance comparison takes the performance evaluation results and generate
combined figures and tables that compare results of multiple systems.

.. command-output:: bob binseg compare --help


.. _bob.ip.binseg.cli.significance:

Performance Difference Significance
===================================

Calculates the significance between results obtained through 2 systems on the
same dataset.

.. command-output:: bob binseg significance --help


.. include:: links.rst
