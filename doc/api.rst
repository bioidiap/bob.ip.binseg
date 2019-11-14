.. -*- coding: utf-8 -*-
.. _bob.ip.binseg.api:

============
 Python API
============

This section lists all the functionality available in this library allowing to
run binary-segmentation benchmarks.


PyTorch bob.db Dataset
======================
.. automodule:: bob.ip.binseg.data.binsegdataset

PyTorch ImageFolder Dataset
===========================
.. automodule:: bob.ip.binseg.data.imagefolder

.. automodule:: bob.ip.binseg.data.imagefolderinference

Transforms
==========
.. note::
    All transforms work with :py:class:`PIL.Image.Image` objects. We make heavy use of the
    `torchvision package`_

.. automodule:: bob.ip.binseg.data.transforms

Losses
======
.. automodule:: bob.ip.binseg.modeling.losses

Training
========
.. automodule:: bob.ip.binseg.engine.trainer

Checkpointer
============
.. automodule:: bob.ip.binseg.utils.checkpointer

Inference and Evaluation
========================
.. automodule:: bob.ip.binseg.engine.inferencer

Plotting
========
.. automodule:: bob.ip.binseg.utils.plot

.. include:: links.rst
