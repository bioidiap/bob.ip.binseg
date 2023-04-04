.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.api:

=====================================
 Application Program Interface (API)
=====================================

.. To update these lists, run the following command on the root of the package:
.. find bob -name '*.py' | sed -e 's#/#.#g;s#.py$##g;s#.__init__##g' | sort
.. You may apply further filtering to update only one of the subsections below


Data Manipulation
-----------------

.. autosummary::
   :toctree: api/data

   deepdraw.data.dataset
   deepdraw.data.loader
   deepdraw.data.sample
   deepdraw.data.utils
   deepdraw.data.transforms


Datasets
--------

Retinography
============

.. autosummary::
   :toctree: api/dataset

   deepdraw.data.drive
   deepdraw.data.stare
   deepdraw.data.chasedb1
   deepdraw.data.hrf
   deepdraw.data.iostar
   deepdraw.data.refuge
   deepdraw.data.drishtigs1
   deepdraw.data.rimoner3
   deepdraw.data.drionsdb
   deepdraw.data.drhagis


Chest X-Ray
===========

.. autosummary::
   :toctree: api/dataset

   deepdraw.data.montgomery
   deepdraw.data.jsrt
   deepdraw.data.shenzhen
   deepdraw.data.cxr8


Engines
-------

.. autosummary::
   :toctree: api/engine

   deepdraw.engine
   deepdraw.engine.trainer
   deepdraw.engine.predictor
   deepdraw.engine.evaluator
   deepdraw.engine.adabound


Neural Network Models
---------------------

.. autosummary::
   :toctree: api/models

   deepdraw.models
   deepdraw.models.backbones
   deepdraw.models.backbones.mobilenetv2
   deepdraw.models.backbones.resnet
   deepdraw.models.backbones.vgg
   deepdraw.models.normalizer
   deepdraw.models.driu
   deepdraw.models.driu_bn
   deepdraw.models.driu_od
   deepdraw.models.driu_pix
   deepdraw.models.hed
   deepdraw.models.m2unet
   deepdraw.models.resunet
   deepdraw.models.unet
   deepdraw.models.lwnet
   deepdraw.models.losses
   deepdraw.models.make_layers


Toolbox
-------

.. autosummary::
   :toctree: api/utils

   deepdraw.utils
   deepdraw.utils.checkpointer
   deepdraw.utils.measure
   deepdraw.utils.plot
   deepdraw.utils.table
   deepdraw.utils.summary


.. include:: links.rst
