.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.binseg.api:

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

   deepdraw.common.data.dataset
   deepdraw.common.data.loader
   deepdraw.common.data.sample
   deepdraw.common.data.utils
   deepdraw.common.data.transforms


Datasets
--------

Retinography
============

.. autosummary::
   :toctree: api/dataset

   deepdraw.binseg.data.drive
   deepdraw.binseg.data.stare
   deepdraw.binseg.data.chasedb1
   deepdraw.binseg.data.hrf
   deepdraw.binseg.data.iostar
   deepdraw.binseg.data.refuge
   deepdraw.binseg.data.drishtigs1
   deepdraw.binseg.data.rimoner3
   deepdraw.binseg.data.drionsdb
   deepdraw.binseg.data.drhagis


Chest X-Ray
===========

.. autosummary::
   :toctree: api/dataset

   deepdraw.binseg.data.montgomery
   deepdraw.binseg.data.jsrt
   deepdraw.binseg.data.shenzhen
   deepdraw.binseg.data.cxr8


Engines
-------

.. autosummary::
   :toctree: api/engine

   deepdraw.binseg.engine
   deepdraw.binseg.engine.trainer
   deepdraw.binseg.engine.predictor
   deepdraw.binseg.engine.evaluator
   deepdraw.binseg.engine.adabound


Neural Network Models
---------------------

.. autosummary::
   :toctree: api/models

   deepdraw.binseg.models
   deepdraw.binseg.models.backbones
   deepdraw.binseg.models.backbones.mobilenetv2
   deepdraw.binseg.models.backbones.resnet
   deepdraw.binseg.models.backbones.vgg
   deepdraw.binseg.models.normalizer
   deepdraw.binseg.models.driu
   deepdraw.binseg.models.driu_bn
   deepdraw.binseg.models.driu_od
   deepdraw.binseg.models.driu_pix
   deepdraw.binseg.models.hed
   deepdraw.binseg.models.m2unet
   deepdraw.binseg.models.resunet
   deepdraw.binseg.models.unet
   deepdraw.binseg.models.lwnet
   deepdraw.binseg.models.losses
   deepdraw.binseg.models.make_layers


Toolbox
-------

.. autosummary::
   :toctree: api/utils

   deepdraw.common.utils
   deepdraw.common.utils.checkpointer
   deepdraw.common.utils.measure
   deepdraw.common.utils.plot
   deepdraw.common.utils.table
   deepdraw.common.utils.summary


.. include:: links.rst
