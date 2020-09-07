.. -*- coding: utf-8 -*-

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

   bob.ip.binseg.data.dataset
   bob.ip.binseg.data.loader
   bob.ip.binseg.data.sample
   bob.ip.binseg.data.utils
   bob.ip.binseg.data.transforms


Datasets
--------

.. autosummary::
   :toctree: api/dataset

   bob.ip.binseg.data.drive
   bob.ip.binseg.data.stare
   bob.ip.binseg.data.chasedb1
   bob.ip.binseg.data.hrf
   bob.ip.binseg.data.iostar
   bob.ip.binseg.data.refuge
   bob.ip.binseg.data.drishtigs1
   bob.ip.binseg.data.rimoner3
   bob.ip.binseg.data.drionsdb


Engines
-------

.. autosummary::
   :toctree: api/engine

   bob.ip.binseg.engine
   bob.ip.binseg.engine.trainer
   bob.ip.binseg.engine.ssltrainer
   bob.ip.binseg.engine.predictor
   bob.ip.binseg.engine.evaluator
   bob.ip.binseg.engine.adabound


Neural Network Models
---------------------

.. autosummary::
   :toctree: api/models

   bob.ip.binseg.models
   bob.ip.binseg.models.backbones
   bob.ip.binseg.models.backbones.mobilenetv2
   bob.ip.binseg.models.backbones.resnet
   bob.ip.binseg.models.backbones.vgg
   bob.ip.binseg.models.normalizer
   bob.ip.binseg.models.driu
   bob.ip.binseg.models.driu_bn
   bob.ip.binseg.models.driu_od
   bob.ip.binseg.models.driu_pix
   bob.ip.binseg.models.hed
   bob.ip.binseg.models.m2unet
   bob.ip.binseg.models.resunet
   bob.ip.binseg.models.unet
   bob.ip.binseg.models.lwnet
   bob.ip.binseg.models.losses
   bob.ip.binseg.models.make_layers


Toolbox
-------

.. autosummary::
   :toctree: api/utils

   bob.ip.binseg.utils
   bob.ip.binseg.utils.checkpointer
   bob.ip.binseg.utils.measure
   bob.ip.binseg.utils.plot
   bob.ip.binseg.utils.table
   bob.ip.binseg.utils.summary


.. _bob.ip.binseg.configs:

Preset Configurations
---------------------

Preset configurations for baseline systems

This module contains preset configurations for baseline FCN architectures and
datasets.


Models
======

.. autosummary::
   :toctree: api/configs/models
   :template: config.rst

   bob.ip.binseg.configs.models.driu
   bob.ip.binseg.configs.models.driu_bn
   bob.ip.binseg.configs.models.driu_bn_ssl
   bob.ip.binseg.configs.models.driu_od
   bob.ip.binseg.configs.models.driu_ssl
   bob.ip.binseg.configs.models.hed
   bob.ip.binseg.configs.models.m2unet
   bob.ip.binseg.configs.models.m2unet_ssl
   bob.ip.binseg.configs.models.resunet
   bob.ip.binseg.configs.models.unet


.. _bob.ip.binseg.configs.datasets:

Datasets
========

.. automodule:: bob.ip.binseg.configs.datasets

.. autosummary::
   :toctree: api/configs/datasets
   :template: config.rst

   bob.ip.binseg.configs.datasets.csv

   bob.ip.binseg.configs.datasets.chasedb1.first_annotator
   bob.ip.binseg.configs.datasets.chasedb1.second_annotator
   bob.ip.binseg.configs.datasets.chasedb1.xtest
   bob.ip.binseg.configs.datasets.chasedb1.mtest
   bob.ip.binseg.configs.datasets.chasedb1.covd
   bob.ip.binseg.configs.datasets.chasedb1.ssl

   bob.ip.binseg.configs.datasets.drive.default
   bob.ip.binseg.configs.datasets.drive.second_annotator
   bob.ip.binseg.configs.datasets.drive.xtest
   bob.ip.binseg.configs.datasets.drive.mtest
   bob.ip.binseg.configs.datasets.drive.covd
   bob.ip.binseg.configs.datasets.drive.ssl

   bob.ip.binseg.configs.datasets.hrf.default
   bob.ip.binseg.configs.datasets.hrf.xtest
   bob.ip.binseg.configs.datasets.hrf.mtest
   bob.ip.binseg.configs.datasets.hrf.default_fullres
   bob.ip.binseg.configs.datasets.hrf.covd
   bob.ip.binseg.configs.datasets.hrf.ssl

   bob.ip.binseg.configs.datasets.iostar.vessel
   bob.ip.binseg.configs.datasets.iostar.vessel_xtest
   bob.ip.binseg.configs.datasets.iostar.vessel_mtest
   bob.ip.binseg.configs.datasets.iostar.optic_disc
   bob.ip.binseg.configs.datasets.iostar.covd
   bob.ip.binseg.configs.datasets.iostar.ssl

   bob.ip.binseg.configs.datasets.stare.ah
   bob.ip.binseg.configs.datasets.stare.vk
   bob.ip.binseg.configs.datasets.stare.xtest
   bob.ip.binseg.configs.datasets.stare.mtest
   bob.ip.binseg.configs.datasets.stare.covd
   bob.ip.binseg.configs.datasets.stare.ssl

   bob.ip.binseg.configs.datasets.refuge.cup
   bob.ip.binseg.configs.datasets.refuge.disc

   bob.ip.binseg.configs.datasets.rimoner3.cup_exp1
   bob.ip.binseg.configs.datasets.rimoner3.cup_exp2
   bob.ip.binseg.configs.datasets.rimoner3.disc_exp1
   bob.ip.binseg.configs.datasets.rimoner3.disc_exp2

   bob.ip.binseg.configs.datasets.drishtigs1.cup_all
   bob.ip.binseg.configs.datasets.drishtigs1.cup_any
   bob.ip.binseg.configs.datasets.drishtigs1.disc_all
   bob.ip.binseg.configs.datasets.drishtigs1.disc_any

   bob.ip.binseg.configs.datasets.drionsdb.expert1
   bob.ip.binseg.configs.datasets.drionsdb.expert2


.. include:: links.rst
