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

   bob.ip.binseg.data.binsegdataset
   bob.ip.binseg.data.folderdataset
   bob.ip.binseg.data.csvdataset
   bob.ip.binseg.data.jsondataset
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
   :toctree: api/modeling

   bob.ip.binseg.modeling
   bob.ip.binseg.modeling.backbones
   bob.ip.binseg.modeling.backbones.mobilenetv2
   bob.ip.binseg.modeling.backbones.resnet
   bob.ip.binseg.modeling.backbones.vgg
   bob.ip.binseg.modeling.driu
   bob.ip.binseg.modeling.driubn
   bob.ip.binseg.modeling.driuod
   bob.ip.binseg.modeling.driupix
   bob.ip.binseg.modeling.hed
   bob.ip.binseg.modeling.losses
   bob.ip.binseg.modeling.m2u
   bob.ip.binseg.modeling.make_layers
   bob.ip.binseg.modeling.resunet
   bob.ip.binseg.modeling.unet


Toolbox
-------

.. autosummary::
   :toctree: api/utils

   bob.ip.binseg.utils
   bob.ip.binseg.utils.checkpointer
   bob.ip.binseg.utils.metric
   bob.ip.binseg.utils.model_serialization
   bob.ip.binseg.utils.model_zoo
   bob.ip.binseg.utils.plot
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

.. autosummary::
   :toctree: api/configs/datasets
   :template: config.rst

   bob.ip.binseg.configs.datasets.csv
   bob.ip.binseg.configs.datasets.folder
   bob.ip.binseg.configs.datasets.utils

   bob.ip.binseg.configs.datasets.chasedb1
   bob.ip.binseg.configs.datasets.chasedb1_test
   bob.ip.binseg.configs.datasets.covd_chasedb1
   bob.ip.binseg.configs.datasets.covd_chasedb1_ssl

   bob.ip.binseg.configs.datasets.drive
   bob.ip.binseg.configs.datasets.drive_test
   bob.ip.binseg.configs.datasets.covd_drive
   bob.ip.binseg.configs.datasets.covd_drive_ssl

   bob.ip.binseg.configs.datasets.hrf
   bob.ip.binseg.configs.datasets.hrf_1168
   bob.ip.binseg.configs.datasets.hrf_1168_test
   bob.ip.binseg.configs.datasets.hrf_test
   bob.ip.binseg.configs.datasets.covd_hrf
   bob.ip.binseg.configs.datasets.covd_hrf_ssl

   bob.ip.binseg.configs.datasets.iostar_vessel
   bob.ip.binseg.configs.datasets.iostar_vessel_test
   bob.ip.binseg.configs.datasets.covd_iostar_vessel
   bob.ip.binseg.configs.datasets.covd_iostar_vessel_ssl
   bob.ip.binseg.configs.datasets.iostar_od
   bob.ip.binseg.configs.datasets.iostar_od_test

   bob.ip.binseg.configs.datasets.stare
   bob.ip.binseg.configs.datasets.stare_test
   bob.ip.binseg.configs.datasets.covd_stare
   bob.ip.binseg.configs.datasets.covd_stare_ssl

   bob.ip.binseg.configs.datasets.drionsdb
   bob.ip.binseg.configs.datasets.drionsdb_test

   bob.ip.binseg.configs.datasets.dristhigs1_cup
   bob.ip.binseg.configs.datasets.dristhigs1_cup_test
   bob.ip.binseg.configs.datasets.dristhigs1_od
   bob.ip.binseg.configs.datasets.dristhigs1_od_test

   bob.ip.binseg.configs.datasets.refuge_cup
   bob.ip.binseg.configs.datasets.refuge_cup_dev
   bob.ip.binseg.configs.datasets.refuge_cup_test
   bob.ip.binseg.configs.datasets.refuge_cup_test
   bob.ip.binseg.configs.datasets.refuge_od
   bob.ip.binseg.configs.datasets.refuge_od_dev
   bob.ip.binseg.configs.datasets.refuge_od_test

   bob.ip.binseg.configs.datasets.rimoner3_cup
   bob.ip.binseg.configs.datasets.rimoner3_cup_test
   bob.ip.binseg.configs.datasets.rimoner3_od
   bob.ip.binseg.configs.datasets.rimoner3_od_test
