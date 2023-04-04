.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.config:

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

   deepdraw.configs.models.driu
   deepdraw.configs.models.driu_bn
   deepdraw.configs.models.driu_od
   deepdraw.configs.models.hed
   deepdraw.configs.models.m2unet
   deepdraw.configs.models.resunet
   deepdraw.configs.models.unet
   deepdraw.configs.models.lwnet



.. _deepdraw.configs.datasets:

Datasets
========

Datasets include iterative accessors to raw data
(:ref:`deepdraw.setup.datasets`) including data pre-processing and augmentation,
if applicable.  Use these datasets for training and evaluating your models.

.. autosummary::
   :toctree: api/configs/datasets
   :template: config.rst

   deepdraw.configs.datasets.__init__

   deepdraw.configs.datasets.csv

   deepdraw.configs.datasets.chasedb1.first_annotator
   deepdraw.configs.datasets.chasedb1.first_annotator_768
   deepdraw.configs.datasets.chasedb1.first_annotator_1024
   deepdraw.configs.datasets.chasedb1.second_annotator
   deepdraw.configs.datasets.chasedb1.xtest
   deepdraw.configs.datasets.chasedb1.mtest
   deepdraw.configs.datasets.chasedb1.covd

   deepdraw.configs.datasets.drive.default
   deepdraw.configs.datasets.drive.default_768
   deepdraw.configs.datasets.drive.default_1024
   deepdraw.configs.datasets.drive.second_annotator
   deepdraw.configs.datasets.drive.xtest
   deepdraw.configs.datasets.drive.mtest
   deepdraw.configs.datasets.drive.covd

   deepdraw.configs.datasets.hrf.default
   deepdraw.configs.datasets.hrf.default_768
   deepdraw.configs.datasets.hrf.default_1024
   deepdraw.configs.datasets.hrf.xtest
   deepdraw.configs.datasets.hrf.mtest
   deepdraw.configs.datasets.hrf.default_fullres
   deepdraw.configs.datasets.hrf.covd

   deepdraw.configs.datasets.iostar.vessel
   deepdraw.configs.datasets.iostar.vessel_768
   deepdraw.configs.datasets.iostar.vessel_xtest
   deepdraw.configs.datasets.iostar.vessel_mtest
   deepdraw.configs.datasets.iostar.optic_disc
   deepdraw.configs.datasets.iostar.optic_disc_768
   deepdraw.configs.datasets.iostar.optic_disc_512
   deepdraw.configs.datasets.iostar.covd

   deepdraw.configs.datasets.stare.ah
   deepdraw.configs.datasets.stare.ah_768
   deepdraw.configs.datasets.stare.ah_1024
   deepdraw.configs.datasets.stare.vk
   deepdraw.configs.datasets.stare.xtest
   deepdraw.configs.datasets.stare.mtest
   deepdraw.configs.datasets.stare.covd

   deepdraw.configs.datasets.refuge.cup
   deepdraw.configs.datasets.refuge.disc
   deepdraw.configs.datasets.refuge.cup_512
   deepdraw.configs.datasets.refuge.cup_768
   deepdraw.configs.datasets.refuge.disc_512
   deepdraw.configs.datasets.refuge.disc_768

   deepdraw.configs.datasets.rimoner3.cup_exp1
   deepdraw.configs.datasets.rimoner3.cup_exp2
   deepdraw.configs.datasets.rimoner3.disc_exp1
   deepdraw.configs.datasets.rimoner3.disc_exp2
   deepdraw.configs.datasets.rimoner3.cup_exp1_512
   deepdraw.configs.datasets.rimoner3.disc_exp1_512

   deepdraw.configs.datasets.rimoner3.cup_exp1_768
   deepdraw.configs.datasets.rimoner3.disc_exp1_768
   deepdraw.configs.datasets.drishtigs1.cup_all
   deepdraw.configs.datasets.drishtigs1.cup_all_512
   deepdraw.configs.datasets.drishtigs1.cup_all_768
   deepdraw.configs.datasets.drishtigs1.cup_any
   deepdraw.configs.datasets.drishtigs1.disc_all
   deepdraw.configs.datasets.drishtigs1.disc_all_512
   deepdraw.configs.datasets.drishtigs1.disc_all_768
   deepdraw.configs.datasets.drishtigs1.disc_any

   deepdraw.configs.datasets.drionsdb.expert1
   deepdraw.configs.datasets.drionsdb.expert2
   deepdraw.configs.datasets.drionsdb.expert1_512
   deepdraw.configs.datasets.drionsdb.expert2_512
   deepdraw.configs.datasets.drionsdb.expert1_768
   deepdraw.configs.datasets.drionsdb.expert2_768

   deepdraw.configs.datasets.drhagis.default

   deepdraw.configs.datasets.montgomery.default
   deepdraw.configs.datasets.montgomery.xtest

   deepdraw.configs.datasets.jsrt.default
   deepdraw.configs.datasets.jsrt.xtest

   deepdraw.configs.datasets.shenzhen.default
   deepdraw.configs.datasets.shenzhen.default_256
   deepdraw.configs.datasets.shenzhen.xtest

   deepdraw.configs.datasets.cxr8.default
   deepdraw.configs.datasets.cxr8.xtest

.. include:: links.rst
