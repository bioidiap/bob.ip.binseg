.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.binseg.config:

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

   deepdraw.binseg.configs.models.driu
   deepdraw.binseg.configs.models.driu_bn
   deepdraw.binseg.configs.models.driu_od
   deepdraw.binseg.configs.models.hed
   deepdraw.binseg.configs.models.m2unet
   deepdraw.binseg.configs.models.resunet
   deepdraw.binseg.configs.models.unet
   deepdraw.binseg.configs.models.lwnet
   deepdraw.binseg.configs.models.mean_teacher



.. _deepdraw.binseg.configs.datasets:

Datasets
========

Datasets include iterative accessors to raw data
(:ref:`deepdraw.setup.datasets`) including data pre-processing and augmentation,
if applicable.  Use these datasets for training and evaluating your models.

.. autosummary::
   :toctree: api/configs/datasets
   :template: config.rst

   deepdraw.binseg.configs.datasets.__init__

   deepdraw.binseg.configs.datasets.csv

   deepdraw.binseg.configs.datasets.chasedb1.first_annotator
   deepdraw.binseg.configs.datasets.chasedb1.first_annotator_768
   deepdraw.binseg.configs.datasets.chasedb1.first_annotator_1024
   deepdraw.binseg.configs.datasets.chasedb1.second_annotator
   deepdraw.binseg.configs.datasets.chasedb1.xtest
   deepdraw.binseg.configs.datasets.chasedb1.mtest
   deepdraw.binseg.configs.datasets.chasedb1.covd

   deepdraw.binseg.configs.datasets.drive.default
   deepdraw.binseg.configs.datasets.drive.default_768
   deepdraw.binseg.configs.datasets.drive.default_1024
   deepdraw.binseg.configs.datasets.drive.second_annotator
   deepdraw.binseg.configs.datasets.drive.xtest
   deepdraw.binseg.configs.datasets.drive.mtest
   deepdraw.binseg.configs.datasets.drive.covd
   deepdraw.binseg.configs.datasets.drive.semi_768
   deepdraw.binseg.configs.datasets.drive.semi_x768

   deepdraw.binseg.configs.datasets.hrf.default
   deepdraw.binseg.configs.datasets.hrf.default_768
   deepdraw.binseg.configs.datasets.hrf.default_1024
   deepdraw.binseg.configs.datasets.hrf.xtest
   deepdraw.binseg.configs.datasets.hrf.mtest
   deepdraw.binseg.configs.datasets.hrf.default_fullres
   deepdraw.binseg.configs.datasets.hrf.covd
   deepdraw.binseg.configs.datasets.hrf.semi_768
   deepdraw.binseg.configs.datasets.hrf.semi_x768

   deepdraw.binseg.configs.datasets.iostar.vessel
   deepdraw.binseg.configs.datasets.iostar.vessel_768
   deepdraw.binseg.configs.datasets.iostar.vessel_xtest
   deepdraw.binseg.configs.datasets.iostar.vessel_mtest
   deepdraw.binseg.configs.datasets.iostar.optic_disc
   deepdraw.binseg.configs.datasets.iostar.optic_disc_768
   deepdraw.binseg.configs.datasets.iostar.optic_disc_512
   deepdraw.binseg.configs.datasets.iostar.covd
   deepdraw.binseg.configs.datasets.iostar.semi_768
   deepdraw.binseg.configs.datasets.iostar.semi_x768

   deepdraw.binseg.configs.datasets.stare.ah
   deepdraw.binseg.configs.datasets.stare.ah_768
   deepdraw.binseg.configs.datasets.stare.ah_1024
   deepdraw.binseg.configs.datasets.stare.vk
   deepdraw.binseg.configs.datasets.stare.xtest
   deepdraw.binseg.configs.datasets.stare.mtest
   deepdraw.binseg.configs.datasets.stare.covd
   deepdraw.binseg.configs.datasets.stare.semi_768
   deepdraw.binseg.configs.datasets.stare.semi_x768

   deepdraw.binseg.configs.datasets.refuge.cup
   deepdraw.binseg.configs.datasets.refuge.disc
   deepdraw.binseg.configs.datasets.refuge.cup_512
   deepdraw.binseg.configs.datasets.refuge.cup_768
   deepdraw.binseg.configs.datasets.refuge.disc_512
   deepdraw.binseg.configs.datasets.refuge.disc_768

   deepdraw.binseg.configs.datasets.rimoner3.cup_exp1
   deepdraw.binseg.configs.datasets.rimoner3.cup_exp2
   deepdraw.binseg.configs.datasets.rimoner3.disc_exp1
   deepdraw.binseg.configs.datasets.rimoner3.disc_exp2
   deepdraw.binseg.configs.datasets.rimoner3.cup_exp1_512
   deepdraw.binseg.configs.datasets.rimoner3.disc_exp1_512

   deepdraw.binseg.configs.datasets.rimoner3.cup_exp1_768
   deepdraw.binseg.configs.datasets.rimoner3.disc_exp1_768
   deepdraw.binseg.configs.datasets.drishtigs1.cup_all
   deepdraw.binseg.configs.datasets.drishtigs1.cup_all_512
   deepdraw.binseg.configs.datasets.drishtigs1.cup_all_768
   deepdraw.binseg.configs.datasets.drishtigs1.cup_any
   deepdraw.binseg.configs.datasets.drishtigs1.disc_all
   deepdraw.binseg.configs.datasets.drishtigs1.disc_all_512
   deepdraw.binseg.configs.datasets.drishtigs1.disc_all_768
   deepdraw.binseg.configs.datasets.drishtigs1.disc_any

   deepdraw.binseg.configs.datasets.drionsdb.expert1
   deepdraw.binseg.configs.datasets.drionsdb.expert2
   deepdraw.binseg.configs.datasets.drionsdb.expert1_512
   deepdraw.binseg.configs.datasets.drionsdb.expert2_512
   deepdraw.binseg.configs.datasets.drionsdb.expert1_768
   deepdraw.binseg.configs.datasets.drionsdb.expert2_768

   deepdraw.binseg.configs.datasets.drhagis.default

   deepdraw.binseg.configs.datasets.montgomery.default
   deepdraw.binseg.configs.datasets.montgomery.xtest

   deepdraw.binseg.configs.datasets.jsrt.default
   deepdraw.binseg.configs.datasets.jsrt.xtest

   deepdraw.binseg.configs.datasets.shenzhen.default
   deepdraw.binseg.configs.datasets.shenzhen.default_256
   deepdraw.binseg.configs.datasets.shenzhen.xtest

   deepdraw.binseg.configs.datasets.cxr8.default
   deepdraw.binseg.configs.datasets.cxr8.xtest

.. include:: links.rst
