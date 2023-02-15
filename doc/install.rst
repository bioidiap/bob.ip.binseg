.. SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
.. SPDX-FileContributor: Tim Laibacher <tim.laibacher@idiap.ch>
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.install:

==============
 Installation
==============

.. todo:: fine-tune installation instructions for deepdraw here


We support two installation modes, through pip_, or mamba_ (conda).


With pip
--------

.. code-block:: sh

   # stable, from PyPI:
   $ pip install deepdraw

   # latest beta, from GitLab package registry:
   $ pip install --pre --index-url https://gitlab.idiap.ch/api/v4/groups/biosignal/-/packages/pypi/simple --extra-index-url https://pypi.org/simple deepdraw

.. tip::

   To avoid long command-lines you may configure pip to define the indexes and
   package search priorities as you like.


With conda
----------

.. code-block:: sh

   # stable:
   $ mamba install -c https://www.idiap.ch/software/biosignal/conda -c conda-forge deepdraw

   # latest beta:
   $ mamba install -c https://www.idiap.ch/software/biosignal/conda/label/beta -c conda-forge deepdraw


.. include:: links.rst
