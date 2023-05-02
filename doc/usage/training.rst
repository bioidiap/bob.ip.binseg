.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
.. SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
.. SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
.. SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
.. SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _deepdraw.training:

==========
 Training
==========

To train a new FCN, use the command-line interface (CLI) application ``binseg train``, available on your prompt.  To use this CLI, you must define the
input dataset that will be used to train the FCN, as well as the type of model
that will be trained.  You may issue ``binseg train --help`` for a help
message containing more detailed instructions.

.. tip::

   We strongly advice training with a GPU (using ``--device="cuda:0"``).
   Depending on the available GPU memory you might have to adjust your batch
   size (``--batch``).

.. include:: ../links.rst
