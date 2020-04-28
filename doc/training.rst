.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.training:

==========
 Training
==========

To train a new FCN, use the command-line interface (CLI) application ``bob
binseg train``, available on your prompt.  To use this CLI, you must define the
input dataset that will be used to train the FCN, as well as the type of model
that will be trained.  You may issue ``bob binseg train --help`` for a help
message containing more detailed instructions.

.. tip::

   We strongly advice training with a GPU (using ``--device="cuda:0"``).
   Depending on the available GPU memory you might have to adjust your batch
   size (``--batch``).


