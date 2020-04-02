.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.cli:

==============================
 Command-Line Interface (CLI)
==============================

This package provides a single entry point for all of its applications using
:ref:`Bob's unified CLI mechanism <bob.extension.cli>`.  A list of available
applications can be retrieved using:

.. command-output:: bob binseg --help


Preset Configuration Resources
------------------------------

A CLI application allows one to list, inspect and copy available configuration
resources exported by this package.

.. _bob.ip.binseg.cli.config:

.. command-output:: bob binseg config --help


.. _bob.ip.binseg.cli.config.list:

Listing Resources
=================

.. command-output:: bob binseg config list --help


.. _bob.ip.binseg.cli.config.list.all:

Available Resources
===================

Here is a list of all resources currently exported.

.. command-output:: bob binseg config list -v


.. _bob.ip.binseg.cli.config.describe:

Describing a Resource
=====================

.. command-output:: bob binseg config describe --help


.. _bob.ip.binseg.cli.config.copy:

Copying a Resource
==================

You may use this command to locally copy a resource file so you can change it.

.. command-output:: bob binseg config copy --help


.. _bob.ip.binseg.cli.train:

Training FCNs
-------------

Training creates of a new PyTorch_ model.  This model can be used for
evaluation tests or for inference.

.. command-output:: bob binseg train --help


.. include:: links.rst
