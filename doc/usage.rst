.. -*- coding: utf-8 -*-

.. _bob.ip.binseg.usage:

=======
 Usage
=======

This package supports a fully reproducible research experimentation cycle for
semantic binary segmentation with support for the following activities:

* Training: Images are fed to a Fully Convolutional Deep Neural Network (FCN),
  that is trained to reconstruct annotations (pre-segmented binary maps),
  automatically, via error back propagation.  The objective of this phase is to
  produce an FCN model.
* Inference: The FCN is used to generate vessel map predictions
* Evaluation: Vessel map predictions are used evaluate FCN performance against
  test data, generate ROC curves or visualize prediction results overlayed on
  the original raw images.

Each application is implemented as a :ref:`command-line utility
<bob.ip.binseg.cli>`, that is configurable using :ref:`Bob's extensible
configuration framework <bob.extension.framework>`.  In essence, each
command-line option may be provided as a variable with the same name in a
Python file.  Each file may combine any number of variables that are pertinent
to an application.

.. tip::

   For reproducibility, we recommend you stick to configuration files when
   parameterizing our CLI.  Notice some of the options in the CLI interface
   (e.g. ``--dataset``) cannot be passed via the actual command-line as it
   requires a :py:class:`concrete PyTorch dataset instance
   <torch.utils.data.dataset.Dataset>`.

We provide a number of :ref:`preset configuration files
<bob.ip.binseg.cli.config.list.all>` that can be used in one or more of the
activities described in this section.  Our command-line framework allows you to
refer to these preset configuration files using special names (a.k.a.
"resources"), that procure and load these for you automatically.  Aside preset
configuration files, you may also create your own to extend existing baseline
experiments by :ref:`locally copying <bob.ip.binseg.cli.config.copy>` and
modifying one of our configuration resources.


.. toctree::
   :maxdepth: 2

   training
   models
   evaluation
   experiment


.. include:: links.rst
