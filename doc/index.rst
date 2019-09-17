.. -*- coding: utf-8 -*-

.. _bob.ip.binseg:

===============================================
 Binary Segmentation Benchmark Package for Bob
===============================================

Package to benchmark and evaluate a range of neural network architectures for
binary segmentation tasks on 2D Eye Fundus Images (2DFI). It is build using
PyTorch.

Please use the BibTeX reference below to cite this work::

  @article{laibacher_anjos_2019,
        author = {Laibacher, Tim and Anjos, Andr\'e},
        title = "{On the Evaluation and Real-World Usage Scenarios of Deep Vessel Segmentation for Funduscopy}",
        journal = {arXiv e-prints},
        keywords = {Computer Science - Computer Vision and Pattern Recognition},
        year = {2019},
        month = {Sep},
        url = {https://arxiv.org/abs/1909.03856},
        eid = {arXiv:1909.03856},
        pages = {arXiv:1909.03856},
        archivePrefix = {arXiv},
        eprint = {1909.03856},
  }


Additional Material
===================

The additional material referred to in the paper can be found under
:ref:`bob.ip.binseg.covdresults` and :download:`here </additionalresults.pdf>`


Users Guide
===========

.. toctree::
   :maxdepth: 2

   setup
   datasets
   training
   evaluation
   benchmarkresults
   covdresults
   configs
   plotting
   visualization
   api
   acknowledgements

.. todolist::

.. include:: links.rst
