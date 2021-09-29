.. -*- coding: utf-8 -*-

.. _bob.ip.binseg:

===============================================
 Binary Segmentation Benchmark Package for Bob
===============================================

Package to benchmark and evaluate a range of neural network architectures for
binary segmentation tasks.  It is build using PyTorch.

Please at least use the BibTeX references below to cite this work:

.. code:: bibtex

   @inproceedings{renzo_2021,
       title     = {Development of a lung segmentation algorithm for analog imaged chest X-Ray: preliminary results},
       author    = {Matheus A. Renzo and Nat\'{a}lia Fernandez and Andr\'e Baceti and Natanael Nunes de Moura Junior and Andr\'e Anjos},
       month     = {10},
       booktitle = {XV Brazilian Congress on Computational Intelligence},
       year      = {2021},
       url       = {https://publications.idiap.ch/index.php/publications/show/4649},
   }

   @misc{laibacher_2019,
       title         = {On the Evaluation and Real-World Usage Scenarios of Deep Vessel Segmentation for Retinography},
       author        = {Tim Laibacher and Andr\'e Anjos},
       year          = {2019},
       eprint        = {1909.03856},
       archivePrefix = {arXiv},
       primaryClass  = {cs.CV},
       url           = {https://arxiv.org/abs/1909.03856},
   }


User Guide
----------

.. toctree::
   :maxdepth: 2

   setup
   usage
   results/index
   acknowledgements
   references
   datasets
   cli
   api
   contribute


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. include:: links.rst
