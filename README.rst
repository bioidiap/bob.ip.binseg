.. -*- coding: utf-8 -*-

.. image:: https://img.shields.io/badge/docs-v1.4.0-orange.svg
   :target: https://www.idiap.ch/software/bob/docs/bob/bob.ip.binseg/v1.4.0/index.html
.. image:: https://gitlab.idiap.ch/bob/bob.ip.binseg/badges/v1.4.0/pipeline.svg
   :target: https://gitlab.idiap.ch/bob/bob.ip.binseg/commits/v1.4.0
.. image:: https://gitlab.idiap.ch/bob/bob.ip.binseg/badges/v1.4.0/coverage.svg
   :target: https://www.idiap.ch/software/bob/docs/bob/bob.ip.binseg/v1.4.0/coverage/index.html
.. image:: https://img.shields.io/badge/gitlab-project-0000c0.svg
   :target: https://gitlab.idiap.ch/bob/bob.ip.binseg


===============================================
 Binary Segmentation Benchmark Package for Bob
===============================================

Package to benchmark and evaluate a range of neural network architectures for
binary segmentation tasks.  It is build using PyTorch.


Installation
------------

Complete bob's `installation`_ instructions. Then, to install this
package, run::

  $ conda install bob.ip.binseg


Citation
--------

If you use this software package in a publication, we would appreciate if you
could cite one or both of these references::

   @inproceedings{renzo_2021,
       title     = {Development of a lung segmentation algorithm for analog imaged chest X-Ray: preliminary results},
       author    = {Matheus A. Renzo and Nat\'{a}lia Fernandez and Andr\'e Baceti and Natanael Nunes de Moura Junior and Andr\'e Anjos},
       month     = {10},
       booktitle = {XV Brazilian Congress on Computational Intelligence},
       year      = {2021},
       url       = {https://publications.idiap.ch/index.php/publications/show/4649},
   }

   @misc{laibacher_anjos_2019,
      title         = {On the Evaluation and Real-World Usage Scenarios of Deep Vessel Segmentation for Retinography},
      author        = {Tim Laibacher and Andr\'e Anjos},
      year          = {2019},
      eprint        = {1909.03856},
      archivePrefix = {arXiv},
      primaryClass  = {cs.CV},
      url           = {https://arxiv.org/abs/1909.03856},
   }


Contact
-------

For questions or reporting issues to this software package, contact our
development `mailing list`_.


.. Place your references here:
.. _bob: https://www.idiap.ch/software/bob
.. _installation: https://www.idiap.ch/software/bob/install
.. _mailing list: https://www.idiap.ch/software/bob/discuss
