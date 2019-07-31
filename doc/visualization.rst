.. -*- coding: utf-8 -*-
.. _bob.ip.binseg.visualization:

=============
Visualization
=============

Two visualization are generated via the ``bob binseg visualize`` command:

1. Visualizations of true positives, false positives and false negatives
overlayed over the test images
2. Visualizations of the probability map outputs overlayed over the test images

The following directory structure is expected:

.. code-block:: bash

    ├── DATABASE
        ├── MODEL
            ├── images
            └── results

Example to generate visualization for outputs for the DRIVE dataset:

.. code-block:: bash

    # Visualizations are stored in the same output folder.
    bob binseg visualize DRIVETEST -o /DRIVE/M2UNet/output

Use ``bob binseg visualize --help`` for more information.
