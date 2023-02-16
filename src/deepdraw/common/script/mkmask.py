#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
# SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
# SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
# SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
# SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import glob
import logging
import os

import numpy
import skimage.color
import skimage.io
import skimage.measure
import skimage.morphology

from ..utils.rc import load_rc

logger = logging.getLogger(__name__)


def base_mkmask(dataset, globs, threshold, output_folder, **kwargs):
    """Base function for mkmask."""

    def threshold_and_closing(input_path, t, width=5):
        """Creates a "rough" mask from the input image, returns binary
        equivalent.

        The mask will be created by running a simple threshold operation followed
        by a morphological closing


        Arguments
        =========

        input_path : str
            The path leading to the image from where the mask needs to be extracted

        t : int
            Threshold to apply on the original image

        width : int
            Width of the disc to use for the closing operation


        Returns
        =======

        mask : numpy.ndarray
            A 2D array, with the same size as the input image, where ``True``
            pixels correspond to the "valid" regions of the mask.
        """

        img = skimage.util.img_as_ubyte(
            skimage.io.imread(input_path, as_gray=True)
        )
        mask = img > t
        return skimage.morphology.binary_opening(
            mask, skimage.morphology.disk(width)
        )

    def count_blobs(mask):
        """Counts "white" blobs in a binary mask, outputs counts.

        Arguments
        =========

        mask : numpy.ndarray
            A 2D array, with the same size as the input image, where ``255``
            pixels correspond to the "valid" regions of the mask.  ``0`` means
            background.


        Returns
        =======

        count : int
            The number of connected blobs in the provided mask.
        """
        return skimage.measure.label(mask, return_num=True)[1]

    def process_glob(base_path, use_glob, output_path, threshold):
        """Recursively process a set of images.

        Arguments
        =========

        base_path : str
            The base directory where to look for files matching a certain name
            patternrc.get("deepdraw.binseg." + dataset + ".datadir"):

        use_glob : list
            A list of globs to use for matching filenames inside ``base_path``

        output_path : str
            Where to place the results of procesing
        """

        files = []
        for g in use_glob:
            files += glob.glob(os.path.join(base_path, g))
        for i, path in enumerate(files):
            basename = os.path.relpath(path, base_path)
            basename_without_extension = os.path.splitext(basename)[0]
            logger.info(
                f"Processing {basename_without_extension} ({i+1}/{len(files)})..."
            )
            dest = os.path.join(
                output_path, basename_without_extension + ".png"
            )
            destdir = os.path.dirname(dest)
            if not os.path.exists(destdir):
                os.makedirs(destdir)
            mask = threshold_and_closing(path, threshold)
            immask = mask.astype(numpy.uint8) * 255
            nblobs = count_blobs(immask)
            if nblobs != 1:
                logger.warning(
                    f"  -> WARNING: found {nblobs} blobs in the saved mask "
                    f"(should be one)"
                )
            skimage.io.imsave(dest, immask)

    rc = load_rc()

    if rc.get("deepdraw.binseg." + dataset + ".datadir"):
        base_path = rc.get("deepdraw.binseg." + dataset + ".datadir")
    else:
        base_path = dataset

    list_globs = []
    for g in globs:
        list_globs.append(g)
    threshold = int(threshold)
    process_glob(
        base_path=base_path,
        use_glob=list_globs,
        output_path=output_folder,
        threshold=threshold,
    )
