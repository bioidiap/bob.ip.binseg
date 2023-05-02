# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later


import glob
import os

import click
import numpy
import skimage.io
import skimage.morphology

from clapper.click import ConfigCommand, ResourceOption, verbosity_option
from clapper.logging import setup

logger = setup(__name__.split(".")[0], format="%(levelname)s: %(message)s")

from ..utils.rc import load_rc


@click.command(
    cls=ConfigCommand,
    epilog="""Examples:

\b
  1. Generate masks for supported dataset by deepdraw. Ex: refuge.

     .. code:: sh

        $ deepdraw mkmask --dataset="refuge" --globs="Training400/*Glaucoma/*.jpg" --globs="Training400/*AMD/*.jpg" --threshold=5

     Or you can generate the same results with this command

     .. code:: sh

        $ deepdraw mkmask -d "refuge" -g "Training400/*Glaucoma/*.jpg" -g "Training400/*AMD/*.jpg" -t 5

\b
  2. Generate masks for non supported dataset by deepdraw

     .. code:: sh

        $ deepdraw mkmask -d "Path/to/dataset" -g "glob1" -g "glob2" -g glob3  -t 4
""",
)
@click.option(
    "--output-folder",
    "-o",
    help="Path where to store the generated model (created if does not exist)",
    required=True,
    type=click.Path(),
    default="masks",
    cls=ResourceOption,
)
@click.option(
    "--dataset",
    "-d",
    help="""The base path to the dataset to which we want to generate the masks. \\
    In case you have already configured the path for the datasets supported by deepdraw, \\
    you can just use the name of the dataset as written in the config.
    """,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--globs",
    "-g",
    help="""The global path to the dataset to which we want to generate the masks.\\
    We need to specify the path for the images ,\\
    Ex : --globs="images/\\*.jpg"\\
    It also can be used multiple time.
    """,
    required=True,
    multiple=True,
    cls=ResourceOption,
)
@click.option(
    "--threshold",
    "-t",
    help="Generating a mask needs a threshold to be fixed in order to transform the image to binary ",
    required=True,
    cls=ResourceOption,
)
@verbosity_option(logger=logger, cls=ResourceOption)
@click.pass_context
def mkmask(ctx, dataset, globs, threshold, output_folder, verbose, **kwargs):
    """Commands for generating masks for images in a dataset."""

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
            patternrc.get("deepdraw." + dataset + ".datadir"):

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

    if rc.get("deepdraw." + dataset + ".datadir"):
        base_path = rc.get("deepdraw." + dataset + ".datadir")
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
