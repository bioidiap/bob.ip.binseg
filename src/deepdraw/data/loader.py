# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Data loading code."""


import functools
import os

import numpy
import PIL.Image
import PIL.ImageFile
import skimage.exposure

from .sample import DelayedSample

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_pil_raw_12bit_jsrt(path, width):
    """Loads a raw 16-bit sample data.

    This method was designed to handle the raw images from the JSRT_ dataset.
    It reads the data file and applies a simple histogram equalization to the
    8-bit representation of the image to obtain something along the lines of
    the PNG (unofficial) version distributed at `JSRT-Kaggle`_.


    Parameters
    ----------

    path : str
        The full path leading to the image to be loaded

    width : int
        The desired width of the output image


    Returns
    -------

    image : PIL.Image.Image
        A PIL image in RGB mode, with `width`x`width` pixels


    .. include:: ../../links.rst
    """

    raw_image = numpy.fromfile(path, numpy.dtype(">u2")).reshape(2048, 2048)
    raw_image[raw_image > 4095] = 4095
    raw_image = 4095 - raw_image  # invert colors
    raw_image = (raw_image >> 4).astype(numpy.uint8)  # 8-bit uint
    raw_image = skimage.exposure.equalize_hist(raw_image)
    return (
        PIL.Image.fromarray((raw_image * 255).astype(numpy.uint8))
        .resize((width, width))
        .convert("RGB")
    )


def load_pil_rgb(path):
    """Loads a sample data.

    Parameters
    ----------

    path : str
        The full path leading to the image to be loaded


    Returns
    -------

    image : PIL.Image.Image
        A PIL image in RGB mode
    """

    return PIL.Image.open(path).convert("RGB")


def load_pil_1(path):
    """Loads a sample binary label or mask.

    Parameters
    ----------

    path : str
        The full path leading to the image to be loaded


    Returns
    -------

    image : PIL.Image.Image
        A PIL image in mode "1"
    """

    return PIL.Image.open(path).convert(mode="1", dither=None)


def make_delayed(sample, loader, key=None):
    """Returns a delayed-loading Sample object.

    Parameters
    ----------

    sample : dict
        A dictionary that maps field names to sample data values (e.g. paths)

    loader : object
        A function that inputs ``sample`` dictionaries and returns the loaded
        data.

    key : str
        A unique key identifier for this sample.  If not provided, assumes
        ``sample`` is a dictionary with a ``data`` entry and uses its path as
        key.


    Returns
    -------

    sample : deepdraw.data.sample.DelayedSample
        In which ``key`` is as provided and ``data`` can be accessed to trigger
        sample loading.
    """

    return DelayedSample(
        functools.partial(loader, sample),
        key=key or os.path.splitext(sample["data"])[0],
    )
