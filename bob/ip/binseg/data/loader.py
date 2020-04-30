#!/usr/bin/env python
# coding=utf-8


"""Data loading code"""


import os
import functools

import PIL.Image

from .sample import DelayedSample


def load_pil_rgb(path):
    """Loads a sample data

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
    """Loads a sample binary label or mask

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
    """Returns a delayed-loading Sample object

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

    sample : bob.ip.binseg.data.sample.DelayedSample
        In which ``key`` is as provided and ``data`` can be accessed to trigger
        sample loading.

    """

    return DelayedSample(
            functools.partial(loader, sample),
            key=key or os.path.splitext(sample["data"])[0],
            )
