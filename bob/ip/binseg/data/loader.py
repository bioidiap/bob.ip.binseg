#!/usr/bin/env python
# coding=utf-8


"""Data loading code"""


import os
import PIL.Image


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


def data_path_keymaker(context, sample):
    """Returns a path without extension as a key

    This method assumes ``sample`` contains at least one entry named ``path``,
    that contains a path to the sample raw data, without extension.  It will
    return the said path without its extension.


    Parameters
    ----------

    context : dict
        Context dictionary with entries (``protocol``, ``subset``), depending
        on the context

    sample : dict
        A dictionary that maps field names to sample entries from the original
        dataset.


    Returns
    -------

    key : str
        A string that uniquely identifies the sample within a given context

    """

    return os.path.splitext(sample["data"])[0]
