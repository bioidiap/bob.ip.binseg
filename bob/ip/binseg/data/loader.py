#!/usr/bin/env python
# coding=utf-8


"""Data loading code"""


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
