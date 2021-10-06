#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image transformations for our pipelines

Differences between methods here and those from
:py:mod:`torchvision.transforms` is that these support multiple simultaneous
image inputs, which are required to feed segmentation networks (e.g. image and
labels or masks).  We also take care of data augmentations, in which random
flipping and rotation needs to be applied across all input images, but color
jittering, for example, only on the input image.
"""

import random

import numpy
import PIL.Image
import PIL.ImageOps
import torchvision.transforms
import torchvision.transforms.functional


class TupleMixin:
    """Adds support to work with tuples of objects to torchvision transforms"""

    def __call__(self, *args):
        return [super(TupleMixin, self).__call__(k) for k in args]


class CenterCrop(TupleMixin, torchvision.transforms.CenterCrop):
    pass


class Pad(TupleMixin, torchvision.transforms.Pad):
    pass


class Resize(TupleMixin, torchvision.transforms.Resize):
    pass


class ToTensor(TupleMixin, torchvision.transforms.ToTensor):
    pass


class Compose(torchvision.transforms.Compose):
    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class SingleCrop:
    """
    Crops one image at the given coordinates.

    Attributes
    ----------
    i : int
        upper pixel coordinate.
    j : int
        left pixel coordinate.
    h : int
        height of the cropped image.
    w : int
        width of the cropped image.
    """

    def __init__(self, i, j, h, w):
        self.i = i
        self.j = j
        self.h = h
        self.w = w

    def __call__(self, img):
        return img.crop((self.j, self.i, self.j + self.w, self.i + self.h))


class Crop(TupleMixin, SingleCrop):
    """
    Crops multiple images at the given coordinates.

    Attributes
    ----------
    i : int
        upper pixel coordinate.
    j : int
        left pixel coordinate.
    h : int
        height of the cropped image.
    w : int
        width of the cropped image.
    """

    pass


class SingleAutoLevel16to8:
    """Converts a 16-bit image to 8-bit representation using "auto-level"

    This transform assumes that the input image is gray-scaled.

    To auto-level, we calculate the maximum and the minimum of the image, and
    consider such a range should be mapped to the [0,255] range of the
    destination image.

    """

    def __call__(self, img):
        imin, imax = img.getextrema()
        irange = imax - imin
        return PIL.Image.fromarray(
            numpy.round(
                255.0 * (numpy.array(img).astype(float) - imin) / irange
            ).astype("uint8"),
        ).convert("L")


class AutoLevel16to8(TupleMixin, SingleAutoLevel16to8):
    """Converts multiple 16-bit images to 8-bit representations using "auto-level"

    This transform assumes that the input images are gray-scaled.

    To auto-level, we calculate the maximum and the minimum of the image, and
    consider such a range should be mapped to the [0,255] range of the
    destination image.
    """

    pass


class SingleToRGB:
    """Converts from any input format to RGB, using an ADAPTIVE conversion.

    This transform takes the input image and converts it to RGB using
    py:method:`PIL.Image.Image.convert`, with `mode='RGB'` and using all other
    defaults.  This may be aggressive if applied to 16-bit images without
    further considerations.
    """

    def __call__(self, img):
        return img.convert(mode="RGB")


class ToRGB(TupleMixin, SingleToRGB):
    """Converts from any input format to RGB, using an ADAPTIVE conversion.

    This transform takes the input image and converts it to RGB using
    py:method:`PIL.Image.Image.convert`, with `mode='RGB'` and using all other
    defaults.  This may be aggressive if applied to 16-bit images without
    further considerations.
    """

    pass


class RandomHorizontalFlip(torchvision.transforms.RandomHorizontalFlip):
    """Randomly flips all input images horizontally"""

    def __call__(self, *args):
        if random.random() < self.p:
            return [
                torchvision.transforms.functional.hflip(img) for img in args
            ]
        else:
            return args


class RandomVerticalFlip(torchvision.transforms.RandomVerticalFlip):
    """Randomly flips all input images vertically"""

    def __call__(self, *args):
        if random.random() < self.p:
            return [
                torchvision.transforms.functional.vflip(img) for img in args
            ]
        else:
            return args


class RandomRotation(torchvision.transforms.RandomRotation):
    """Randomly rotates all input images by the same amount

    Unlike the current torchvision implementation, we also accept a probability
    for applying the rotation.


    Parameters
    ----------

    p : :py:class:`float`, Optional
        probability at which the operation is applied

    **kwargs : dict
        passed to parent.  Notice that, if not set, we use the following
        defaults here for the underlying transform from torchvision:

        * ``degrees``: 15
        * ``interpolation``: ``torchvision.transforms.functional.InterpolationMode.BILINEAR``

    """

    def __init__(self, p=0.5, **kwargs):
        kwargs.setdefault("degrees", 15)
        kwargs.setdefault(
            "interpolation",
            torchvision.transforms.functional.InterpolationMode.BILINEAR,
        )
        super(RandomRotation, self).__init__(**kwargs)
        self.p = p

    def __call__(self, *args):
        # applies **the same** rotation to all inputs (data and ground-truth)
        if random.random() < self.p:
            angle = self.get_params(self.degrees)
            return [
                torchvision.transforms.functional.rotate(
                    img, angle, self.interpolation, self.expand, self.center
                )
                for img in args
            ]
        else:
            return args

    def __repr__(self):
        retval = super(RandomRotation, self).__repr__()
        return retval.replace("(", f"(p={self.p},", 1)


class ColorJitter(torchvision.transforms.ColorJitter):
    """Randomly applies a color jitter transformation on the **first** image

    Notice this transform extension, unlike others in this module, only affects
    the first image passed as input argument.  Unlike the current torchvision
    implementation, we also accept a probability for applying the jitter.


    Parameters
    ----------

    p : :py:class:`float`, Optional
        probability at which the operation is applied

    **kwargs : dict
        passed to parent.  Notice that, if not set, we use the following
        defaults here for the underlying transform from torchvision:

        * ``brightness``: 0.3
        * ``contrast``: 0.3
        * ``saturation``: 0.02
        * ``hue``: 0.02

    """

    def __init__(self, p=0.5, **kwargs):
        kwargs.setdefault("brightness", 0.3)
        kwargs.setdefault("contrast", 0.3)
        kwargs.setdefault("saturation", 0.02)
        kwargs.setdefault("hue", 0.02)
        super(ColorJitter, self).__init__(**kwargs)
        self.p = p

    def __call__(self, *args):
        if random.random() < self.p:
            # applies color jitter only to the input image not ground-truth
            return [super(ColorJitter, self).__call__(args[0]), *args[1:]]
        else:
            return args

    def __repr__(self):
        retval = super(ColorJitter, self).__repr__()
        return retval.replace("(", f"(p={self.p},", 1)


def _expand2square(pil_img, background_color):
    """
    Function that pad the minimum between the height and the width to fit a square

    Parameters
    ----------

    pil_img : PIL.Image.Image
        A PIL image that represents the image for analysis.

    background_color: py:class:`tuple`, Optional
        A tuple to represent the color of the background of the image in order to pad with the same color.
        If the image is an RGB image background_color should be a tuple of size 3 , if it's a grayscale image the variable can be represented with an integer.

    Returns
    -------

    image : PIL.Image.Image
        A new image with height equal to width.


    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = PIL.Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = PIL.Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class ResizeCrop:
    """
    Crop all the images by removing the black pixels in the width and height until it finds a non-black pixel.

    """

    def __call__(self, *args):

        img = args[0]
        label = args[1]
        mask = args[2]
        mask_data = numpy.asarray(mask)
        wid = numpy.sum(mask_data, axis=0)
        heig = numpy.sum(mask_data, axis=1)

        crop_left, crop_right = (wid != 0).argmax(axis=0), (
            wid[::-1] != 0
        ).argmax(axis=0)
        crop_up, crop_down = (heig != 0).argmax(axis=0), (
            heig[::-1] != 0
        ).argmax(axis=0)

        border = (crop_left, crop_up, crop_right, crop_down)

        new_mask = PIL.ImageOps.crop(mask, border)
        new_img = PIL.ImageOps.crop(img, border)
        new_label = PIL.ImageOps.crop(label, border)

        new_img = _expand2square(new_img, (0, 0, 0))
        new_label = _expand2square(new_label, 0)
        new_mask = _expand2square(new_mask, 0)

        args = (new_img, new_label, new_mask)

        return args
