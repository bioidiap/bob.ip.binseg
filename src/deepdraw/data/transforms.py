# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Image transformations for our pipelines.

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
    """Adds support to work with tuples of objects to torchvision
    transforms."""

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


# NEVER USED IN THE PACKAGE
# Should it be kept?
class SingleCrop:
    """Crops one image at the given coordinates.

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
    """Crops multiple images at the given coordinates.

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
    """Converts a 16-bit image to 8-bit representation using "auto-level".

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
    """Converts multiple 16-bit images to 8-bit representations using "auto-
    level".

    This transform assumes that the input images are gray-scaled.

    To auto-level, we calculate the maximum and the minimum of the image, and
    consider such a range should be mapped to the [0,255] range of the
    destination image.
    """

    pass


class SingleToRGB:
    """Converts from any input format to RGB, using an ADAPTIVE conversion.

    This transform takes the input image and converts it to RGB using
    py:method:`PIL.Image.Image.convert`, with `mode='RGB'` and using all
    other defaults.  This may be aggressive if applied to 16-bit images
    without further considerations.
    """

    def __call__(self, img):
        return img.convert(mode="RGB")


class ToRGB(TupleMixin, SingleToRGB):
    """Converts from any input format to RGB, using an ADAPTIVE conversion.

    This transform takes the input image and converts it to RGB using
    py:method:`PIL.Image.Image.convert`, with `mode='RGB'` and using all
    other defaults.  This may be aggressive if applied to 16-bit images
    without further considerations.
    """

    pass


class RandomHorizontalFlip(torchvision.transforms.RandomHorizontalFlip):
    """Randomly flips all input images horizontally."""

    def __call__(self, *args):
        if random.random() < self.p:
            return [
                torchvision.transforms.functional.hflip(img) for img in args
            ]
        else:
            return args


class RandomVerticalFlip(torchvision.transforms.RandomVerticalFlip):
    """Randomly flips all input images vertically."""

    def __call__(self, *args):
        if random.random() < self.p:
            return [
                torchvision.transforms.functional.vflip(img) for img in args
            ]
        else:
            return args


class RandomRotation(torchvision.transforms.RandomRotation):
    """Randomly rotates all input images by the same amount.

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
        super().__init__(**kwargs)
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
        retval = super().__repr__()
        return retval.replace("(", f"(p={self.p},", 1)


class ColorJitter(torchvision.transforms.ColorJitter):
    """Randomly applies a color jitter transformation on the **first** image.

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
        super().__init__(**kwargs)
        self.p = p

    def __call__(self, *args):
        if random.random() < self.p:
            # applies color jitter only to the input image not ground-truth
            return [super().__call__(args[0]), *args[1:]]
        else:
            return args

    def __repr__(self):
        retval = super().__repr__()
        return retval.replace("(", f"(p={self.p},", 1)


def _expand2square(pil_img, background_color):
    """Function that pad the minimum between the height and the width to fit a
    square.

    Parameters
    ----------

    pil_img : PIL.Image.Image
        A PIL image that represents the image for analysis.

    background_color: py:class:`tuple`, Optional
        A tuple to represent the color of the background of the image in order
        to pad with the same color. If the image is an RGB image
        background_color should be a tuple of size 3 , if it's a grayscale
        image the variable can be represented with an integer.

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


class ShrinkIntoSquare:
    """Crops black borders and then resize to a square with minimal padding.

    This transform can crop all the images by removing the black pixels in the
    width and height until it finds a non-black pixel.  Then, expands the image
    back until it makes a square with minimal size.


    Parameters
    ----------

    reference : :py:class:`int`, Optional
        Which reference part of the sample to use for cropping black borders.
        If not set, use the first object on the sample (typically, the image).

    threshold : :py:class:`int`, Optional
        Threshold to use for when considering what is a "black" border
    """

    def __init__(self, reference=0, threshold=0):
        self.reference = reference
        self.threshold = threshold

    def __call__(self, *args):
        ref = numpy.asarray(args[self.reference].convert("L"))
        width = numpy.sum(ref, axis=0) > self.threshold
        height = numpy.sum(ref, axis=1) > self.threshold

        border = (
            width.argmax(),
            height.argmax(),
            width[::-1].argmax(),
            height[::-1].argmax(),
        )

        new_args = [PIL.ImageOps.crop(k, border) for k in args]

        def _black_background(i):
            return (0, 0, 0) if i.mode == "RGB" else 0

        return [_expand2square(k, _black_background(k)) for k in new_args]


class GaussianBlur(torchvision.transforms.GaussianBlur):
    """Randomly applies a gaussian blur transformation on the **first** image.

    Notice this transform extension, unlike others in this module, only affects
    the first image passed as input argument.  Unlike the current torchvision
    implementation, we also accept a probability for applying the blur.


    Parameters
    ----------

    p : :py:class:`float`, Optional
        probability at which the operation is applied

    **kwargs : dict
        passed to parent.  Notice that, if not set, we use the following
        defaults here for the underlying transform from torchvision:

        * ``kernel_size``: (5, 5)
        * ``sigma``: (0.1, 5)
    """

    def __init__(self, p=0.5, **kwargs):
        kwargs.setdefault("kernel_size", (5, 5))
        kwargs.setdefault("sigma", (0.1, 5))

        super().__init__(**kwargs)
        self.p = p

    def __call__(self, *args):
        if random.random() < self.p:
            # applies gaussian blur only to the input image not ground-truth
            return [super().__call__(args[0]), *args[1:]]
        else:
            return args


class GroundTruthCrop:
    """Crop image in a square keeping only the area with the ground truth.

    This transform can crop all images given a ground-truth mask as reference.
    Notice that the crop will result in a square image at the end, which means
    that it will keep the bigger dimension and adjust the smaller one to fit
    into a square. There's an option to add extra area around the gt bounding
    box. If resulting dimensions are larger than the boundaries of the image,
    minimal padding will be done to keep the image in a square shape.

    Parameters
    ----------

    reference : :py:class:`int`, Optional
        Which reference part of the sample to use for getting coordinates.
        If not set, use the second object on the sample (typically, the mask).

    extra_area : :py:class:`float`, Optional
        Multiplier that will add the extra area around the ground-truth
        bounding box. Example: 0.1 will result in a crop with dimensions of
        the largest side increased by 10%. If not set, the default will be 0
        (only the ground-truth box).
    """

    def __init__(self, reference=1, extra_area=0.0):
        self.reference = reference
        self.extra_area = extra_area

    def __call__(self, *args):
        ref = args[self.reference]

        max_w, max_h = ref.size

        where = numpy.where(ref)
        y0 = numpy.min(where[0])
        y1 = numpy.max(where[0])
        x0 = numpy.min(where[1])
        x1 = numpy.max(where[1])

        w = x1 - x0
        h = y1 - y0

        extra_x = self.extra_area * w / 2
        extra_y = self.extra_area * h / 2

        new_w = (1 + self.extra_area) * w
        new_h = (1 + self.extra_area) * h

        diff = abs(new_w - new_h) / 2

        if new_w == new_h:
            x0_new = x0.copy() - extra_x
            x1_new = x1.copy() + extra_x
            y0_new = y0.copy() - extra_y
            y1_new = y1.copy() + extra_y

        elif new_w > new_h:
            x0_new = x0.copy() - extra_x
            x1_new = x1.copy() + extra_x
            y0_new = y0.copy() - extra_y - diff
            y1_new = y1.copy() + extra_y + diff

        else:
            x0_new = x0.copy() - extra_x - diff
            x1_new = x1.copy() + extra_x + diff
            y0_new = y0.copy() - extra_y
            y1_new = y1.copy() + extra_y

        border = (x0_new, y0_new, max_w - x1_new, max_h - y1_new)

        def _expand_img(
            pil_img, background_color, x0_pad=0, x1_pad=0, y0_pad=0, y1_pad=0
        ):
            width = pil_img.size[0] + x0_pad + x1_pad
            height = pil_img.size[1] + y0_pad + y1_pad

            result = PIL.Image.new(
                pil_img.mode, (width, height), background_color
            )
            result.paste(pil_img, (x0_pad, y0_pad))
            return result

        def _black_background(i):
            return (0, 0, 0) if i.mode == "RGB" else 0

        d_x0 = numpy.rint(max([0 - x0_new, 0])).astype(int)
        d_y0 = numpy.rint(max([0 - y0_new, 0])).astype(int)
        d_x1 = numpy.rint(max([x1_new - max_w, 0])).astype(int)
        d_y1 = numpy.rint(max([y1_new - max_h, 0])).astype(int)

        new_args = [
            _expand_img(
                k,
                _black_background(k),
                x0_pad=d_x0,
                x1_pad=d_x1,
                y0_pad=d_y0,
                y1_pad=d_y1,
            )
            for k in args
        ]

        new_args = [PIL.ImageOps.crop(k, border) for k in new_args]

        return new_args
