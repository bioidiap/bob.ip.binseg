#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torchvision.transforms.functional as VF
import random
import PIL
from PIL import Image
from torchvision.transforms.transforms import Lambda
from torchvision.transforms.transforms import Compose as TorchVisionCompose
import math
from math import floor
import warnings
import collections
import bob.core

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}
Iterable = collections.abc.Iterable

# Compose

class Compose:
    """Composes several transforms.

    Attributes
    ----------
    transforms : list
        list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

# Preprocessing

class CenterCrop:
    """
    Crop at the center.

    Attributes
    ----------
    size : int
        target size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, *args):
        return [VF.center_crop(img, self.size) for img in args]


class Crop:
    """
    Crop at the given coordinates.

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

    def __call__(self, *args):
        return [img.crop((self.j, self.i, self.j + self.w, self.i + self.h)) for img in args]

class Pad:
    """
    Constant padding

    Attributes
    ----------
    padding : int or tuple
        padding on each border. If a single int is provided this is used to pad all borders.
        If tuple of length 2 is provided this is the padding on left/right and top/bottom respectively.
        If a tuple of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.

    fill : int
        pixel fill value for constant fill. Default is 0. If a tuple of length 3, it is used to fill R, G, B channels respectively.
        This value is only used when the padding_mode is constant
    """
    def __init__(self, padding, fill=0):
        self.padding = padding
        self.fill = fill

    def __call__(self, *args):
        return [VF.pad(img, self.padding, self.fill, padding_mode='constant') for img in args]

class AutoLevel16to8:
    """Converts a 16-bit image to 8-bit representation using "auto-level"

    This transform assumes that the input images are gray-scaled.

    To auto-level, we calculate the maximum and the minimum of the image, and
    consider such a range should be mapped to the [0,255] range of the
    destination image.
    """
    def _process_one(self, img):
        return Image.fromarray(bob.core.convert(img, 'uint8', (0,255),
            img.getextrema()))

    def __call__(self, *args):
        return [self._process_one(img) for img in args]

class ToRGB:
    """Converts from any input format to RGB, using an ADAPTIVE conversion.

    This transform takes the input image and converts it to RGB using
    py:method:`Image.Image.convert`, with `mode='RGB'` and using all other
    defaults.  This may be aggressive if applied to 16-bit images without
    further considerations.
    """
    def __call__(self, *args):
        return [img.convert(mode="RGB") for img in args]

class ToTensor:
    """Converts :py:class:`PIL.Image.Image` to :py:class:`torch.Tensor` """
    def __call__(self, *args):
        return [VF.to_tensor(img) for img in args]


# Augmentations

class RandomHFlip:
    """
    Flips horizontally

    Attributes
    ----------
    prob : float
        probability at which imgage is flipped. Defaults to ``0.5``
    """
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, *args):
        if random.random() < self.prob:
            return [VF.hflip(img) for img in args]

        else:
            return args


class RandomVFlip:
    """
    Flips vertically

    Attributes
    ----------
    prob : float
        probability at which imgage is flipped. Defaults to ``0.5``
    """
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, *args):
        if random.random() < self.prob:
            return [VF.vflip(img) for img in args]

        else:
            return args


class RandomRotation:
    """
    Rotates by degree

    Attributes
    ----------
    degree_range : tuple
        range of degrees in which image and ground truth are rotated. Defaults to ``(-15, +15)``
    prob : float
        probability at which imgage is rotated. Defaults to ``0.5``
    """
    def __init__(self, degree_range = (-15, +15), prob = 0.5):
        self.prob = prob
        self.degree_range = degree_range

    def __call__(self, *args):
        if random.random() < self.prob:
            degree = random.randint(*self.degree_range)
            return [VF.rotate(img, degree, resample = Image.BILINEAR) for img in args]
        else:
            return args

class ColorJitter(object):
    """
    Randomly change the brightness, contrast, saturation and hue

    Attributes
    ----------
    brightness : float
        how much to jitter brightness. brightness_factor
        is chosen uniformly from ``[max(0, 1 - brightness), 1 + brightness]``.
    contrast : float
        how much to jitter contrast. contrast_factor
        is chosen uniformly from ``[max(0, 1 - contrast), 1 + contrast]``.
    saturation : float
        how much to jitter saturation. saturation_factor
        is chosen uniformly from ``[max(0, 1 - saturation), 1 + saturation]``.
    hue : float
        how much to jitter hue. hue_factor is chosen uniformly from
        ``[-hue, hue]``. Should be >=0 and <= 0.5
    prob : float
        probability at which the operation is applied
    """
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.02, hue=0.02, prob=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: VF.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: VF.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: VF.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: VF.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = TorchVisionCompose(transforms)

        return transform

    def __call__(self, *args):
        if random.random() < self.prob:
            transform = self.get_params(self.brightness, self.contrast,
                                        self.saturation, self.hue)
            trans_img = transform(args[0])
            return [trans_img, *args[1:]]
        else:
            return args


class RandomResizedCrop:
    """Crop to random size and aspect ratio.
    A crop of random size of the original size and a random aspect ratio of
    the original aspect ratio is made. This crop is finally resized to
    given size. This is popularly used to train the Inception networks.

    Attributes
    ----------
    size : int
        expected output size of each edge
    scale : tuple
        range of size of the origin size cropped. Defaults to ``(0.08, 1.0)``
    ratio : tuple
        range of aspect ratio of the origin aspect ratio cropped. Defaults to ``(3. / 4., 4. / 3.)``
    interpolation :
        Defaults to ``PIL.Image.BILINEAR``
    prob : float
        probability at which the operation is applied. Defaults to ``0.5``
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR, prob = 0.5):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.prob = prob

    @staticmethod
    def get_params(img, scale, ratio):
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if (in_ratio < min(ratio)):
            w = img.size[0]
            h = w / min(ratio)
        elif (in_ratio > max(ratio)):
            h = img.size[1]
            w = h * max(ratio)
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, *args):
        if random.random() < self.prob:
            imgs = []
            for img in args:
                i, j, h, w = self.get_params(img, self.scale, self.ratio)
                img = VF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
                imgs.append(img)
            return imgs
        else:
            return args

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class Resize:
    """Resize to given size.

    Attributes
    ----------
    size : tuple or int
        Desired output size. If size is a sequence like
        (h, w), output size will be matched to this. If size is an int,
        smaller edge of the image will be matched to this number.
        i.e, if height > width, then image will be rescaled to
        (size * height / width, size)
    interpolation : int
        Desired interpolation. Default is``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, *args):
        return [VF.resize(img, self.size, self.interpolation) for img in args]

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)
