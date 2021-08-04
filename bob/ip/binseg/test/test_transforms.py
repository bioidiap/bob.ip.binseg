#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random

import numpy
import PIL.Image
import pkg_resources
import torch
import torchvision.transforms.functional

from ..data.transforms import CenterCrop
from ..data.transforms import ColorJitter
from ..data.transforms import Compose
from ..data.transforms import Crop
from ..data.transforms import Pad
from ..data.transforms import RandomHorizontalFlip
from ..data.transforms import RandomRotation
from ..data.transforms import RandomVerticalFlip
from ..data.transforms import Resize
from ..data.transforms import SingleAutoLevel16to8
from ..data.transforms import ToTensor


def _create_img(size):
    t = torch.randn(size)
    pil = torchvision.transforms.functional.to_pil_image(t)
    return pil


def test_center_crop():

    # parameters
    im_size = (3, 22, 20)  # (planes, height, width)
    crop_size = (10, 12)  # (height, width)

    # test
    bh = (im_size[1] - crop_size[0]) // 2
    bw = (im_size[2] - crop_size[1]) // 2
    idx = (slice(bh, -bh), slice(bw, -bw), slice(0, im_size[0]))
    transforms = CenterCrop(crop_size)
    img, gt, mask = [_create_img(im_size) for i in range(3)]
    assert img.size == (im_size[2], im_size[1])  # confirms the above
    img_t, gt_t, mask_t = transforms(img, gt, mask)
    assert img_t.size == (crop_size[1], crop_size[0])  # confirms the above
    # notice that PIL->array does array.transpose(1, 2, 0)
    # so it creates an array that is (height, width, planes)
    assert numpy.all(numpy.array(img_t) == numpy.array(img)[idx])
    assert numpy.all(numpy.array(gt_t) == numpy.array(gt)[idx])
    assert numpy.all(numpy.array(mask_t) == numpy.array(mask)[idx])


def test_center_crop_uneven():

    # parameters
    im_size = (3, 23, 20)  # (planes, height, width)
    crop_size = (10, 13)  # (height, width)

    # test
    bh = (im_size[1] - crop_size[0]) // 2
    bw = (im_size[2] - crop_size[1]) // 2
    # when the crop size is uneven, this is what happens - notice here that the
    # image height is uneven, and the crop width as well - the attributions of
    # extra pixels will depend on what is uneven (original image or crop)
    idx = (slice(bh + 1, -bh), slice(bw + 1, -bw), slice(0, im_size[0]))
    transforms = CenterCrop(crop_size)
    img, gt, mask = [_create_img(im_size) for i in range(3)]
    assert img.size == (im_size[2], im_size[1])  # confirms the above
    img_t, gt_t, mask_t = transforms(img, gt, mask)
    assert img_t.size == (crop_size[1], crop_size[0])  # confirms the above
    # notice that PIL->array does array.transpose(1, 2, 0)
    # so it creates an array that is (height, width, planes)
    assert numpy.all(numpy.array(img_t) == numpy.array(img)[idx])
    assert numpy.all(numpy.array(gt_t) == numpy.array(gt)[idx])
    assert numpy.all(numpy.array(mask_t) == numpy.array(mask)[idx])


def test_pad_default():

    # parameters
    im_size = (3, 22, 20)  # (planes, height, width)
    pad_size = 2

    # test
    idx = (
        slice(pad_size, -pad_size),
        slice(pad_size, -pad_size),
        slice(0, im_size[0]),
    )
    transforms = Pad(pad_size)
    img, gt, mask = [_create_img(im_size) for i in range(3)]
    assert img.size == (im_size[2], im_size[1])  # confirms the above
    img_t, gt_t, mask_t = transforms(img, gt, mask)
    # notice that PIL->array does array.transpose(1, 2, 0)
    # so it creates an array that is (height, width, planes)
    assert numpy.all(numpy.array(img_t)[idx] == numpy.array(img))
    assert numpy.all(numpy.array(gt_t)[idx] == numpy.array(gt))
    assert numpy.all(numpy.array(mask_t)[idx] == numpy.array(mask))

    # checks that the border introduced with padding is all about "fill"
    img_t = numpy.array(img_t)
    img_t[idx] = 0
    # border_size_plane = img_t[:, :, 0].size - numpy.array(img)[:, :, 0].size
    assert img_t.sum() == 0

    gt_t = numpy.array(gt_t)
    gt_t[idx] = 0
    assert gt_t.sum() == 0

    mask_t = numpy.array(mask_t)
    mask_t[idx] = 0
    assert mask_t.sum() == 0


def test_pad_2tuple():

    # parameters
    im_size = (3, 22, 20)  # (planes, height, width)
    pad_size = (1, 2)  # left/right, top/bottom
    fill = (3, 4, 5)

    # test
    idx = (
        slice(pad_size[1], -pad_size[1]),
        slice(pad_size[0], -pad_size[0]),
        slice(0, im_size[0]),
    )
    transforms = Pad(pad_size, fill)
    img, gt, mask = [_create_img(im_size) for i in range(3)]
    assert img.size == (im_size[2], im_size[1])  # confirms the above
    img_t, gt_t, mask_t = transforms(img, gt, mask)
    # notice that PIL->array does array.transpose(1, 2, 0)
    # so it creates an array that is (height, width, planes)
    assert numpy.all(numpy.array(img_t)[idx] == numpy.array(img))
    assert numpy.all(numpy.array(gt_t)[idx] == numpy.array(gt))
    assert numpy.all(numpy.array(mask_t)[idx] == numpy.array(mask))

    # checks that the border introduced with padding is all about "fill"
    img_t = numpy.array(img_t)
    img_t[idx] = 0
    border_size_plane = img_t[:, :, 0].size - numpy.array(img)[:, :, 0].size
    expected_sum = sum((fill[k] * border_size_plane) for k in range(3))
    assert img_t.sum() == expected_sum

    gt_t = numpy.array(gt_t)
    gt_t[idx] = 0
    assert gt_t.sum() == expected_sum

    mask_t = numpy.array(mask_t)
    mask_t[idx] = 0
    assert mask_t.sum() == expected_sum


def test_pad_4tuple():

    # parameters
    im_size = (3, 22, 20)  # (planes, height, width)
    pad_size = (1, 2, 3, 4)  # left, top, right, bottom
    fill = (3, 4, 5)

    # test
    idx = (
        slice(pad_size[1], -pad_size[3]),
        slice(pad_size[0], -pad_size[2]),
        slice(0, im_size[0]),
    )
    transforms = Pad(pad_size, fill)
    img, gt, mask = [_create_img(im_size) for i in range(3)]
    assert img.size == (im_size[2], im_size[1])  # confirms the above
    img_t, gt_t, mask_t = transforms(img, gt, mask)
    # notice that PIL->array does array.transpose(1, 2, 0)
    # so it creates an array that is (height, width, planes)
    assert numpy.all(numpy.array(img_t)[idx] == numpy.array(img))
    assert numpy.all(numpy.array(gt_t)[idx] == numpy.array(gt))
    assert numpy.all(numpy.array(mask_t)[idx] == numpy.array(mask))

    # checks that the border introduced with padding is all about "fill"
    img_t = numpy.array(img_t)
    img_t[idx] = 0
    border_size_plane = img_t[:, :, 0].size - numpy.array(img)[:, :, 0].size
    expected_sum = sum((fill[k] * border_size_plane) for k in range(3))
    assert img_t.sum() == expected_sum

    gt_t = numpy.array(gt_t)
    gt_t[idx] = 0
    assert gt_t.sum() == expected_sum

    mask_t = numpy.array(mask_t)
    mask_t[idx] = 0
    assert mask_t.sum() == expected_sum


def test_resize_downscale_w():

    # parameters
    im_size = (3, 22, 20)  # (planes, height, width)
    new_size = 10  # (smallest edge)

    # test
    transforms = Resize(new_size)
    img, gt, mask = [_create_img(im_size) for i in range(3)]
    assert img.size == (im_size[2], im_size[1])  # confirms the above
    img_t, gt_t, mask_t = transforms(img, gt, mask)
    new_size = (new_size, (new_size * im_size[1]) / im_size[2])
    assert img_t.size == new_size
    assert gt_t.size == new_size
    assert mask_t.size == new_size


def test_resize_downscale_hw():

    # parameters
    im_size = (3, 22, 20)  # (planes, height, width)
    new_size = (10, 12)  # (height, width)

    # test
    transforms = Resize(new_size)
    img, gt, mask = [_create_img(im_size) for i in range(3)]
    assert img.size == (im_size[2], im_size[1])  # confirms the above
    img_t, gt_t, mask_t = transforms(img, gt, mask)
    assert img_t.size == (new_size[1], new_size[0])
    assert gt_t.size == (new_size[1], new_size[0])
    assert mask_t.size == (new_size[1], new_size[0])


def test_crop():

    # parameters
    im_size = (3, 22, 20)  # (planes, height, width)
    crop_size = (3, 2, 10, 12)  # (upper, left, height, width)

    # test
    idx = (
        slice(crop_size[0], crop_size[0] + crop_size[2]),
        slice(crop_size[1], crop_size[1] + crop_size[3]),
        slice(0, im_size[0]),
    )
    transforms = Crop(*crop_size)
    img, gt, mask = [_create_img(im_size) for i in range(3)]
    assert img.size == (im_size[2], im_size[1])  # confirms the above
    img_t, gt_t, mask_t = transforms(img, gt, mask)
    # notice that PIL->array does array.transpose(1, 2, 0)
    # so it creates an array that is (height, width, planes)
    assert numpy.all(numpy.array(img_t) == numpy.array(img)[idx])
    assert numpy.all(numpy.array(gt_t) == numpy.array(gt)[idx])
    assert numpy.all(numpy.array(mask_t) == numpy.array(mask)[idx])


def test_to_tensor():

    transforms = ToTensor()
    img, gt, mask = [_create_img((3, 5, 5)) for i in range(3)]
    gt = gt.convert("1", dither=None)
    mask = mask.convert("1", dither=None)
    img_t, gt_t, mask_t = transforms(img, gt, mask)
    assert img_t.dtype == torch.float32
    assert gt_t.dtype == torch.float32
    assert mask_t.dtype == torch.float32


def test_horizontal_flip():

    transforms = RandomHorizontalFlip(p=1)

    im_size = (3, 24, 42)  # (planes, height, width)
    img, gt, mask = [_create_img(im_size) for i in range(3)]
    img_t, gt_t, mask_t = transforms(img, gt, mask)

    # notice that PIL->array does array.transpose(1, 2, 0)
    # so it creates an array that is (height, width, planes)
    assert numpy.all(numpy.flip(img_t, axis=1) == numpy.array(img))
    assert numpy.all(numpy.flip(gt_t, axis=1) == numpy.array(gt))
    assert numpy.all(numpy.flip(mask_t, axis=1) == numpy.array(mask))


def test_vertical_flip():

    transforms = RandomVerticalFlip(p=1)

    im_size = (3, 24, 42)  # (planes, height, width)
    img, gt, mask = [_create_img(im_size) for i in range(3)]
    img_t, gt_t, mask_t = transforms(img, gt, mask)

    # notice that PIL->array does array.transpose(1, 2, 0)
    # so it creates an array that is (height, width, planes)
    assert numpy.all(numpy.flip(img_t, axis=0) == numpy.array(img))
    assert numpy.all(numpy.flip(gt_t, axis=0) == numpy.array(gt))
    assert numpy.all(numpy.flip(mask_t, axis=0) == numpy.array(mask))


def test_rotation():

    im_size = (3, 24, 42)  # (planes, height, width)
    transforms = RandomRotation(degrees=90, p=1)
    img = _create_img(im_size)

    # asserts all images are rotated the same
    # and they are different from the original
    random.seed(42)
    img1_t, img2_t, img3_t = transforms(img, img, img)
    assert img1_t.size == (im_size[2], im_size[1])
    assert numpy.all(numpy.array(img1_t) == numpy.array(img2_t))
    assert numpy.all(numpy.array(img1_t) == numpy.array(img3_t))
    assert numpy.any(numpy.array(img1_t) != numpy.array(img))

    # asserts two random transforms are not the same
    (img_t2,) = transforms(img)
    assert numpy.any(numpy.array(img_t2) != numpy.array(img1_t))


def test_color_jitter():

    im_size = (3, 24, 42)  # (planes, height, width)
    transforms = ColorJitter(p=1)
    img = _create_img(im_size)

    # asserts only the first image is jittered
    # and it is different from the original
    # all others match the input data
    random.seed(42)
    img1_t, img2_t, img3_t = transforms(img, img, img)
    assert img1_t.size == (im_size[2], im_size[1])
    assert numpy.any(numpy.array(img1_t) != numpy.array(img))
    assert numpy.any(numpy.array(img1_t) != numpy.array(img2_t))
    assert numpy.all(numpy.array(img2_t) == numpy.array(img3_t))
    assert numpy.all(numpy.array(img2_t) == numpy.array(img))

    # asserts two random transforms are not the same
    img1_t2, img2_t2, img3_t2 = transforms(img, img, img)
    assert numpy.any(numpy.array(img1_t2) != numpy.array(img1_t))
    assert numpy.all(numpy.array(img2_t2) == numpy.array(img))
    assert numpy.all(numpy.array(img3_t2) == numpy.array(img))


def test_compose():

    transforms = Compose(
        [
            RandomVerticalFlip(p=1),
            RandomHorizontalFlip(p=1),
            RandomVerticalFlip(p=1),
            RandomHorizontalFlip(p=1),
        ]
    )

    img, gt, mask = [_create_img((3, 24, 42)) for i in range(3)]
    img_t, gt_t, mask_t = transforms(img, gt, mask)
    assert numpy.all(numpy.array(img_t) == numpy.array(img))
    assert numpy.all(numpy.array(gt_t) == numpy.array(gt))
    assert numpy.all(numpy.array(mask_t) == numpy.array(mask))


def test_16bit_autolevel():

    path = pkg_resources.resource_filename(
        __name__, os.path.join("data", "img-16bit.png")
    )
    # the way to load a 16-bit PNG image correctly, according to:
    # https://stackoverflow.com/questions/32622658/read-16-bit-png-image-file-using-python
    # https://github.com/python-pillow/Pillow/issues/3011
    img = PIL.Image.fromarray(numpy.array(PIL.Image.open(path)).astype("uint16"))
    assert img.mode == "I;16"
    assert img.getextrema() == (0, 65281)

    timg = SingleAutoLevel16to8()(img)
    assert timg.mode == "L"
    assert timg.getextrema() == (0, 255)
    # timg.show()
    # import ipdb; ipdb.set_trace()
