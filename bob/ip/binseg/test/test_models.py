#!/usr/bin/env python
# coding=utf-8

"""Tests model loading"""


from ..models.backbones.vgg import VGG4Segmentation
from ..models.normalizer import TorchVisionNormalizer


def test_driu():

    from ..models.driu import DRIU
    from ..models.driu import driu

    model = driu(pretrained_backbone=True, progress=True)
    assert len(model) == 3
    assert type(model[0]) == TorchVisionNormalizer
    assert type(model[1]) == VGG4Segmentation  # backbone
    assert type(model[2]) == DRIU  # head

    model = driu(pretrained_backbone=False)
    assert len(model) == 2
    assert type(model[0]) == VGG4Segmentation  # backbone
    assert type(model[1]) == DRIU  # head


def test_driu_bn():

    from ..models.driu_bn import DRIUBN
    from ..models.driu_bn import driu_bn

    model = driu_bn(pretrained_backbone=True, progress=True)
    assert len(model) == 3
    assert type(model[0]) == TorchVisionNormalizer
    assert type(model[1]) == VGG4Segmentation  # backbone
    assert type(model[2]) == DRIUBN  # head

    model = driu_bn(pretrained_backbone=False)
    assert len(model) == 2
    assert type(model[0]) == VGG4Segmentation  # backbone
    assert type(model[1]) == DRIUBN  # head


def test_driu_od():

    from ..models.driu_od import DRIUOD
    from ..models.driu_od import driu_od

    model = driu_od(pretrained_backbone=True, progress=True)
    assert len(model) == 3
    assert type(model[0]) == TorchVisionNormalizer
    assert type(model[1]) == VGG4Segmentation  # backbone
    assert type(model[2]) == DRIUOD  # head

    model = driu_od(pretrained_backbone=False)
    assert len(model) == 2
    assert type(model[0]) == VGG4Segmentation  # backbone
    assert type(model[1]) == DRIUOD  # head


def test_driu_pix():

    from ..models.driu_pix import DRIUPIX
    from ..models.driu_pix import driu_pix

    model = driu_pix(pretrained_backbone=True, progress=True)
    assert len(model) == 3
    assert type(model[0]) == TorchVisionNormalizer
    assert type(model[1]) == VGG4Segmentation  # backbone
    assert type(model[2]) == DRIUPIX  # head

    model = driu_pix(pretrained_backbone=False)
    assert len(model) == 2
    assert type(model[0]) == VGG4Segmentation  # backbone
    assert type(model[1]) == DRIUPIX  # head


def test_unet():

    from ..models.unet import UNet
    from ..models.unet import unet

    model = unet(pretrained_backbone=True, progress=True)
    assert len(model) == 3
    assert type(model[0]) == TorchVisionNormalizer
    assert type(model[1]) == VGG4Segmentation  # backbone
    assert type(model[2]) == UNet  # head

    model = unet(pretrained_backbone=False)
    assert len(model) == 2
    assert type(model[0]) == VGG4Segmentation  # backbone
    assert type(model[1]) == UNet  # head


def test_hed():

    from ..models.hed import HED
    from ..models.hed import hed

    model = hed(pretrained_backbone=True, progress=True)
    assert len(model) == 3
    assert type(model[0]) == TorchVisionNormalizer
    assert type(model[1]) == VGG4Segmentation  # backbone
    assert type(model[2]) == HED  # head

    model = hed(pretrained_backbone=False)
    assert len(model) == 2
    assert type(model[0]) == VGG4Segmentation  # backbone
    assert type(model[1]) == HED  # head


def test_m2unet():

    from ..models.backbones.mobilenetv2 import MobileNetV24Segmentation
    from ..models.m2unet import M2UNet
    from ..models.m2unet import m2unet

    model = m2unet(pretrained_backbone=True, progress=True)
    assert len(model) == 3
    assert type(model[0]) == TorchVisionNormalizer
    assert type(model[1]) == MobileNetV24Segmentation  # backbone
    assert type(model[2]) == M2UNet  # head

    model = m2unet(pretrained_backbone=False)
    assert len(model) == 2
    assert type(model[0]) == MobileNetV24Segmentation  # backbone
    assert type(model[1]) == M2UNet  # head


def test_resunet50():

    from ..models.backbones.resnet import ResNet4Segmentation
    from ..models.resunet import ResUNet
    from ..models.resunet import resunet50

    model = resunet50(pretrained_backbone=True, progress=True)
    assert len(model) == 3
    assert type(model[0]) == TorchVisionNormalizer
    assert type(model[1]) == ResNet4Segmentation  # backbone
    assert type(model[2]) == ResUNet  # head

    model = resunet50(pretrained_backbone=False)
    assert len(model) == 2
    assert type(model[0]) == ResNet4Segmentation  # backbone
    assert type(model[1]) == ResUNet  # head
    print(model)
