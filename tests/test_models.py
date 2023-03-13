# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests model loading."""


from deepdraw.binseg.models.backbones.vgg import VGG4Segmentation
from deepdraw.binseg.models.normalizer import TorchVisionNormalizer


def test_driu():
    from deepdraw.binseg.models.driu import DRIU, driu

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
    from deepdraw.binseg.models.driu_bn import DRIUBN, driu_bn

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
    from deepdraw.binseg.models.driu_od import DRIUOD, driu_od

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
    from deepdraw.binseg.models.driu_pix import DRIUPIX, driu_pix

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
    from deepdraw.binseg.models.unet import UNet, unet

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
    from deepdraw.binseg.models.hed import HED, hed

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
    from deepdraw.binseg.models.backbones.mobilenetv2 import (
        MobileNetV24Segmentation,
    )
    from deepdraw.binseg.models.m2unet import M2UNet, m2unet

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
    from deepdraw.binseg.models.backbones.resnet import ResNet4Segmentation
    from deepdraw.binseg.models.resunet import ResUNet, resunet50

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


def test_fasterrcnn():
    import torchvision

    from deepdraw.detect.models.faster_rcnn import faster_rcnn

    model = faster_rcnn()
    assert type(model) == torchvision.models.detection.faster_rcnn.FasterRCNN
    assert (
        type(model.backbone)
        == torchvision.models.detection.backbone_utils.BackboneWithFPN
    )
    assert (
        type(model.roi_heads.box_predictor)
        == torchvision.models.detection.faster_rcnn.FastRCNNPredictor
    )