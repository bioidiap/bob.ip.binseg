#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
# SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
# SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
# SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
# SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def faster_rcnn(
    weights="FasterRCNN_ResNet50_FPN_Weights.COCO_V1", progress=True
):

    """Build Faster RCNN implementation on PyTorch.

    Parameters
    ----------
    weights : :py:class:`str`, Optional
        If set to None, then it will not load a pre-trained backbone
        (not the head) for the Faster-RCNN network. Otherwise, it will load
        the pretrained weights specified as a string.
    progress : :py:class:`bool`, Optional
        If set to ``True``, and you decided to use a ``pretrained_backbone``,
        then, shows a progress bar of the backbone model downloading if
        download is necesssary.
    Returns
    -------
    module : :py:class:`torch.nn.Module`
        Network model for Faster R-CNN
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights, progress=progress
    )

    num_classes = 2  # 1 class (person) + background

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
