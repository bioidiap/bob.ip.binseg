#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def faster_rcnn(pretrained_backbone=True, progress=True):
    """Build Faster RCNN implementation on PyTorch.

    Parameters
    ----------
    pretrained_backbone : :py:class:`bool`, Optional
        If set to ``True``, then loads a pre-trained version of the backbone
        (not the head) for the DRIU network using VGG-16 trained for ImageNet
        classification.
    progress : :py:class:`bool`, Optional
        If set to ``True``, and you decided to use a ``pretrained_backbone``,
        then, shows a progress bar of the backbone model downloading if
        download is necesssary.
    Returns
    -------
    module : :py:class:`torch.nn.Module`
        Network model for Faster R-CNN
    """
    model = (
        torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained_backbone,
                                                             progress=progress))

    num_classes = 2  # 1 class (person) + background

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
