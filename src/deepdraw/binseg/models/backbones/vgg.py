# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import torchvision.models

try:
    # pytorch >= 1.12
    from torch.hub import load_state_dict_from_url
except ImportError:
    # pytorch < 1.12
    from torchvision.models.utils import load_state_dict_from_url


class VGG4Segmentation(torchvision.models.vgg.VGG):
    """Adaptation of base VGG functionality to U-Net style segmentation.

    This version of VGG is slightly modified so it can be used through
    torchvision's API.  It outputs intermediate features which are normally not
    output by the base VGG implementation, but are required for segmentation
    operations.


    Parameters
    ==========

    return_features : :py:class:`list`, Optional
        A list of integers indicating the feature layers to be returned from
        the original module.
    """

    def __init__(self, *args, **kwargs):
        self._return_features = kwargs.pop("return_features")
        super().__init__(*args, **kwargs)

    def forward(self, x):
        outputs = []
        # hardwiring of input
        outputs.append(x.shape[2:4])
        for index, m in enumerate(self.features):
            x = m(x)
            # extract layers
            if index in self._return_features:
                outputs.append(x)
        return outputs


def _make_vgg16_typeD_for_segmentation(
    pretrained, batch_norm, progress, **kwargs
):
    if pretrained:
        kwargs["init_weights"] = False

    model = VGG4Segmentation(
        torchvision.models.vgg.make_layers(
            torchvision.models.vgg.cfgs["D"],
            batch_norm=batch_norm,
        ),
        **kwargs,
    )

    if pretrained:
        weights = (
            torchvision.models.vgg.VGG16_Weights.DEFAULT.url
            if not batch_norm
            else torchvision.models.vgg.VGG16_BN_Weights.DEFAULT.url
        )

        state_dict = load_state_dict_from_url(weights, progress=progress)
        model.load_state_dict(state_dict)

    # erase VGG head (for classification), not used for segmentation
    delattr(model, "classifier")
    delattr(model, "avgpool")

    return model


def vgg16_for_segmentation(pretrained=False, progress=True, **kwargs):
    return _make_vgg16_typeD_for_segmentation(
        pretrained=pretrained, batch_norm=False, progress=progress, **kwargs
    )


vgg16_for_segmentation.__doc__ = torchvision.models.vgg16.__doc__


def vgg16_bn_for_segmentation(pretrained=False, progress=True, **kwargs):
    return _make_vgg16_typeD_for_segmentation(
        pretrained=pretrained, batch_norm=True, progress=progress, **kwargs
    )


vgg16_bn_for_segmentation.__doc__ = torchvision.models.vgg16_bn.__doc__
