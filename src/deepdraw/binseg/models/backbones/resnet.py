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


class ResNet4Segmentation(torchvision.models.resnet.ResNet):
    """Adaptation of base ResNet functionality to U-Net style segmentation.

    This version of ResNet is slightly modified so it can be used through
    torchvision's API.  It outputs intermediate features which are normally not
    output by the base ResNet implementation, but are required for segmentation
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
            if index in self.return_features:
                outputs.append(x)
        return outputs


def resnet50_for_segmentation(pretrained=False, progress=True, **kwargs):
    model = ResNet4Segmentation(
        torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], **kwargs
    )

    if pretrained:
        state_dict = load_state_dict_from_url(
            torchvision.models.resnet.ResNet50_Weights.DEFAULT.url,
            progress=progress,
        )
        model.load_state_dict(state_dict)

    # erase ResNet head (for classification), not used for segmentation
    delattr(model, "avgpool")
    delattr(model, "fc")

    return model


resnet50_for_segmentation.__doc__ = torchvision.models.resnet50.__doc__
