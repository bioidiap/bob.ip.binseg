# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import torchvision.models
import torchvision.models.mobilenetv2

try:
    # pytorch >= 1.12
    from torch.hub import load_state_dict_from_url
except ImportError:
    # pytorch < 1.12
    from torchvision.models.utils import load_state_dict_from_url


class MobileNetV24Segmentation(torchvision.models.mobilenetv2.MobileNetV2):
    """Adaptation of base MobileNetV2 functionality to U-Net style
    segmentation.

    This version of MobileNetV2 is slightly modified so it can be used through
    torchvision's API.  It outputs intermediate features which are normally not
    output by the base MobileNetV2 implementation, but are required for
    segmentation operations.


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
        # hw of input, needed for DRIU and HED
        outputs.append(x.shape[2:4])
        outputs.append(x)
        for index, m in enumerate(self.features):
            x = m(x)
            # extract layers
            if index in self._return_features:
                outputs.append(x)
        return outputs


def mobilenet_v2_for_segmentation(pretrained=False, progress=True, **kwargs):
    model = MobileNetV24Segmentation(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(
            torchvision.models.mobilenetv2.MobileNet_V2_Weights.DEFAULT.url,
            progress=progress,
        )
        model.load_state_dict(state_dict)

    # erase MobileNetV2 head (for classification), not used for segmentation
    delattr(model, "classifier")

    return_features = kwargs.get("return_features")
    if return_features is not None:
        model.features = model.features[: (max(return_features) + 1)]

    return model


mobilenet_v2_for_segmentation.__doc__ = torchvision.models.mobilenetv2.__doc__
