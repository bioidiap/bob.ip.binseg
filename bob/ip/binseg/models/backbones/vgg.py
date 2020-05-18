#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torchvision.models.vgg


class VGG4Segmentation(torchvision.models.vgg.VGG):
    """Adaptation of base VGG functionality to U-Net style segmentation

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
        super(VGG4Segmentation, self).__init__(*args, **kwargs)

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


def _vgg_for_segmentation(
    arch, cfg, batch_norm, pretrained, progress, **kwargs
):

    if pretrained:
        kwargs["init_weights"] = False

    model = VGG4Segmentation(
        torchvision.models.vgg.make_layers(
            torchvision.models.vgg.cfgs[cfg], batch_norm=batch_norm
        ),
        **kwargs
    )

    if pretrained:
        state_dict = torchvision.models.vgg.load_state_dict_from_url(
            torchvision.models.vgg.model_urls[arch], progress=progress
        )
        model.load_state_dict(state_dict)

    # erase VGG head (for classification), not used for segmentation
    delattr(model, 'classifier')
    delattr(model, 'avgpool')

    return model


def vgg16_for_segmentation(pretrained=False, progress=True, **kwargs):
    return _vgg_for_segmentation(
        "vgg16", "D", False, pretrained, progress, **kwargs
    )


vgg16_for_segmentation.__doc__ = torchvision.models.vgg16.__doc__


def vgg16_bn_for_segmentation(pretrained=False, progress=True, **kwargs):
    return _vgg_for_segmentation(
        "vgg16_bn", "D", True, pretrained, progress, **kwargs
    )


vgg16_bn_for_segmentation.__doc__ = torchvision.models.vgg16_bn.__doc__
