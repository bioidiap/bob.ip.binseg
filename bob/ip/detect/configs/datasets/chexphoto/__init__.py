#!/usr/bin/env python
# coding=utf-8


def _maker(protocol):

    from .....common.data.transforms import Resize
    from ....data.chexphoto import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Resize((256, 256))])


def _maker_augmented(protocol):

    from .....common.data.transforms import ColorJitter as _jitter
    from .....common.data.transforms import Compose as _compose
    from .....common.data.transforms import GaussianBlur as _blur
    from .....common.data.transforms import Resize as _resize
    from ....data.chexphoto import dataset as raw
    from .. import make_subset

    def _mk_aug_subset(subsets, train_transforms, all_transforms):
        retval = {}

        for key in subsets.keys():
            retval[key] = make_subset(subsets[key], transforms=all_transforms)
            if key == "train":
                retval["__train__"] = make_subset(
                    subsets[key],
                    transforms=train_transforms,
                )
            else:
                if key == "validation":
                    retval["__valid__"] = retval[key]

        if ("__train__" in retval) and ("__valid__" not in retval):
            retval["__valid__"] = retval["__train__"]

        return retval

    return _mk_aug_subset(
        subsets=raw.subsets(protocol),
        all_transforms=[_resize((256, 256))],
        train_transforms=[
            _compose(
                [
                    _resize((256, 256)),
                    _jitter(p=0.5),
                    _blur(p=0.5),
                ]
            )
        ],
    )
