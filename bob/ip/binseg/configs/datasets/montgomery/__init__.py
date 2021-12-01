#!/usr/bin/env python
# coding=utf-8


def _maker(protocol):

    from ....data.montgomery import dataset as raw
    from ....data.transforms import Resize
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Resize((512, 512))])


def _maker_augmented(protocol):

    from ....data.montgomery import dataset as raw
    from .. import make_subset
    from ....data.transforms import Compose as _compose
    from ....data.transforms import Resize as _resize
    from ....data.transforms import ColorJitter as _jitter
    from ....data.transforms import RandomHorizontalFlip as _hflip
    from ....data.transforms import RandomRotation as _rotation
    from ....data.transforms import GaussianBlur as _blur

    def make_augmented_dataset(subsets, train_transforms, all_transforms):
        retval = {}

        for key in subsets.keys():
            retval[key] = make_subset(subsets[key], transforms=all_transforms)
            if key == "train":
                retval["__train__"] = make_subset(
                    subsets[key],
                    transforms=train_transforms,
                )
            else:
                # also use it for validation during training
                if key == "validation":
                    retval["__valid__"] = retval[key]

        if (
            ("__train__" in retval)
            and ("__valid__" not in retval)
        ):
            # if the dataset does not have a validation set, we use the unaugmented
            # training set as validation set
            retval["__valid__"] = retval["__train__"]

        return retval


    return make_augmented_dataset(
               subsets=raw.subsets(protocol),
               all_transforms=[_resize((256, 256))],
               train_transforms=[_compose([
                                          _resize((256, 256)),
                                          _rotation(degrees=15, p=0.5),
                                          _hflip(p=0.5),
                                          _jitter(p=0.5),
                                          _blur(p=0.5),])])
