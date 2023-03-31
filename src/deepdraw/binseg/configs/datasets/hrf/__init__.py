# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later


def _maker_1168(protocol):
    from .....common.data.transforms import Crop, Resize
    from ....data.hrf import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Crop(0, 108, 2336, 3296), Resize(1168)])


def _maker(protocol):
    from .....common.data.transforms import Crop
    from ....data.hrf import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Crop(0, 108, 2336, 3296)])


def _maker_square_768(protocol):
    from .....common.data.transforms import Pad, Resize
    from ....data.hrf import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((0, 584)), Resize((768, 768))])


def _maker_square_1024(protocol):
    from .....common.data.transforms import Pad, Resize
    from ....data.hrf import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((0, 584)), Resize((1024, 1024))])


def _maker_square(protocol, size):
    from .....common.data.transforms import Pad, Resize
    from ....data.hrf import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((0, 584)), Resize((size, size))])


def _semi_data_augmentation(protocol, size):
    from .....common.data.transforms import ColorJitter as _jitter
    from .....common.data.transforms import Compose as _compose
    from .....common.data.transforms import Gaussian_noise as _noise
    from .....common.data.transforms import Grayscale as _gray
    from .....common.data.transforms import Pad, Resize
    from ....data.hrf import dataset as raw
    from .. import make_subset

    # resize all data
    # add gaussian noise, color jitter and grayscale to the training data

    def make_aug_subsets(subsets, train_transforms, all_transforms):
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

    return make_aug_subsets(
        subsets=raw.subsets(protocol),
        all_transforms=[Pad((0, 584)), Resize((size, size))],
        train_transforms=[
            _compose(
                [
                    Pad((0, 584)),
                    Resize((size, size)),
                    _jitter(
                        p=1, brightness=0.3, contrast=0, saturation=0, hue=0
                    ),
                    _gray(3),
                    _noise(0, 0.01),
                ]
            )
        ],
    )
