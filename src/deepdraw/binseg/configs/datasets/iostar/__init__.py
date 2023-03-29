# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later
import torchvision.transforms as T

gray = T.Grayscale(num_output_channels=3)
jitter = T.ColorJitter(hue=0.05)


def _maker(protocol):
    from ....data.iostar import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [])


def _maker_square(protocol, size):
    from .....common.data.transforms import Resize
    from ....data.iostar import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Resize((size, size))])


def _semi_data_augmentation(protocol, size):
    from .....common.data.transforms import Compose
    from .....common.data.transforms import Gaussian_noise as noise
    from .....common.data.transforms import Resize
    from ....data.iostar import dataset as raw
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
        all_transforms=[Resize((size, size))],
        train_transforms=[
            Compose([Resize((size, size)), jitter, gray, noise(0, 0.01)])
        ],
    )
