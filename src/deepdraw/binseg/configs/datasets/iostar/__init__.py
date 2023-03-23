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
    from .....common.data.transforms import Gaussian_noise as noise
    from .....common.data.transforms import Resize
    from ....data.iostar import dataset as raw
    from .. import make_dataset as mk

    return mk(
        raw.subsets(protocol),
        [
            Resize((size, size)),
            jitter,
            gray,
            noise(0, 0.01),
        ],
    )
