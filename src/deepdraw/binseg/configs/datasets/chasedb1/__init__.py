# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later


def _maker(protocol):
    from .....common.data.transforms import Crop
    from ....data.chasedb1 import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Crop(0, 18, 960, 960)])


def _maker_square(protocol, size):
    from .....common.data.transforms import Pad, Resize
    from ....data.chasedb1 import dataset as raw
    from .. import make_dataset as mk

    return mk(
        raw.subsets(protocol), [Pad((1, 20, 0, 20)), Resize((size, size))]
    )
