# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later


def _maker(protocol):
    from .....binseg.data.transforms import Pad
    from ....data.drionsdb import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((4, 8, 4, 8))])


def _maker_square(protocol, size):
    from .....binseg.data.transforms import Pad, Resize
    from ....data.drionsdb import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((0, 100)), Resize((size, size))])
