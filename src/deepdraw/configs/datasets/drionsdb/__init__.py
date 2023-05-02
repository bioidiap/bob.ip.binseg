# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later


def _maker(protocol):
    from ....data.drionsdb import dataset as raw
    from ....data.transforms import Pad
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((4, 8, 4, 8))])


def _maker_square(protocol, size):
    from ....data.drionsdb import dataset as raw
    from ....data.transforms import Pad, Resize
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((0, 100)), Resize((size, size))])
