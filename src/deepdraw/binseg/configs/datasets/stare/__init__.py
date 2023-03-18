# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later


def _maker(protocol):
    from .....common.data.transforms import Pad
    from ....data.stare import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((2, 1, 2, 2))])


def _maker_square(protocol, size):
    from .....common.data.transforms import Pad, Resize
    from ....data.stare import dataset as raw
    from .. import make_dataset as mk

    return mk(
        raw.subsets(protocol), [Pad((1, 48, 0, 48)), Resize((size, size))]
    )
