# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later


def _maker(protocol):
    from .....common.data.transforms import CenterCrop as ccrop
    from ....data.drishtigs1 import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [ccrop((1760, 2048))])


def _maker_square(protocol, size):
    from .....common.data.transforms import CenterCrop as ccrop
    from .....common.data.transforms import Pad, Resize
    from ....data.drishtigs1 import dataset as raw
    from .. import make_dataset as mk

    return mk(
        raw.subsets(protocol),
        [ccrop((1760, 2048)), Pad((0, 144)), Resize((size, size))],
    )
