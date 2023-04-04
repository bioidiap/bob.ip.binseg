# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later


def _maker_1168(protocol):
    from ....data.hrf import dataset as raw
    from ....data.transforms import Crop, Resize
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Crop(0, 108, 2336, 3296), Resize(1168)])


def _maker(protocol):
    from ....data.hrf import dataset as raw
    from ....data.transforms import Crop
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Crop(0, 108, 2336, 3296)])


def _maker_square_768(protocol):
    from ....data.hrf import dataset as raw
    from ....data.transforms import Pad, Resize
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((0, 584)), Resize((768, 768))])


def _maker_square_1024(protocol):
    from ....data.hrf import dataset as raw
    from ....data.transforms import Pad, Resize
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((0, 584)), Resize((1024, 1024))])


def _maker_square(protocol, size):
    from ....data.hrf import dataset as raw
    from ....data.transforms import Pad, Resize
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((0, 584)), Resize((size, size))])
