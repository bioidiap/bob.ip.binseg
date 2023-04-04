# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later


def _maker(protocol):
    from ....data.refuge import dataset as raw
    from ....data.transforms import CenterCrop, Pad, Resize
    from .. import make_dataset as mk

    # due to different sizes, we need to make the dataset twice
    train = mk(raw.subsets(protocol), [Resize(1539), Pad((21, 46, 22, 47))])
    # we'll keep "dev" and "test" from the next one
    retval = mk(raw.subsets(protocol), [CenterCrop(1632)])
    # and we keep the "train" set with the right transforms
    retval["train"] = train["train"]
    return retval


def _maker_square(protocol, size):
    from ....data.refuge import dataset as raw
    from ....data.transforms import CenterCrop, Pad, Resize
    from .. import make_dataset as mk

    # due to different sizes, we need to make the dataset twice
    train = mk(
        raw.subsets(protocol),
        [Resize(1539), Pad((21, 46, 22, 47)), Resize((size, size))],
    )
    # we'll keep "dev" and "test" from the next one
    retval = mk(raw.subsets(protocol), [CenterCrop(1632), Resize((size, size))])
    # and we keep the "train" set with the right transforms
    retval["train"] = train["train"]
    return retval
