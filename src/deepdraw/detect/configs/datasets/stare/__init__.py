# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later


def _maker(protocol, raw=None):
    from .....binseg.data.stare import dataset as _raw
    from .....common.data.transforms import Pad

    raw = raw or _raw  # allows user to recreate dataset for testing purposes
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((2, 1, 2, 2))])
