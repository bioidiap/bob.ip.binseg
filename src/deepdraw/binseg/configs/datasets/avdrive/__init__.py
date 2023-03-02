# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later


def _maker(protocol):
    from .....common.data.transforms import CenterCrop as ccrop
    from ....data.drive import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [ccrop((544, 544))])
