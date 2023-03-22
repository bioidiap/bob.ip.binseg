# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from clapper.rc import UserDefaults


def load_rc() -> UserDefaults:
    """Returns global configuration variables."""
    return UserDefaults("deepdraw.toml")
