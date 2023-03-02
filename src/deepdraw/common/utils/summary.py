# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Adapted from https://github.com/pytorch/pytorch/issues/2001#issuecomment-405675488
from functools import reduce

from torch.nn.modules.module import _addindent


def summary(model):
    """Counts the number of parameters in each model layer.

    Parameters
    ----------

    model : :py:class:`torch.nn.Module`
        model to summarize

    Returns
    -------

    repr : str
        a multiline string representation of the network

    nparam : int
        number of parameters
    """

    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, "dtype"):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        main_str += f", {total_params:,} params"
        return main_str, total_params

    return repr(model)
