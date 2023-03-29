# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Functions for ramping hyperparameters up or down Each function takes the
current training step or epoch, and the ramp length in the same format, and
returns a multiplier between 0 and 1."""

import numpy

consistency = 15
consistency_rampup = 100

"""
Parameters
    ----------

    consistency : :control the consistency weight

    consistency_rampup : :float, the weight of consistency loss in the whole loss. At the beginning it is 0, since the teacher model may not produce good predict and then increases very slowly.

"""


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242."""
    if rampup_length == 0:
        return 1.0
    else:
        current = numpy.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(numpy.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup."""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983."""
    assert 0 <= current <= rampdown_length
    return float(0.5 * (numpy.cos(numpy.pi * current / rampdown_length) + 1))


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def update_ema_variables(S_model, T_model, global_step):
    # set alpha to 0.99 in first 50 steps and change to 0.999 later
    if global_step < 50:
        alpha = 0.99
    else:
        alpha = 0.999
    for T_param, S_param in zip(T_model.parameters(), S_model.parameters()):
        T_param.data = alpha * T_param.data + (1 - alpha) * S_param.data
