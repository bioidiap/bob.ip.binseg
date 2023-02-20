#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
# SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
# SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
# SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
# SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Implementation of the `AdaBound optimizer.

<https://github.com/Luolc/AdaBound/blob/master/adabound/adabound.py>`::

    @inproceedings{Luo2019AdaBound,
      author = {Luo, Liangchen and Xiong, Yuanhao and Liu, Yan and Sun, Xu},
      title = {Adaptive Gradient Methods with Dynamic Bound of Learning Rate},
      booktitle = {Proceedings of the 7th International Conference on Learning Representations},
      month = {May},
      year = {2019},
      address = {New Orleans, Louisiana}
    }
"""

import math

import torch
import torch.optim


class AdaBound(torch.optim.Optimizer):
    """Implements the AdaBound algorithm.

    Parameters
    ----------

    params : list
        Iterable of parameters to optimize or dicts defining parameter groups

    lr : :obj:`float`, optional
        Adam learning rate

    betas : :obj:`tuple`, optional
        Coefficients (as a 2-tuple of floats) used for computing running
        averages of gradient and its square

    final_lr : :obj:`float`, optional
        Final (SGD) learning rate

    gamma : :obj:`float`, optional
        Convergence speed of the bound functions

    eps : :obj:`float`, optional
        Term added to the denominator to improve numerical stability

    weight_decay : :obj:`float`, optional
        Weight decay (L2 penalty)

    amsbound : :obj:`bool`, optional
        Whether to use the AMSBound variant of this algorithm
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        final_lr=0.1,
        gamma=1e-3,
        eps=1e-8,
        weight_decay=0,
        amsbound=False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= final_lr:
            raise ValueError(f"Invalid final learning rate: {final_lr}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        defaults = dict(
            lr=lr,
            betas=betas,
            final_lr=final_lr,
            gamma=gamma,
            eps=eps,
            weight_decay=weight_decay,
            amsbound=amsbound,
        )
        super().__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group["lr"], self.param_groups))

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsbound", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters
        ----------

        closure : :obj:`callable`, optional
            A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsbound = group["amsbound"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsbound:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = (
                    group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                )

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group["final_lr"] * group["lr"] / base_lr
                lower_bound = final_lr * (
                    1 - 1 / (group["gamma"] * state["step"] + 1)
                )
                upper_bound = final_lr * (
                    1 + 1 / (group["gamma"] * state["step"])
                )
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(
                    exp_avg
                )

                p.data.add_(-step_size)

        return loss


class AdaBoundW(torch.optim.Optimizer):
    """Implements AdaBound algorithm with Decoupled Weight Decay (See
    https://arxiv.org/abs/1711.05101)

    Parameters
    ----------

    params : list
        Iterable of parameters to optimize or dicts defining parameter groups

    lr : :obj:`float`, optional
        Adam learning rate

    betas : :obj:`tuple`, optional
        Coefficients (as a 2-tuple of floats) used for computing running
        averages of gradient and its square

    final_lr : :obj:`float`, optional
        Final (SGD) learning rate

    gamma : :obj:`float`, optional
        Convergence speed of the bound functions

    eps : :obj:`float`, optional
        Term added to the denominator to improve numerical stability

    weight_decay : :obj:`float`, optional
        Weight decay (L2 penalty)

    amsbound : :obj:`bool`, optional
        Whether to use the AMSBound variant of this algorithm
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        final_lr=0.1,
        gamma=1e-3,
        eps=1e-8,
        weight_decay=0,
        amsbound=False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= final_lr:
            raise ValueError(f"Invalid final learning rate: {final_lr}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        defaults = dict(
            lr=lr,
            betas=betas,
            final_lr=final_lr,
            gamma=gamma,
            eps=eps,
            weight_decay=weight_decay,
            amsbound=amsbound,
        )
        super().__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group["lr"], self.param_groups))

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsbound", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters
        ----------

        closure : :obj:`callable`, optional
            A closure that reevaluates the model and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsbound = group["amsbound"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsbound:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = (
                    group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                )

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to
                # apply lr decay
                final_lr = group["final_lr"] * group["lr"] / base_lr
                lower_bound = final_lr * (
                    1 - 1 / (group["gamma"] * state["step"] + 1)
                )
                upper_bound = final_lr * (
                    1 + 1 / (group["gamma"] * state["step"])
                )
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(
                    exp_avg
                )

                if group["weight_decay"] != 0:
                    decayed_weights = torch.mul(p.data, group["weight_decay"])
                    p.data.add_(-step_size)
                    p.data.sub_(decayed_weights)
                else:
                    p.data.add_(-step_size)

        return loss
