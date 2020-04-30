"""Loss implementations"""

import torch
from torch.nn.modules.loss import _Loss

# Conditionally decorates a method if a decorator exists in PyTorch
# This overcomes an import error with versions of PyTorch >= 1.2, where the
# decorator ``weak_script_method`` is not anymore available.  See:
# https://github.com/pytorch/pytorch/commit/10c4b98ade8349d841518d22f19a653a939e260c#diff-ee07db084d958260fd24b4b02d4f078d
# from July 4th, 2019.
try:
    from torch._jit_internal import weak_script_method
except ImportError:

    def weak_script_method(x):
        return x


class WeightedBCELogitsLoss(_Loss):
    """
    Implements Equation 1 in [MANINIS-2016]_. Based on
    :py:class:`torch.nn.BCEWithLogitsLoss`.

    Calculate sum of weighted cross entropy loss.
    """

    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        pos_weight=None,
    ):
        super(WeightedBCELogitsLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    @weak_script_method
    def forward(self, input, target, masks=None):
        """
        Parameters
        ----------
        input : :py:class:`torch.Tensor`
        target : :py:class:`torch.Tensor`
        masks : :py:class:`torch.Tensor`, optional

        Returns
        -------
        :py:class:`torch.Tensor`
        """
        n, c, h, w = target.shape
        num_pos = (
            torch.sum(target, dim=[1, 2, 3]).float().reshape(n, 1)
        )  # torch.Size([n, 1])
        if hasattr(masks, "dtype"):
            num_mask_neg = c * h * w - torch.sum(masks, dim=[1, 2, 3]).float().reshape(
                n, 1
            )  # torch.Size([n, 1])
            num_neg = c * h * w - num_pos - num_mask_neg
        else:
            num_neg = c * h * w - num_pos
        numposnumtotal = torch.ones_like(target) * (
            num_pos / (num_pos + num_neg)
        ).unsqueeze(1).unsqueeze(2)
        numnegnumtotal = torch.ones_like(target) * (
            num_neg / (num_pos + num_neg)
        ).unsqueeze(1).unsqueeze(2)
        weight = torch.where((target <= 0.5), numposnumtotal, numnegnumtotal)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input, target, weight=weight, reduction=self.reduction
        )
        return loss


class SoftJaccardBCELogitsLoss(_Loss):
    """
    Implements Equation 3 in [IGLOVIKOV-2018]_.  Based on
    ``torch.nn.BCEWithLogitsLoss``.

    Attributes
    ----------
    alpha : float
        determines the weighting of SoftJaccard and BCE. Default: ``0.7``
    """

    def __init__(
        self,
        alpha=0.7,
        size_average=None,
        reduce=None,
        reduction="mean",
        pos_weight=None,
    ):
        super(SoftJaccardBCELogitsLoss, self).__init__(size_average, reduce, reduction)
        self.alpha = alpha

    @weak_script_method
    def forward(self, input, target, masks=None):
        """
        Parameters
        ----------
        input : :py:class:`torch.Tensor`
        target : :py:class:`torch.Tensor`
        masks : :py:class:`torch.Tensor`, optional

        Returns
        -------
        :py:class:`torch.Tensor`
        """
        eps = 1e-8
        probabilities = torch.sigmoid(input)
        intersection = (probabilities * target).sum()
        sums = probabilities.sum() + target.sum()

        softjaccard = intersection / (sums - intersection + eps)

        bceloss = torch.nn.functional.binary_cross_entropy_with_logits(
            input, target, weight=None, reduction=self.reduction
        )
        loss = self.alpha * bceloss + (1 - self.alpha) * (1 - softjaccard)
        return loss


class HEDWeightedBCELogitsLoss(_Loss):
    """
    Implements Equation 2 in [HE-2015]_. Based on
    ``torch.nn.modules.loss.BCEWithLogitsLoss``.

    Calculate sum of weighted cross entropy loss.
    """

    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        pos_weight=None,
    ):
        super(HEDWeightedBCELogitsLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    @weak_script_method
    def forward(self, inputlist, target, masks=None):
        """
        Parameters
        ----------
        inputlist : list of :py:class:`torch.Tensor`
            HED uses multiple side-output feature maps for the loss calculation
        target : :py:class:`torch.Tensor`
        masks : :py:class:`torch.Tensor`, optional
        Returns
        -------
        :py:class:`torch.Tensor`
        """
        loss_over_all_inputs = []
        for input in inputlist:
            n, c, h, w = target.shape
            num_pos = (
                torch.sum(target, dim=[1, 2, 3]).float().reshape(n, 1)
            )  # torch.Size([n, 1])
            if hasattr(masks, "dtype"):
                num_mask_neg = c * h * w - torch.sum(
                    masks, dim=[1, 2, 3]
                ).float().reshape(
                    n, 1
                )  # torch.Size([n, 1])
                num_neg = c * h * w - num_pos - num_mask_neg
            else:
                num_neg = c * h * w - num_pos  # torch.Size([n, 1])
            numposnumtotal = torch.ones_like(target) * (
                num_pos / (num_pos + num_neg)
            ).unsqueeze(1).unsqueeze(2)
            numnegnumtotal = torch.ones_like(target) * (
                num_neg / (num_pos + num_neg)
            ).unsqueeze(1).unsqueeze(2)
            weight = torch.where((target <= 0.5), numposnumtotal, numnegnumtotal)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                input, target, weight=weight, reduction=self.reduction
            )
            loss_over_all_inputs.append(loss.unsqueeze(0))
        final_loss = torch.cat(loss_over_all_inputs).mean()
        return final_loss


class HEDSoftJaccardBCELogitsLoss(_Loss):
    """

    Implements  Equation 3 in [IGLOVIKOV-2018]_ for the hed network. Based on
    :py:class:`torch.nn.BCEWithLogitsLoss`.

    Attributes
    ----------
    alpha : float
        determines the weighting of SoftJaccard and BCE. Default: ``0.3``
    """

    def __init__(
        self,
        alpha=0.3,
        size_average=None,
        reduce=None,
        reduction="mean",
        pos_weight=None,
    ):
        super(HEDSoftJaccardBCELogitsLoss, self).__init__(
            size_average, reduce, reduction
        )
        self.alpha = alpha

    @weak_script_method
    def forward(self, inputlist, target, masks=None):
        """
        Parameters
        ----------
        input : :py:class:`torch.Tensor`
        target : :py:class:`torch.Tensor`
        masks : :py:class:`torch.Tensor`, optional

        Returns
        -------
        :py:class:`torch.Tensor`
        """
        eps = 1e-8
        loss_over_all_inputs = []
        for input in inputlist:
            probabilities = torch.sigmoid(input)
            intersection = (probabilities * target).sum()
            sums = probabilities.sum() + target.sum()

            softjaccard = intersection / (sums - intersection + eps)

            bceloss = torch.nn.functional.binary_cross_entropy_with_logits(
                input, target, weight=None, reduction=self.reduction
            )
            loss = self.alpha * bceloss + (1 - self.alpha) * (1 - softjaccard)
            loss_over_all_inputs.append(loss.unsqueeze(0))
        final_loss = torch.cat(loss_over_all_inputs).mean()
        return final_loss


class MixJacLoss(_Loss):
    """

    Parameters
    ----------

    lambda_u : int
        determines the weighting of SoftJaccard and BCE.

    """

    def __init__(
        self,
        lambda_u=100,
        jacalpha=0.7,
        size_average=None,
        reduce=None,
        reduction="mean",
        pos_weight=None,
    ):
        super(MixJacLoss, self).__init__(size_average, reduce, reduction)
        self.lambda_u = lambda_u
        self.labeled_loss = SoftJaccardBCELogitsLoss(alpha=jacalpha)
        self.unlabeled_loss = torch.nn.BCEWithLogitsLoss()

    @weak_script_method
    def forward(self, input, target, unlabeled_input, unlabeled_traget, ramp_up_factor):
        """
        Parameters
        ----------

        input : :py:class:`torch.Tensor`
        target : :py:class:`torch.Tensor`
        unlabeled_input : :py:class:`torch.Tensor`
        unlabeled_traget : :py:class:`torch.Tensor`
        ramp_up_factor : float

        Returns
        -------

        list

        """
        ll = self.labeled_loss(input, target)
        ul = self.unlabeled_loss(unlabeled_input, unlabeled_traget)

        loss = ll + self.lambda_u * ramp_up_factor * ul
        return loss, ll, ul
