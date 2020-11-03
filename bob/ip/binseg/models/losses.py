"""Loss implementations"""

import torch
from torch.nn.modules.loss import _Loss


class WeightedBCELogitsLoss(_Loss):
    """Calculates sum of weighted cross entropy loss.

    Implements Equation 1 in [MANINIS-2016]_.  The weight depends on the
    current proportion between negatives and positives in the ground-truth
    sample being analyzed.
    """

    def __init__(self):
        super(WeightedBCELogitsLoss, self).__init__()

    def forward(self, input, target, mask):
        """

        Parameters
        ----------

        input : :py:class:`torch.Tensor`
            Value produced by the model to be evaluated, with the shape ``[n, c,
            h, w]``

        target : :py:class:`torch.Tensor`
            Ground-truth information with the shape ``[n, c, h, w]``

        mask : :py:class:`torch.Tensor`
            Mask to be use for specifying the region of interest where to
            compute the loss, with the shape ``[n, c, h, w]``


        Returns
        -------

        loss : :py:class:`torch.Tensor`
            The average loss for all input data

        """

        # calculates the proportion of negatives to the total number of pixels
        # available in the masked region
        valid = mask > 0.5
        num_pos = target[valid].sum()
        num_neg = valid.sum() - num_pos
        pos_weight = num_neg / num_pos

        return torch.nn.functional.binary_cross_entropy_with_logits(
            input[valid], target[valid], reduction="mean", pos_weight=pos_weight
        )


class SoftJaccardBCELogitsLoss(_Loss):
    """
    Implements the generalized loss function of Equation (3) in
    [IGLOVIKOV-2018]_, with J being the Jaccard distance, and H, the Binary
    Cross-Entropy Loss:

    .. math::

       L = \alpha H + (1-\alpha)(1-J)


    Our implementation is based on :py:class:`torch.nn.BCEWithLogitsLoss`.


    Attributes
    ----------

    alpha : float
        determines the weighting of J and H. Default: ``0.7``

    """

    def __init__(self, alpha=0.7):
        super(SoftJaccardBCELogitsLoss, self).__init__()
        self.alpha = alpha

    def forward(self, input, target, mask):
        """

        Parameters
        ----------

        input : :py:class:`torch.Tensor`
            Value produced by the model to be evaluated, with the shape ``[n, c,
            h, w]``

        target : :py:class:`torch.Tensor`
            Ground-truth information with the shape ``[n, c, h, w]``

        mask : :py:class:`torch.Tensor`
            Mask to be use for specifying the region of interest where to
            compute the loss, with the shape ``[n, c, h, w]``


        Returns
        -------

        loss : :py:class:`torch.Tensor`
            Loss, in a single entry

        """

        eps = 1e-8
        valid = mask > 0.5
        probabilities = torch.sigmoid(input[valid])
        intersection = (probabilities * target[valid]).sum()
        sums = probabilities.sum() + target[valid].sum()
        J = intersection / (sums - intersection + eps)

        # this implements the support for looking just into the RoI
        H = torch.nn.functional.binary_cross_entropy_with_logits(
            input[valid], target[valid], reduction="mean"
        )
        return (self.alpha * H) + ((1 - self.alpha) * (1 - J))


class MultiWeightedBCELogitsLoss(WeightedBCELogitsLoss):
    """
    Weighted Binary Cross-Entropy Loss for multi-layered inputs (e.g. for
    Holistically-Nested Edge Detection in [XIE-2015]_).
    """

    def __init__(self):
        super(MultiWeightedBCELogitsLoss, self).__init__()

    def forward(self, input, target, mask):
        """
        Parameters
        ----------

        input : iterable over :py:class:`torch.Tensor`
            Value produced by the model to be evaluated, with the shape ``[L,
            n, c, h, w]``

        target : :py:class:`torch.Tensor`
            Ground-truth information with the shape ``[n, c, h, w]``

        mask : :py:class:`torch.Tensor`
            Mask to be use for specifying the region of interest where to
            compute the loss, with the shape ``[n, c, h, w]``


        Returns
        -------

        loss : torch.Tensor
            The average loss for all input data

        """

        return torch.cat(
            [
                super(MultiWeightedBCELogitsLoss, self).forward(i, target,
                    mask).unsqueeze(0)
                for i in input
            ]
        ).mean()


class MultiSoftJaccardBCELogitsLoss(SoftJaccardBCELogitsLoss):
    """

    Implements  Equation 3 in [IGLOVIKOV-2018]_ for the multi-output networks
    such as HED or Little W-Net.


    Attributes
    ----------

    alpha : float
        determines the weighting of SoftJaccard and BCE. Default: ``0.3``

    """

    def __init__(self, alpha=0.7):
        super(MultiSoftJaccardBCELogitsLoss, self).__init__(alpha=alpha)

    def forward(self, inputlist, target):
        """
        Parameters
        ----------

        input : iterable over :py:class:`torch.Tensor`
            Value produced by the model to be evaluated, with the shape ``[L,
            n, c, h, w]``

        target : :py:class:`torch.Tensor`
            Ground-truth information with the shape ``[n, c, h, w]``

        mask : :py:class:`torch.Tensor`
            Mask to be use for specifying the region of interest where to
            compute the loss, with the shape ``[n, c, h, w]``


        Returns
        -------

        loss : torch.Tensor
            The average loss for all input data

        """

        return torch.cat(
            [
                super(MultiSoftJaccardBCELogitsLoss, self).forward(
                    i, target, mask
                ).unsqueeze(0)
                for i in input
            ]
        ).mean()


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

    def forward(
        self, input, target, unlabeled_input, unlabeled_target, ramp_up_factor
    ):
        """
        Parameters
        ----------

        input : :py:class:`torch.Tensor`
        target : :py:class:`torch.Tensor`
        unlabeled_input : :py:class:`torch.Tensor`
        unlabeled_target : :py:class:`torch.Tensor`
        ramp_up_factor : float

        Returns
        -------

        list

        """
        ll = self.labeled_loss(input, target)
        ul = self.unlabeled_loss(unlabeled_input, unlabeled_target)

        loss = ll + self.lambda_u * ramp_up_factor * ul
        return loss, ll, ul
