import torch
from torch.nn.modules.loss import _Loss
from torch._jit_internal import weak_script_method




class WeightedBCELogitsLoss(_Loss):
    """ 
    Implements Equation 1 in [DRIU16]_. Based on :py:class:`torch.torch.nn.modules.loss.BCEWithLogitsLoss`. 
    Calculate sum of weighted cross entropy loss.

    Attributes
    ----------
    size_average : bool, optional
        Deprecated (see :attr:`reduction`). By default, the losses are averaged over each loss element in the batch. Note that for 
        some losses, there are multiple elements per sample. If the field :attr:`size_average` is set to ``False``, the losses are 
        instead summed for each minibatch. Ignored when reduce is ``False``. Default: ``True``
    reduce : bool, optional 
        Deprecated (see :attr:`reduction`). By default, the
        losses are averaged or summed over observations for each minibatch depending
        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
        batch element instead and ignores :attr:`size_average`. Default: ``True``
    reduction : string, optional
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
        and :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    pos_weight : :py:class:`torch.Tensor`, optional
        a weight of positive examples. Must be a vector with length equal to the number of classes.
    """
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(WeightedBCELogitsLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    @weak_script_method
    def forward(self, input, target, masks=None):
        n, c, h, w = target.shape
        num_pos = torch.sum(target, dim=[1, 2, 3]).float().reshape(n,1) # torch.Size([n, 1])
        if hasattr(masks,'dtype'):
            num_mask_neg = c * h * w - torch.sum(masks, dim=[1, 2, 3]).float().reshape(n,1) # torch.Size([n, 1])
            num_neg =  c * h * w - num_pos - num_mask_neg
        else:
            num_neg = c * h * w - num_pos 
        numposnumtotal = torch.ones_like(target) * (num_pos / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2)
        numnegnumtotal = torch.ones_like(target) * (num_neg / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2)
        weight = torch.where((target <= 0.5) , numposnumtotal, numnegnumtotal)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, reduction=self.reduction)
        return loss 

class SoftJaccardBCELogitsLoss(_Loss):
    """ 
    Implements Equation 6 in [SAT17]_. Based on :py:class:`torch.torch.nn.modules.loss.BCEWithLogitsLoss`. 

    Attributes
    ----------
    alpha : float
        determines the weighting of SoftJaccard and BCE. Default: ``0.3``
    size_average : bool, optional
        Deprecated (see :attr:`reduction`). By default, the losses are averaged over each loss element in the batch. Note that for 
        some losses, there are multiple elements per sample. If the field :attr:`size_average` is set to ``False``, the losses are 
        instead summed for each minibatch. Ignored when reduce is ``False``. Default: ``True``
    reduce : bool, optional 
        Deprecated (see :attr:`reduction`). By default, the
        losses are averaged or summed over observations for each minibatch depending
        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
        batch element instead and ignores :attr:`size_average`. Default: ``True``
    reduction : string, optional
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
        and :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    pos_weight : :py:class:`torch.Tensor`, optional
        a weight of positive examples. Must be a vector with length equal to the number of classes.
    """
    def __init__(self, alpha=0.3, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(SoftJaccardBCELogitsLoss, self).__init__(size_average, reduce, reduction) 
        self.alpha = alpha   

    @weak_script_method
    def forward(self, input, target):
        eps = 1e-8
        probabilities = torch.sigmoid(input)
        intersection = (probabilities * target).sum()
        sums = probabilities.sum() + target.sum()
        
        softjaccard = intersection/(sums - intersection + eps)

        bceloss = torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=None, reduction=self.reduction)
        loss = self.alpha * bceloss + (1 - self.alpha) * (1-softjaccard)
        return loss


class HEDWeightedBCELogitsLoss(_Loss):
    """ 
    Implements Equation 2 in [HED15]_. Based on :py:class:`torch.torch.nn.modules.loss.BCEWithLogitsLoss`. 
    Calculate sum of weighted cross entropy loss.

    Attributes
    ----------
    size_average : bool, optional
        Deprecated (see :attr:`reduction`). By default, the losses are averaged over each loss element in the batch. Note that for 
        some losses, there are multiple elements per sample. If the field :attr:`size_average` is set to ``False``, the losses are 
        instead summed for each minibatch. Ignored when reduce is ``False``. Default: ``True``
    reduce : bool, optional 
        Deprecated (see :attr:`reduction`). By default, the
        losses are averaged or summed over observations for each minibatch depending
        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
        batch element instead and ignores :attr:`size_average`. Default: ``True``
    reduction : string, optional
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
        and :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    pos_weight : :py:class:`torch.Tensor`, optional
        a weight of positive examples. Must be a vector with length equal to the number of classes.
    """
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(HEDWeightedBCELogitsLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    @weak_script_method
    def forward(self, inputlist, target, masks=None):
        """[summary]
        
        Parameters
        ----------
        inputlist : list of :py:class:`torch.Tensor`
            HED uses multiple side-output feature maps for the loss calculation
        target : :py:class:`torch.Tensor`
        
        Returns
        -------
        :py:class:`torch.Tensor`
            
        """
        loss_over_all_inputs = []
        for input in inputlist:
            n, c, h, w = target.shape
            num_pos = torch.sum(target, dim=[1, 2, 3]).float().reshape(n,1) # torch.Size([n, 1])
            if hasattr(masks,'dtype'):
                num_mask_neg = c * h * w - torch.sum(masks, dim=[1, 2, 3]).float().reshape(n,1) # torch.Size([n, 1])
                num_neg =  c * h * w - num_pos - num_mask_neg
            else: 
                num_neg = c * h * w - num_pos  # torch.Size([n, 1])
            numposnumtotal = torch.ones_like(target) * (num_pos / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2)
            numnegnumtotal = torch.ones_like(target) * (num_neg / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2)
            weight = torch.where((target <= 0.5) , numposnumtotal, numnegnumtotal)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, reduction=self.reduction)
            loss_over_all_inputs.append(loss.unsqueeze(0))
        final_loss = torch.cat(loss_over_all_inputs).mean()
        return final_loss 