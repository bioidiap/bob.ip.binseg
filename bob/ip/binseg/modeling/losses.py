import torch
from torch.nn.modules.loss import _Loss
from torch._jit_internal import weak_script_method




class WeightedBCELogitsLoss(_Loss):
    """ 
    Implements Equation 1 in `Maninis et al. (2016)`_. Based on ``torch.nn.modules.loss.BCEWithLogitsLoss``. 
    Calculate sum of weighted cross entropy loss.
    """
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(WeightedBCELogitsLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

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
    Implements Equation 3 in `Iglovikov  et al. (2018)`_. Based on ``torch.nn.modules.loss.BCEWithLogitsLoss``. 

    Attributes
    ----------
    alpha : float
        determines the weighting of SoftJaccard and BCE. Default: ``0.7``
    """
    def __init__(self, alpha=0.7, size_average=None, reduce=None, reduction='mean', pos_weight=None):
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
        
        softjaccard = intersection/(sums - intersection + eps)

        bceloss = torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=None, reduction=self.reduction)
        loss = self.alpha * bceloss + (1 - self.alpha) * (1-softjaccard)
        return loss


class HEDWeightedBCELogitsLoss(_Loss):
    """ 
    Implements Equation 2 in `He et al. (2015)`_. Based on ``torch.nn.modules.loss.BCEWithLogitsLoss``. 
    Calculate sum of weighted cross entropy loss.
    """
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(HEDWeightedBCELogitsLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

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


class HEDSoftJaccardBCELogitsLoss(_Loss):
    """ 
    Implements  Equation 3 in `Iglovikov  et al. (2018)`_ for the hed network. Based on ``torch.nn.modules.loss.BCEWithLogitsLoss``. 

    Attributes
    ----------
    alpha : float
        determines the weighting of SoftJaccard and BCE. Default: ``0.3``
    """
    def __init__(self, alpha=0.3, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(HEDSoftJaccardBCELogitsLoss, self).__init__(size_average, reduce, reduction) 
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
            
            softjaccard = intersection/(sums - intersection + eps)
    
            bceloss = torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=None, reduction=self.reduction)
            loss = self.alpha * bceloss + (1 - self.alpha) * (1-softjaccard)
            loss_over_all_inputs.append(loss.unsqueeze(0))
        final_loss = torch.cat(loss_over_all_inputs).mean()
        return loss



class MixJacLoss(_Loss):
    """ 
    Attributes
    ----------
    lambda_u : int
        determines the weighting of SoftJaccard and BCE.
    """
    def __init__(self, lambda_u=100, jacalpha=0.7, size_average=None, reduce=None, reduction='mean', pos_weight=None):
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
        ll = self.labeled_loss(input,target)
        ul = self.unlabeled_loss(unlabeled_input, unlabeled_traget)
        
        loss = ll + self.lambda_u * ramp_up_factor * ul
        return loss, ll, ul