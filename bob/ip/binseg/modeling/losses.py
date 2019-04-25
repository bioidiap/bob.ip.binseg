import torch
from torch.nn.modules.loss import _Loss
from torch._jit_internal import weak_script_method

class WeightedBCELogitsLoss(_Loss):
    """ 
    Calculate sum of weighted cross entropy loss. Use for binary classification.
    """
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(WeightedBCELogitsLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    @weak_script_method
    def forward(self, input, target):
        n, c, h, w = target.shape
        num_pos = torch.sum(target, dim=[1, 2, 3]).float().reshape(n,1) # torch.Size([n, 1])
        num_neg = c * h * w - num_pos  # torch.Size([n, 1])
        numposnumtotal = torch.ones_like(target) * (num_pos / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2)
        numnegnumtotal = torch.ones_like(target) * (num_neg / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2)
        weight = torch.where((target <= 0.5) , numposnumtotal, numnegnumtotal)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, reduction=self.reduction)
        return loss 

class HEDWeightedBCELogitsLoss(_Loss):
    """ 
    Calculate sum of weighted cross entropy loss. Use for binary classification.
    """
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(HEDWeightedBCELogitsLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    @weak_script_method
    def forward(self, inputlist, target):
        loss_over_all_inputs = []
        for input in inputlist:
            n, c, h, w = target.shape
            num_pos = torch.sum(target, dim=[1, 2, 3]).float().reshape(n,1) # torch.Size([n, 1])
            num_neg = c * h * w - num_pos  # torch.Size([n, 1])
            numposnumtotal = torch.ones_like(target) * (num_pos / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2)
            numnegnumtotal = torch.ones_like(target) * (num_neg / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2)
            weight = torch.where((target <= 0.5) , numposnumtotal, numnegnumtotal)

            loss = torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, reduction=self.reduction)
            loss_over_all_inputs.append(loss.unsqueeze(0))
        final_loss = torch.cat(loss_over_all_inputs).mean()
        return final_loss 