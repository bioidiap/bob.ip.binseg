import torch

# TODO: REWRITE AS loss class, similary to BCELoss!
def weighted_cross_entropy_loss(preds, edges):
    """ Calculate sum of weighted cross entropy loss.
    https://github.com/xwjabc/hed/blob/master/hed.py
    """
    mask = (edges > 0.5).float()
    b, c, h, w = mask.shape
    num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
    num_neg = c * h * w - num_pos                     # Shape: [b,].
    weight = torch.zeros_like(mask)
    weight[edges > 0.5]  = num_neg / (num_pos + num_neg)
    weight[edges <= 0.5] = num_pos / (num_pos + num_neg)
    # Calculate loss.
    losses = torch.nn.functional.binary_cross_entropy(preds.float(), edges.float(), weight=weight, reduction='none')
    loss   = torch.sum(losses) / b
    return loss
