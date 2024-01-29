import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    r""" Focal loss for classification tasks.

    .. math::
        FL(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)
    Args:
        gamma (float, optional): Focusing parameter. Default: 0
        alpha (float, optional): Weighting parameter. Default: None
        size_average (bool, optional): Average the loss over the batch.
            Default: True
    """

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, long)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        return focal_loss(input, target, self.alpha, self.gamma, self.size_average)


def focal_loss(input, target, alpha, gamma, size_average, **kwargs):
    r"""
    Compute the focal loss between the input and target tensors.

    Args:
        input (Tensor): The input tensor.
        target (Tensor): The target tensor.
        alpha (Tensor): The alpha tensor for class balancing.
        gamma (float): The gamma value for focal loss.
        size_average (bool): Whether to average the loss.

    Returns:
        Tensor: The computed focal loss.

    """
    if input.dim() > 2:
        input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
    target = target.view(-1, 1)

    logpt = F.log_softmax(input)
    logpt = logpt.gather(1, target)
    logpt = logpt.view(-1)
    pt = Variable(logpt.data.exp())

    if alpha is not None:
        if alpha.type() != input.data.type():
            alpha = alpha.type_as(input.data)
        at = alpha.gather(0, target.data.view(-1))
        logpt = logpt * Variable(at)

    loss = -1 * (1 - pt)**gamma * logpt
    if size_average:
        return loss.mean()
    else:
        return loss.sum()
