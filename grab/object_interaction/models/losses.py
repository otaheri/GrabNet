import torch
import torch.nn as tnn
import numpy as np
from IPython.core.debugger import set_trace

# only count the loss where target is non-zero
class TextureLoss(tnn.Module):
  def __init__(self, pos_weight=10):
    super(TextureLoss, self).__init__()
    self.loss = tnn.CrossEntropyLoss(weight=torch.Tensor([1, pos_weight]),
        ignore_index=2, reduction='none')

  def forward(self, preds, targs):
    loss = self.loss(preds, targs)
    loss = torch.mean(loss, 1)
    return loss


def classification_error(preds, targs):
  _, pred_class = torch.max(preds, dim=1)
  errors = []
  for pred, targ in zip(pred_class, targs):
    mask = targ != 2
    masked_pred = pred.masked_select(mask)
    masked_targ = targ.masked_select(mask)
    if torch.sum(masked_pred) == 0:  # ignore degenerate masks
      error = torch.tensor(1000).to(device=preds.device, dtype=preds.dtype)
    else:
      error = masked_pred != masked_targ
      error = torch.mean(error.to(dtype=preds.dtype))
    errors.append(error)
  errors = torch.stack(errors)
  return errors


class DiverseLoss(tnn.Module):
  def __init__(self, pos_weight=10, beta=1, train=True, eval_mode=False):
    """
    :param eval_mode: returns match indices, and computes loss as L2 loss
    """
    super(DiverseLoss, self).__init__()
    self.loss = classification_error if eval_mode \
        else TextureLoss(pos_weight=pos_weight)
    self.beta = beta
    self.train = train
    if eval_mode:
      self.train = False

  def forward(self, preds, targs):
    """
    :param preds: N x Ep x 2 x P
    :param targs: N x Et x P
    :return:
    """
    preds = preds.view(preds.shape[:3],-1)
    targs = targs.view(targs.shape[:2],-1)
    N, Ep, _, P = preds.shape
    _, Et, _ = targs.shape

    losses = []
    match_indices = []
    for pred, targ in zip(preds, targs):
      pred = pred.repeat(Et, 1, 1)
      targ = targ.repeat(1, Ep).view(-1, P)
      loss_matrix = self.loss(pred, targ).view(Et, Ep)
      loss, match_idx = loss_matrix.min(1)
      loss = loss.mean()
      if self.train:
        l_catchup = loss_matrix.min(0)[0].max() / Ep
        loss = loss + self.beta * l_catchup
      losses.append(loss)
      match_indices.append(match_idx)
    loss = torch.stack(losses).mean()
    match_indices = torch.stack(match_indices)
    
    return loss, match_indices
