
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#


import torch
import numpy as np
import chamfer_distance as chd

def point2point_signed(
        x,
        y,
        x_normals=None,
        y_normals=None,
):
    """
    signed distance between two pointclouds

    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).

    Returns:

        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - yidx_near: Torch.tensor
            the indices of x vertices closest to y

    """


    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    ch_dist = chd.ChamferDistance()

    x_near, y_near, xidx_near, yidx_near = ch_dist(x,y)

    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near
    y2x = y - y_near

    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded)
        in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out

    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    return y2x_signed, x2y_signed, yidx_near



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.trace_func is not None:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop
