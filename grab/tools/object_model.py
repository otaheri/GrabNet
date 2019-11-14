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
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2018.12.13

import numpy as np

import torch
import torch.nn as nn

# from smplx.lbs import lbs
from human_body_prior.body_model.lbs import lbs
import os


class ObjectModel(nn.Module):

    def __init__(self,
                 v_template,
                 batch_size=1,
                 dtype=torch.float32):

        super(ObjectModel, self).__init__()


        self.dtype = dtype

        # Mean template vertices
        v_template = np.repeat(v_template[np.newaxis], batch_size, axis=0)
        self.register_buffer('v_template', torch.tensor(v_template, dtype=dtype))

        trans = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('trans', nn.Parameter(trans, requires_grad=True))

        root_orient = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('root_orient', nn.Parameter(root_orient, requires_grad=True))

        self.batch_size = batch_size

    def forward(self, root_orient=None, trans=None, dmpls=None, expression=None, return_dict=False, v_template=None, **kwargs):
        '''

        :param root_orient: Nx3
        :param pose_body:
        :param pose_hand:
        :param pose_jaw:
        :param pose_eye:
        :param kwargs:
        :return:
        '''
        from human_body_prior.body_model.lbs import batch_rodrigues

        if root_orient is None:  root_orient = self.root_orient
        if trans is None: trans = self.trans
        if v_template is None: v_template = self.v_template

        rot_mats = batch_rodrigues(root_orient.view(-1, 3)).view([self.batch_size, 3, 3])

        verts = torch.matmul(v_template, rot_mats) + trans.unsqueeze(dim=1)

        res = {}
        res['v'] = verts
        res['root_orient'] = root_orient
        res['trans'] = trans


        if not return_dict:
            class result_meta(object):
                pass

            res_class = result_meta()
            for k, v in res.items():
                res_class.__setattr__(k, v)
            res = res_class

        return res

