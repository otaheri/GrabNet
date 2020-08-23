
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


import numpy as np
import torch
import torchgeometry as tgm
import logging

import torch.nn.functional as F
from copy import copy
from collections import namedtuple


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todense(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array.astype(dtype)


def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path

def makelogger(log_dir,mode='w'):


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)

    logger.addHandler(ch)

    fh = logging.FileHandler('%s'%log_dir, mode=mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        v = DotDict(v)
                    if isinstance(v, list):
                        self.__convert(v)
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    v = DotDict(v)
                elif isinstance(v, list):
                    self.__convert(v)
                self[k] = v

    def __convert(self, v):
        for elem in range(0, len(v)):
            if isinstance(v[elem], dict):
                v[elem] = DotDict(v[elem])
            elif isinstance(v[elem], list):
                self.__convert(v[elem])

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

def convert_dict2ntuple(in_dict):
    out_dict = copy(in_dict)
    for k,v in out_dict.items():
       if isinstance(v,dict):
           out_dict[k] = convert_dict2ntuple(v)
    return namedtuple('GDict', out_dict.keys())(**out_dict)

def CRot2rotmat(pose):

    reshaped_input = pose.view(-1, 3, 2)

    b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

    dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
    b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=1)

    return torch.stack([b1, b2, b3], dim=-1)

def rotmat2aa(rotmat):
    '''
    :param rotmat: Nx1xnum_jointsx9
    :return: Nx1xnum_jointsx3
    '''
    batch_size = rotmat.size(0)
    homogen_matrot = F.pad(rotmat.view(-1, 3, 3), [0,1])
    pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
    return pose

def aa2rotmat(axis_angle):
    '''
    :param Nx1xnum_jointsx3
    :return: pose_matrot: Nx1xnum_jointsx9
    '''
    batch_size = axis_angle.size(0)
    pose_body_matrot = tgm.angle_axis_to_rotation_matrix(axis_angle.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, 1, -1, 9)
    return pose_body_matrot


def euler(rots, order='xyz', units='deg'):

    rots = np.asarray(rots)
    single_val = False if len(rots.shape)>1 else True
    rots = rots.reshape(-1,3)
    rotmats = []

    for xyz in rots:
        if units == 'deg':
            xyz = np.radians(xyz)
        r = np.eye(3)
        for theta, axis in zip(xyz,order):
            c = np.cos(theta)
            s = np.sin(theta)
            if axis=='x':
                r = np.dot(np.array([[1,0,0],[0,c,-s],[0,s,c]]), r)
            if axis=='y':
                r = np.dot(np.array([[c,0,s],[0,1,0],[-s,0,c]]), r)
            if axis=='z':
                r = np.dot(np.array([[c,-s,0],[s,c,0],[0,0,1]]), r)
        rotmats.append(r)
    rotmats = np.stack(rotmats).astype(np.float32)
    if single_val:
        return rotmats[0]
    else:
        return rotmats

def batch_euler(bxyz,order='xyz', units='deg'):

    br = []
    for frame in range(bxyz.shape[0]):
        br.append(euler(bxyz[frame], order, units))
    return np.stack(br).astype(np.float32)

def rotate(points,R):
    shape = points.shape
    if len(shape)>3:
        points = points.squeeze()
    if len(shape)<3:
        points = points[:,np.newaxis]
    r_points = torch.matmul(torch.from_numpy(points).to(device), torch.from_numpy(R).to(device).transpose(1,2))
    return r_points.cpu().numpy().reshape(shape)

def rotmul(rotmat,R):

    shape = rotmat.shape
    rotmat = rotmat.squeeze()
    R = R.squeeze()
    rot = torch.matmul(torch.from_numpy(R).to(device),torch.from_numpy(rotmat).to(device))
    return rot.cpu().numpy().reshape(shape)