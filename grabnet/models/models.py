
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
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from grabnet.tools.utils import rotmat2aa
from grabnet.tools.utils import CRot2rotmat
from grabnet.tools.train_tools import point2point_signed


class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=256):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout


class CoarseNet(nn.Module):
    def __init__(self,
                 n_neurons = 512,
                 latentD = 16,
                 in_bps = 4096,
                 in_pose = 12,
                 **kwargs):

        super(CoarseNet, self).__init__()

        self.latentD = latentD

        self.enc_bn0 = nn.BatchNorm1d(in_bps)
        self.enc_bn1 = nn.BatchNorm1d(in_bps + in_pose)
        self.enc_rb1 = ResBlock(in_bps + in_pose, n_neurons)
        self.enc_rb2 = ResBlock(n_neurons + in_bps + in_pose, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, latentD)
        self.enc_var = nn.Linear(n_neurons, latentD)
        self.do = nn.Dropout(p=.1, inplace=False)

        self.dec_bn1 = nn.BatchNorm1d(in_bps)  # normalize the bps_torch for object
        self.dec_rb1 = ResBlock(latentD + in_bps, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + latentD + in_bps, n_neurons)

        self.dec_pose = nn.Linear(n_neurons, 16 * 6)
        self.dec_trans = nn.Linear(n_neurons, 3)

    def encode(self, bps_object, trans_rhand, global_orient_rhand_rotmat):
        
        bs = bps_object.shape[0]

        X = torch.cat([bps_object, global_orient_rhand_rotmat.view(bs, -1), trans_rhand], dim=1)

        X0 = self.enc_bn1(X)
        X  = self.enc_rb1(X0, True)
        X  = self.enc_rb2(torch.cat([X0, X], dim=1), True)

        return torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X)))

    def decode(self, Zin, bps_object):

        bs = Zin.shape[0]
        o_bps = self.dec_bn1(bps_object)

        X0 = torch.cat([Zin, o_bps], dim=1)
        X = self.dec_rb1(X0, True)
        X = self.dec_rb2(torch.cat([X0, X], dim=1), True)

        pose = self.dec_pose(X)
        trans = self.dec_trans(X)

        results = parms_decode(pose, trans)
        results['z'] = Zin

        return results

    def forward(self, bps_object, trans_rhand, global_orient_rhand_rotmat, **kwargs):
        '''

        :param bps_object: bps_delta of object: Nxn_bpsx3
        :param delta_hand_mano: bps_delta of subject, e.g. hand: Nxn_bpsx3
        :param output_type: bps_delta of something, e.g. hand: Nxn_bpsx3
        :return:
        '''
        z = self.encode(bps_object, trans_rhand, global_orient_rhand_rotmat)
        z_s = z.rsample()

        hand_parms = self.decode(z_s, bps_object)
        results = {'mean': z.mean, 'std': z.scale}
        results.update(hand_parms)

        return results

    def sample_poses(self, bps_object, seed=None):
        bs = bps_object.shape[0]
        np.random.seed(seed)
        dtype = bps_object.dtype
        device = bps_object.device
        self.eval()
        with torch.no_grad():
            Zgen = np.random.normal(0., 1., size=(bs, self.latentD))
            Zgen = torch.tensor(Zgen,dtype=dtype).to(device)

        return self.decode(Zgen, bps_object)


class RefineNet(nn.Module):
    def __init__(self,
                 in_size=778 + 16 * 6 + 3,
                 h_size=512,
                 n_iters=3):

        super(RefineNet, self).__init__()

        self.n_iters = n_iters
        self.bn1 = nn.BatchNorm1d(778)
        self.rb1 = ResBlock(in_size,  h_size)
        self.rb2 = ResBlock(in_size + h_size, h_size)
        self.rb3 = ResBlock(in_size + h_size, h_size)
        self.out_p = nn.Linear(h_size, 16 * 6)
        self.out_t = nn.Linear(h_size, 3)
        self.dout = nn.Dropout(0.3)
        self.actvf = nn.LeakyReLU(.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, h2o_dist, fpose_rhand_rotmat_f, trans_rhand_f, global_orient_rhand_rotmat_f, verts_object, **kwargs):

        bs = h2o_dist.shape[0]
        init_pose = fpose_rhand_rotmat_f[..., :2].reshape(bs, -1)
        init_rpose = global_orient_rhand_rotmat_f[..., :2].reshape(bs, -1)
        init_pose = torch.cat([init_rpose, init_pose], dim=1)
        init_trans = trans_rhand_f

        for i in range(self.n_iters):

            if i != 0:
                hand_parms = parms_decode(init_pose, init_trans)
                verts_rhand = self.rhm_train(**hand_parms).vertices
                _, h2o_dist, _ = point2point_signed(verts_rhand, verts_object)

            h2o_dist = self.bn1(h2o_dist)
            X0 = torch.cat([h2o_dist, init_pose, init_trans], dim=1)
            X = self.rb1(X0)
            X = self.dout(X)
            X = self.rb2(torch.cat([X, X0], dim=1))
            X = self.dout(X)
            X = self.rb3(torch.cat([X, X0], dim=1))
            X = self.dout(X)

            pose = self.out_p(X)
            trans = self.out_t(X)

            init_trans = init_trans + trans
            init_pose = init_pose + pose

        hand_parms = parms_decode(init_pose, init_trans)
        return hand_parms

def parms_decode(pose,trans):

    bs = trans.shape[0]

    pose_full = CRot2rotmat(pose)
    pose = pose_full.view([bs, 1, -1, 9])
    pose = rotmat2aa(pose).view(bs, -1)

    global_orient = pose[:, :3]
    hand_pose = pose[:, 3:]
    pose_full = pose_full.view([bs, -1, 3, 3])

    hand_parms = {'global_orient': global_orient, 'hand_pose': hand_pose, 'transl': trans, 'fullpose': pose_full}

    return hand_parms