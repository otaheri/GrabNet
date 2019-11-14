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
# 2018.05.12


import os
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
from configer import Configer
from human_body_prior.tools.omni_tools import makepath, log2file
from human_body_prior.tools.training_tools import EarlyStopping
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from grab.data.dataloader import GRAB_DS
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.train.vposer_smpl import ContinousRotReprDecoder
from human_body_prior.train.vposer_smpl import VPoser
from basis_point_sets.bps_torch import convert_to_bps_batch


class ResBlock(nn.Module):

    def __init__(self, Fin, Fout, n_neurons=256):
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

    def forward(self, x, final_nl = True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x) # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl: return self.ll(Xout)
        return Xout

class GRAB(nn.Module):
    def __init__(self, n_neurons, latentD, in_features, out_features, **kwargs):
        super(GRAB, self).__init__()

        self.latentD = latentD
        self.rot_decoder = ContinousRotReprDecoder()

        self.grab_enc_bn1 = nn.BatchNorm1d(in_features)
        # self.grab_enc_fc1 = nn.Linear(in_features, n_neurons)
        self.grab_enc_fcb1 = ResBlock(in_features, n_neurons)
        self.grab_enc_fcb2 = ResBlock(n_neurons+in_features, n_neurons)

        # self.grab_enc_bn2 = nn.BatchNorm1d(n_neurons)
        # self.grab_enc_fc2 = nn.Linear(n_neurons, n_neurons)
        self.grab_enc_mu = nn.Linear(n_neurons, latentD)
        self.grab_enc_logvar = nn.Linear(n_neurons, latentD)
        self.dropout = nn.Dropout(p=.1, inplace=False)

        # self.grab_dec_fc1 = nn.Linear(latentD+in_features//2, n_neurons)
        # self.grab_dec_fc2 = nn.Linear(n_neurons, n_neurons)

        self.grab_dec_fcb1 = ResBlock(latentD+in_features//2, n_neurons)
        self.grab_dec_fcb2 = ResBlock(n_neurons+latentD+in_features//2, n_neurons)

        self.grab_dec_out_poseH = nn.Linear(n_neurons, 16*6)
        self.grab_dec_out_trans = nn.Linear(n_neurons, 3)
        # self.grab_dec_out = nn.Linear(n_neurons, out_features)

    def encode(self, o_delta, s_delta):
        '''

        :param Pin: Nx(numjoints*3)
        :param rep_type: 'matrot'/'aa' for matrix rotations or axis-angle
        :return:
        '''
        o_bps = torch.sqrt(torch.pow(o_delta, 2).sum(2))
        s_bps = torch.sqrt(torch.pow(s_delta, 2).sum(2))

        in_bps = torch.cat([o_bps, s_bps], dim=1)

        Xbps = self.grab_enc_bn1(in_bps)
        X1 = self.grab_enc_fcb1(Xbps, True)
        X2 = self.grab_enc_fcb2(torch.cat([Xbps, X1], dim=1), True)

        # Xout = F.leaky_relu(self.grab_enc_fc1(in_bps), negative_slope=.2)
        # Xout = self.grab_enc_bn2(Xout)
        # Xout = self.dropout(Xout)
        # Xout = F.leaky_relu(self.grab_enc_fc2(Xout), negative_slope=.2)
        return torch.distributions.normal.Normal(self.grab_enc_mu(X2), F.softplus(self.grab_enc_logvar(X2)))

    def decode(self, Zin, o_delta):
        o_bps = torch.sqrt(torch.pow(o_delta, 2).sum(2))
        z_conditional = torch.cat([Zin, o_bps], dim=1)

        X1 = self.grab_dec_fcb1(z_conditional, True)
        X2 = self.grab_dec_fcb2(torch.cat([z_conditional, X1], dim=1), True)

        # Xout = F.leaky_relu(self.grab_dec_fc1(z_conditional), negative_slope=.2)
        # Xout = self.dropout(Xout)
        # Xout = F.leaky_relu(self.grab_dec_fc2(Xout), negative_slope=.2)

        poseHR = self.grab_dec_out_poseH(X2)
        poseHR = self.rot_decoder(poseHR)
        poseHR = poseHR.view([Zin.shape[0], 1, -1, 9])
        poseHR = VPoser.matrot2aa(poseHR).view(Zin.shape[0],-1)

        trans = self.grab_dec_out_trans(X2)

        return poseHR, trans

    def forward(self, o_delta, s_delta, **kwargs):
        '''

        :param o_delta: bps_delta of object: Nxn_bpsx3
        :param s_delta: bps_delta of subject, e.g. hand: Nxn_bpsx3
        :param output_type: bps_delta of something, e.g. hand: Nxn_bpsx3
        :return:
        '''
        q_z = self.encode(o_delta, s_delta)
        q_z_sample = q_z.rsample()
        poseHR, trans = self.decode(q_z_sample, o_delta)

        results = {'mean':q_z.mean, 'std':q_z.scale}
        results['poseHR'] = poseHR
        results['trans'] = trans
        return results


class GRABTrainer:

    def __init__(self, work_dir, ps):

        from tensorboardX import SummaryWriter

        self.pt_dtype = torch.float64 if ps.fp_precision == '64' else torch.float32

        torch.manual_seed(ps.seed)

        starttime = datetime.now().replace(microsecond=0)
        ps.work_dir = makepath(work_dir, isfile=False)

        logger = log2file(makepath(os.path.join(work_dir, '%s.log' % (expr_code)), isfile=True))

        summary_logdir = os.path.join(work_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        logger('[%s] - Started training grab experiment code %s' % (expr_code, starttime))
        logger('tensorboard --logdir=%s' % summary_logdir)
        logger('Torch Version: %s\n' % torch.__version__)

        logger('Base dataset_dir is %s' % ps.dataset_dir)

        shutil.copy2(os.path.basename(sys.argv[0]), work_dir)

        use_cuda = torch.cuda.is_available()
        if use_cuda: torch.cuda.empty_cache()
        self.comp_device = torch.device("cuda:%d" % ps.cuda_id if torch.cuda.is_available() else "cpu")

        gpu_brand = torch.cuda.get_device_name(ps.cuda_id) if use_cuda else None
        gpu_count = torch.cuda.device_count() if ps.use_multigpu else 1
        if use_cuda: logger('Using %d CUDA cores [%s] for training!' % (gpu_count, gpu_brand))

        kwargs = {'num_workers': ps.n_workers}
        # kwargs = {'num_workers': ps.n_workers, 'pin_memory': True} if use_cuda else {'num_workers': ps.n_workers}
        ds_train = GRAB_DS(dataset_dir=os.path.join(ps.dataset_dir, 'train'))
        self.ds_train = DataLoader(ds_train, batch_size=ps.batch_size, shuffle=True, drop_last=True, **kwargs)
        ds_val = GRAB_DS(dataset_dir=os.path.join(ps.dataset_dir, 'vald'))
        self.ds_val = DataLoader(ds_val, batch_size=ps.batch_size, shuffle=True, drop_last=True, **kwargs)
        ds_test = GRAB_DS(dataset_dir=os.path.join(ps.dataset_dir, 'test'))
        self.ds_test = DataLoader(ds_test, batch_size=ps.batch_size, shuffle=True, drop_last=False)

        logger('Dataset Train, Vald, Test size respectively: %.2f M, %.2f K, %.2f K' %
               (len(self.ds_train.dataset) * 1e-6, len(self.ds_val.dataset) * 1e-3, len(self.ds_test.dataset) * 1e-3))

        with torch.no_grad():
            self.bm = BodyModel(ps.bm_path, batch_size=ps.n_samples_to_display, use_posedirs=True).to(self.comp_device)
            self.bm_train = BodyModel(ps.bm_path, batch_size=ps.batch_size//gpu_count, use_posedirs=True).to(self.comp_device)

        self.grab_model = GRAB(n_neurons=ps.n_neurons, latentD=ps.latentD, in_features=ps.in_features, out_features=ps.out_features).to(self.comp_device)
        self.LossL1 = torch.nn.L1Loss(reduction='mean')
        self.LossL2 = torch.nn.MSELoss(reduction='mean')
        if ps.use_multigpu:
            self.grab_model = nn.DataParallel(self.grab_model)
            self.bm_train = nn.DataParallel(self.bm_train)

            logger("Training on Multiple GPU's")

        varlist = [var[1] for var in self.grab_model.named_parameters()]

        params_count = sum(p.numel() for p in varlist if p.requires_grad)
        logger('Total Trainable Parameters Count is %2.2f M.' % ((params_count) * 1e-6))

        self.optimizer = optim.Adam(varlist, lr=ps.base_lr, weight_decay=ps.reg_coef)

        self.bps_basis = np.load(os.path.join(ps.dataset_dir, 'bps_basis_1024.npz'))['basis']
        self.bps_basis_train = torch.from_numpy(np.repeat(self.bps_basis[None], repeats=ps.batch_size//gpu_count, axis=0).astype(np.float32)).to(self.comp_device)

        self.logger = logger
        self.best_loss_total = np.inf
        self.try_num = ps.try_num
        self.epochs_completed = 0
        self.ps = ps

        if ps.best_model_fname is not None:
            self._get_grabmodel().load_state_dict(torch.load(ps.best_model_fname, map_location=self.comp_device), strict=False)
            logger('Restored model from %s' % ps.best_model_fname)

        for data in self.ds_val:
            one_batch = data
            rnd_ids = np.random.choice(ps.batch_size, ps.n_samples_to_display)
            break
        self.vis_porig = {k: one_batch[k][rnd_ids].to(self.comp_device) for k in one_batch.keys()}


        # self.swriter.add_graph(self.grab_model.module, self.vis_porig, True)

    def _get_grabmodel(self):
        return self.grab_model.module if isinstance(self.grab_model, torch.nn.DataParallel) else self.grab_model

    def train(self):
        self.grab_model.train()
        save_every_it = len(self.ds_train) / self.ps.log_every_epoch
        train_loss_dict = {}
        for it, dorig in enumerate(self.ds_train):
            dorig = {k:dorig[k].to(self.comp_device) for k in dorig.keys()}

            self.optimizer.zero_grad()
            drec = self.grab_model(**dorig)

            loss_total, cur_loss_dict = self.compute_loss(dorig, drec)
            loss_total.backward()
            self.optimizer.step()

            train_loss_dict = {k: train_loss_dict.get(k, 0.0) + v.item() for k, v in cur_loss_dict.items()}
            if it % (save_every_it + 1) == 0:
                cur_train_loss_dict = {k: v / (it + 1) for k, v in train_loss_dict.items()}
                train_msg = GRABTrainer.creat_loss_message(cur_train_loss_dict, expr_code=self.ps.expr_code,
                                                              epoch_num=self.epochs_completed, it=it,
                                                              try_num=self.try_num, mode='train')

                self.logger(train_msg)

        train_loss_dict = {k: v / len(self.ds_train) for k, v in train_loss_dict.items()}
        return train_loss_dict

    def evaluate(self, split_name='vald'):
        self.grab_model.eval()
        eval_loss_dict = {}
        data = self.ds_val if split_name == 'vald' else self.ds_test
        with torch.no_grad():
            for dorig in data:
                dorig = {k: dorig[k].to(self.comp_device) for k in dorig.keys()}
                drec = self.grab_model(**dorig)
                _, cur_loss_dict = self.compute_loss(dorig, drec)

                eval_loss_dict = {k: eval_loss_dict.get(k, 0.0) + v.item() for k, v in cur_loss_dict.items()}

        eval_loss_dict = {k: v / len(data) for k, v in eval_loss_dict.items()}
        return eval_loss_dict

    def compute_loss(self, dorig, drec):

        device = dorig['s_delta'].device
        dtype = dorig['s_delta'].dtype

        q_z = torch.distributions.normal.Normal(drec['mean'], drec['std'])

        hand_mesh = self.bm_train(root_orient = drec['poseHR'][:, :3], pose_hand = drec['poseHR'][:, 3:], trans = drec['trans']).v - dorig['offset']

        # _, s_delta_rec = convert_to_bps_batch(hand_mesh, self.bps_basis_train, normalize=False, return_deltas=True)
        #hand_mesh = self.bm(pose_body=).v*MESH_SCALER

        loss_mesh_rec = (1. - self.ps.kl_coef) * self.LossL1(dorig['s_verts'], hand_mesh)
        # loss_bps_rec = 2.*(1. - self.ps.kl_coef) * self.LossL1(dorig['s_delta'], s_delta_rec)
        # loss_mesh_rec = (1. - self.ps.kl_coef) * torch.mean(torch.abs(dorig['s_delta'] - drec['s_delta_rec']))

        # KL loss
        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([self.ps.batch_size, self.ps.latentD]), requires_grad=False).to(device).type(dtype),
            scale=torch.tensor(np.ones([self.ps.batch_size, self.ps.latentD]), requires_grad=False).to(device).type(dtype))
        loss_kl = self.ps.kl_coef * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))

        loss_dict = {'loss_kl': loss_kl,
                     'loss_mesh_rec': loss_mesh_rec,
                     # 'loss_bps_rec': loss_bps_rec,
                     }

        if self.grab_model.training and self.epochs_completed < 5:
            loss_dict['loss_poseHR'] = .5*(1. - self.ps.kl_coef) * torch.mean(torch.sum(torch.pow(dorig['poseHR'] - drec['poseHR'][:, 3:], 2), dim=[-1]))
            loss_dict['loss_trans'] = .3*(1. - self.ps.kl_coef) * self.LossL2(dorig['offset'][:,0], drec['trans'])

        loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_dict['loss_total'] = loss_total

        return loss_total, loss_dict

    def perform_training(self, n_epochs=None, message=None):
        starttime = datetime.now().replace(microsecond=0)
        if n_epochs is None: n_epochs = self.ps.n_epochs

        self.logger(
            'Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))
        if message is not None: self.logger(expr_message)

        prev_lr = np.inf

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        early_stopping = EarlyStopping(patience=20)

        for epoch_num in range(1, n_epochs + 1):
            train_loss_dict = self.train()
            eval_loss_dict = self.evaluate()

            scheduler.step(eval_loss_dict['loss_total'])

            cur_lr = self.optimizer.param_groups[0]['lr']
            if cur_lr != prev_lr:
                self.logger('--- Optimizer learning rate changed from %.2e to %.2e ---' % (prev_lr, cur_lr))
                prev_lr = cur_lr
            self.epochs_completed += 1

            with torch.no_grad():
                eval_msg = GRABTrainer.creat_loss_message(eval_loss_dict, expr_code=self.ps.expr_code,
                                                              epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                              try_num=self.try_num, mode='evald')
                if eval_loss_dict['loss_total'] < self.best_loss_total:
                    self.ps.best_model_fname = makepath(
                        os.path.join(self.ps.work_dir, 'snapshots', 'TR%02d_E%03d.pt' % (
                            self.try_num, self.epochs_completed)), isfile=True)
                    self.logger(eval_msg + ' ** ')
                    self.best_loss_total = eval_loss_dict['loss_total']
                    torch.save(self.grab_model.module.state_dict() if isinstance(self.grab_model,
                                                                                     torch.nn.DataParallel) else self.grab_model.state_dict(),
                               self.ps.best_model_fname)

                    imgname = '[%s]_TR%02d_E%03d.png' % (self.ps.expr_code, self.try_num, self.epochs_completed)
                    imgpath = os.path.join(self.ps.work_dir, 'images', imgname)
                    GRABTrainer.vis_results(self.vis_porig, self.grab_model, self.bps_basis, self.bm, imgpath=imgpath)
                else:
                    self.logger(eval_msg)

                self.swriter.add_scalars('total_loss/scalars', {'train_loss_total': train_loss_dict['loss_total'],
                                                                'evald_loss_total': eval_loss_dict['loss_total'], },
                                         self.epochs_completed)

            if early_stopping(eval_loss_dict['loss_total']):
                self.logger("Early stopping")
                break

        endtime = datetime.now().replace(microsecond=0)
        self.logger(expr_message)
        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger(
            'Training done in %s! Best val total loss achieved: %.2e\n' % (endtime - starttime, self.best_loss_total))
        self.logger('Best model path: %s\n' % self.ps.best_model_fname)

    @staticmethod
    def creat_loss_message(loss_dict, expr_code='XX', epoch_num=0, it=0, try_num=0, mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s: [T:%.2e] - [%s]' % (
            expr_code, try_num, epoch_num, it, mode, loss_dict['loss_total'], ext_msg)

    @staticmethod
    def vis_results(dorig, grab_model, basis, bm, imgpath, view_angles=[0, 180]):
        from human_body_prior.mesh import MeshViewer
        from human_body_prior.tools.omni_tools import copy2cpu as c2c
        import trimesh
        from human_body_prior.tools.omni_tools import colors
        from human_body_prior.mesh.sphere import points_to_spheres
        from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_

        from human_body_prior.tools.visualization_tools import imagearray2file

        imw, imh = 400, 400

        mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
        mv.render_wireframe = True

        drec = grab_model(**dorig)
        from basis_point_sets.bps import reconstruct_from_bps

        images = np.zeros([len(view_angles) + 1, len(dorig['o_delta']), 1, imw, imh, 3])

        o_orig_pc = reconstruct_from_bps(c2c(dorig['o_delta']), basis)
        s_orig_pc = reconstruct_from_bps(c2c(dorig['s_delta']), basis)

        # s_rec_pc = reconstruct_from_bps(c2c(drec['s_delta_rec']), basis)
        hand_mesh = bm(root_orient = drec['poseHR'][:, :3], pose_hand = drec['poseHR'][:, 3:], trans = drec['trans']).v - dorig['offset']

        for cId in range(0, len(dorig['o_delta'])):

            o_orig_mesh = points_to_spheres(o_orig_pc[cId], radius=0.01, vc=colors['red'])
            s_orig_mesh = points_to_spheres(s_orig_pc[cId], radius=0.01, vc=colors['blue'])

            s_rec_mesh = trimesh.Trimesh(vertices=c2c(hand_mesh[cId]), faces=c2c(bm.f), vertex_colors=np.tile(colors['blue'], (6890, 1)))

            all_meshes = [s_rec_mesh] + o_orig_mesh

            # apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(-90), (1, 0, 0)))
            #
            mv.set_meshes(o_orig_mesh+s_orig_mesh, group_name='static')
            images[0, cId, 0] = mv.render()[:,:,:3]

            for rId, angle in enumerate(view_angles):
                if angle != 0: apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(angle), (0, 1, 0)))
                mv.set_meshes(all_meshes, group_name='static')
                images[rId+1, cId, 0] = mv.render()[:,:,:3]
                if angle != 0: apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(-angle), (0, 1, 0)))

        imagearray2file(images, imgpath)


if __name__ == '__main__':

    expr_code = 'V03_06_01'

    default_ps_fname = 'grab_defaults.ini'

    base_dir = '/ps/scratch/body_hand_object_contact/grab_net/experiments'

    work_dir = os.path.join(base_dir, expr_code)

    params = {
        'n_neurons': 256,
        'batch_size': 256,
        'n_workers': 10,
        'cuda_id': 1,

        'use_multigpu':False,
        'latentD': 128,
        'kl_coef': 1e-3,

        'in_features': 2048,
        'out_features':1024*3,
        'bm_path': '/ps/scratch/body_hand_object_contact/body_models/models/models_unchumpy/MANO_RIGHT.npz',

        'reg_coef': 5e-4,

        'base_lr': 5e-4,

        'best_model_fname': None, # trained without betas before
        'log_every_epoch': 2,
        'expr_code': expr_code,
        'work_dir': work_dir,
        'n_epochs': 10000,
        'dataset_dir' : '/ps/scratch/body_hand_object_contact/grab_net/data/V01_04_00',
    }

    grab_trainer = GRABTrainer(work_dir, ps=Configer(default_ps_fname=default_ps_fname, **params))
    ps = grab_trainer.ps

    ps.dump_settings(os.path.join(work_dir, 'TR%02d_%s' % (ps.try_num, os.path.basename(default_ps_fname))))

    expr_message = '\n[%s] %d H neurons, batch_size=%d, BPS_NUM=512\n'% (ps.expr_code, ps.n_neurons, ps.batch_size)
    expr_message += 'Given concatenated BPS representation of the hand and the object predict hand delta\n'
    expr_message += 'Conditional VAE.\n'
    expr_message += 'Regressing HAND Model parameters\n'
    expr_message += 'Using 2 times data augmentation\n'
    expr_message += 'Removing BatchNorm\n'
    expr_message += 'Reconstruction loss on handpose, the resulting mesh, and trans to keep it close to offset.\n'
    expr_message += 'More sophisticated network\n'

    grab_trainer.logger(expr_message)
    grab_trainer.perform_training()
    ps.dump_settings(os.path.join(work_dir, 'TR%02d_%s' % (ps.try_num, os.path.basename(default_ps_fname))))