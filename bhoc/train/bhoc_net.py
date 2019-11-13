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
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import makepath, log2file
from human_body_prior.tools.training_tools import EarlyStopping
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from supercap.data.dataloader import SuperCapDS
from supercap.data.prepare_data_I import SC_Synth
from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.train.vposer_smpl import VPoser, ContinousRotReprDecoder

import supercap.data.markers as markersets

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

class SuperCap(nn.Module):
    NJoints = 52

    def __init__(self, n_neurons, n_bps, **kwargs):
        super(SuperCap, self).__init__()

        self.mrk_vids = list(markersets.MPI_Mosh.values())

        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.ll = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

        self.sc_bn1 = nn.BatchNorm1d(n_bps)
        self.sc_fcb1 = ResBlock(n_bps, n_neurons)
        self.sc_fcb2 = ResBlock(n_neurons+n_bps, n_neurons)
        self.sc_fcb3 = ResBlock((2*n_neurons)+n_bps, n_neurons)
        self.sc_fc_betas = nn.Linear(n_neurons, n_neurons)

        self.rot_decoder = ContinousRotReprDecoder()

        self.sc_out_pose = nn.Linear(n_neurons, 6*22)
        self.sc_out_betas = nn.Linear(n_neurons, SC_Synth.BM_NUM_BETAS)

    def forward(self, bps):
        '''

        :param bps: N x n_bps points
        :param bps: N x 3 permuted markers
        :return:
        '''

        Xbps = self.sc_bn1(bps)
        X1 = self.sc_fcb1(Xbps, True)
        X2 = self.sc_fcb2(torch.cat([Xbps, X1], dim=1), True)
        X3 = self.sc_fcb3(torch.cat([Xbps, X1, X2], dim=1), True)
        X_betas = self.ll(self.sc_fc_betas(X3))

        out_betas = self.sc_out_betas(X_betas)

        out_pose = self.sc_out_pose(X3)
        out_pose = self.rot_decoder(out_pose)
        out_pose = VPoser.matrot2aa(out_pose).view(Xbps.shape[0], -1)

        result = {'pose_body': out_pose[:,3:], 'root_orient':out_pose[:,:3], 'betas': out_betas}
        return result


class SuperCapTrainer:

    def __init__(self, work_dir, ps):

        from tensorboardX import SummaryWriter

        self.pt_dtype = torch.float64 if ps.fp_precision == '64' else torch.float32

        torch.manual_seed(ps.seed)

        starttime = datetime.now().replace(microsecond=0)
        ps.work_dir = makepath(work_dir, isfile=False)

        logger = log2file(makepath(os.path.join(work_dir, '%s.log' % (expr_code)), isfile=True))

        summary_logdir = os.path.join(work_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        logger('[%s] - Started training supercap experiment code %s' % (expr_code, starttime))
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
        ds_train = SuperCapDS(dataset_dir=os.path.join(ps.dataset_dir, 'train'))
        self.ds_train = DataLoader(ds_train, batch_size=ps.batch_size, shuffle=True, drop_last=True, **kwargs)
        ds_val = SuperCapDS(dataset_dir=os.path.join(ps.dataset_dir, 'vald'))
        self.ds_val = DataLoader(ds_val, batch_size=ps.batch_size, shuffle=True, drop_last=True, **kwargs)
        ds_test = SuperCapDS(dataset_dir=os.path.join(ps.dataset_dir, 'test'))
        self.ds_test = DataLoader(ds_test, batch_size=ps.batch_size, shuffle=True, drop_last=False)

        logger('Dataset Train, Vald, Test size respectively: %.2f M, %.2f K, %.2f K' %
               (len(self.ds_train.dataset) * 1e-6, len(self.ds_val.dataset) * 1e-3, len(self.ds_test.dataset) * 1e-3))

        with torch.no_grad():
            self.bm = BodyModel(ps.bm_path, batch_size=1, num_betas=SC_Synth.BM_NUM_BETAS, use_posedirs=False).to(self.comp_device)
            self.bm_train = BodyModel(ps.bm_path, batch_size=ps.batch_size//gpu_count, num_betas=SC_Synth.BM_NUM_BETAS, use_posedirs=False).to(self.comp_device)

        self.sc_model = SuperCap(n_neurons=ps.n_neurons, n_bps=ps.n_bps).to(self.comp_device)
        self.LossL1 = torch.nn.L1Loss(reduction='mean')
        self.LossL2 = torch.nn.MSELoss(reduction='mean')
        if ps.use_multigpu:
            self.sc_model = nn.DataParallel(self.sc_model)
            self.bm_train = nn.DataParallel(self.bm_train)

            logger("Training on Multiple GPU's")

        varlist = [var[1] for var in self.sc_model.named_parameters()]

        params_count = sum(p.numel() for p in varlist if p.requires_grad)
        logger('Total Trainable Parameters Count is %2.2f M.' % ((params_count) * 1e-6))

        self.optimizer = optim.Adam(varlist, lr=ps.base_lr, weight_decay=ps.reg_coef)

        self.logger = logger
        self.best_loss_total = np.inf
        self.try_num = ps.try_num
        self.epochs_completed = 0
        self.ps = ps

        if ps.best_model_fname is not None:
            self._get_scmodel().load_state_dict(torch.load(ps.best_model_fname, map_location=self.comp_device), strict=False)
            logger('Restored model from %s' % ps.best_model_fname)

        for data in self.ds_val:
            one_batch = data
            rnd_ids = np.random.choice(ps.batch_size, ps.num_bodies_to_display)
            break
        self.vis_porig = {k: one_batch[k][rnd_ids].to(self.comp_device) for k in one_batch.keys()}


        # self.swriter.add_graph(self.sc_model.module, self.vis_porig, True)

    def _get_scmodel(self):
        return self.sc_model.module if isinstance(self.sc_model, torch.nn.DataParallel) else self.sc_model

    def train(self):
        self.sc_model.train()
        save_every_it = len(self.ds_train) / self.ps.log_every_epoch
        train_loss_dict = {}
        for it, dorig in enumerate(self.ds_train):
            dorig = {k:dorig[k].to(self.comp_device) for k in dorig.keys()}

            self.optimizer.zero_grad()
            drec = self.sc_model(dorig['markers_bps'])

            loss_total, cur_loss_dict = self.compute_loss(dorig, drec)
            loss_total.backward()
            self.optimizer.step()

            train_loss_dict = {k: train_loss_dict.get(k, 0.0) + v.item() for k, v in cur_loss_dict.items()}
            if it % (save_every_it + 1) == 0:
                cur_train_loss_dict = {k: v / (it + 1) for k, v in train_loss_dict.items()}
                train_msg = SuperCapTrainer.creat_loss_message(cur_train_loss_dict, expr_code=self.ps.expr_code,
                                                              epoch_num=self.epochs_completed, it=it,
                                                              try_num=self.try_num, mode='train')

                self.logger(train_msg)

        train_loss_dict = {k: v / len(self.ds_train) for k, v in train_loss_dict.items()}
        return train_loss_dict

    def evaluate(self, split_name='vald'):
        self.sc_model.eval()
        eval_loss_dict = {}
        data = self.ds_val if split_name == 'vald' else self.ds_test
        with torch.no_grad():
            for dorig in data:
                dorig = {k: dorig[k].to(self.comp_device) for k in dorig.keys()}
                drec = self.sc_model(dorig['markers_bps'])
                _, cur_loss_dict = self.compute_loss(dorig, drec)

                eval_loss_dict = {k: eval_loss_dict.get(k, 0.0) + v.item() for k, v in cur_loss_dict.items()}

        eval_loss_dict = {k: v / len(data) for k, v in eval_loss_dict.items()}
        return eval_loss_dict

    def compute_loss(self, dorig, drec):
        '''
        given two data dictionaries, compute L2 distance on values represented by keys in drec
        :param dorig: original data terms
        :param drec: reconstructed data
        :return:
        '''

        MESH_SCALER = 1e3

        if 'betas' not in list(drec.keys()): drec['betas'] = dorig['betas']

        body_rec = self.bm_train(root_orient=drec['root_orient'], pose_body=drec['pose_body'], betas=drec['betas'], return_dict=True)['v']
        body_orig = self.bm_train(root_orient=dorig['root_orient'], pose_body=dorig['pose_body'], betas=dorig['betas'], return_dict=True)['v']

        loss_dict = {
            'mesh_rec': self.LossL1(body_rec, body_orig) * MESH_SCALER,
            # 'betas': torch.pow(drec['betas'] - dorig['betas'], 2).mean(),
            }

        if self.sc_model.training and self.epochs_completed < 10:
            loss_dict['loss_root_orient'] = self.LossL2(dorig['root_orient'], drec['root_orient'])
            loss_dict['loss_pose_body'] = self.LossL2(dorig['pose_body'], drec['pose_body'])

        loss_dict['loss_total'] = torch.stack(list(loss_dict.values())).sum()

        return loss_dict['loss_total'], loss_dict

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
                eval_msg = SuperCapTrainer.creat_loss_message(eval_loss_dict, expr_code=self.ps.expr_code,
                                                              epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                              try_num=self.try_num, mode='evald')
                if eval_loss_dict['loss_total'] < self.best_loss_total:
                    self.ps.best_model_fname = makepath(
                        os.path.join(self.ps.work_dir, 'snapshots', 'TR%02d_E%03d.pt' % (
                            self.try_num, self.epochs_completed)), isfile=True)
                    self.logger(eval_msg + ' ** ')
                    self.best_loss_total = eval_loss_dict['loss_total']
                    torch.save(self.sc_model.module.state_dict() if isinstance(self.sc_model,
                                                                                     torch.nn.DataParallel) else self.sc_model.state_dict(),
                               self.ps.best_model_fname)

                    imgname = '[%s]_TR%02d_E%03d.png' % (self.ps.expr_code, self.try_num, self.epochs_completed)
                    imgpath = os.path.join(self.ps.work_dir, 'images', imgname)
                    # SuperCapTrainer.vis_results(self.vis_porig, bm=self.bm, sc_model=self.sc_model, vposer=self.vposer, imgpath=imgpath)
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
    def vis_results(dorig, bm, sc_model, imgpath, vposer=None, view_angles=[0, 180]):
        from human_body_prior.mesh import MeshViewer
        from human_body_prior.tools.omni_tools import copy2cpu as c2c
        import trimesh
        from human_body_prior.tools.omni_tools import colors
        from human_body_prior.mesh.sphere import points_to_spheres
        from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_

        from human_body_prior.tools.visualization_tools import imagearray2file

        imw, imh = 800, 800

        mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
        mv.render_wireframe = True


        drec = sc_model(dorig['markers_bps'])

        images = np.zeros([len(view_angles) + 1, len(dorig['pose_body']), 1, imw, imh, 3])
        faces = c2c(bm.f)

        for cId in range(0, len(dorig['pose_body'])):

            orig_bodyB = bm(betas=dorig['betas'][cId:cId + 1])
            orig_bodyB_mesh = trimesh.Trimesh(vertices=c2c(orig_bodyB.v[0]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))

            orig_body = bm(root_orient= dorig['root_orient'][cId:cId+1],
                         pose_body=dorig['pose_body'][cId:cId+1],
                         betas=dorig['betas'][cId:cId+1])

            orig_body_mesh = trimesh.Trimesh(vertices=c2c(orig_body.v[0]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))

            pose_body = drec['pose_body'][cId:cId+1] if 'pose_body' in list(drec.keys()) else vposer.decode(drec['poZ_body'][cId:cId+1], output_type='aa').view(1, -1)
            betas = drec['betas'][cId:cId+1] if 'betas' in list(drec.keys()) else dorig['betas'][cId:cId+1]

            rec_bodyB = bm(betas=betas)
            rec_bodyB_mesh = trimesh.Trimesh(vertices=c2c(rec_bodyB.v[0]), faces=faces, vertex_colors=np.tile(colors['red'], (6890, 1)))

            rec_body = bm(root_orient=drec['root_orient'][cId:cId+1],
                           pose_body=pose_body,
                           betas=betas,
                           )# body gender is assumed to be given. trans is for visualization only
            rec_body_mesh = trimesh.Trimesh(vertices=c2c(rec_body.v[0]), faces=faces, vertex_colors=np.tile(colors['red'], (6890, 1)))



            all_meshes = [orig_body_mesh, rec_body_mesh]
            all_meshesB = [orig_bodyB_mesh, rec_bodyB_mesh]

            apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(-90), (1, 0, 0)))
            # apply_mesh_tranfsormations_(all_meshesB, trimesh.transformations.rotation_matrix(np.radians(-90), (1, 0, 0)))
            mv.set_meshes(all_meshesB, group_name='static')
            images[0, cId, 0] = mv.render()

            for rId, angle in enumerate(view_angles):
                if angle != 0: apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(angle), (0, 1, 0)))
                mv.set_meshes(all_meshes, group_name='static')
                images[rId+1, cId, 0] = mv.render()
                if angle != 0: apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(-angle), (0, 1, 0)))

        imagearray2file(images, imgpath)


if __name__ == '__main__':
    from supercap.data.prepare_data_I import gdr2num

    expr_code = 'V01_01_00'

    default_ps_fname = 'bhoc_defaults.ini'

    base_dir = '/ps/scratch/body_hand_object_contact/bhoc_network/experiments'

    work_dir = os.path.join(base_dir, expr_code)

    params = {
        'n_neurons': 512,
        'batch_size': 512,
        'n_workers': 10,
        'cuda_id': 0,
        'use_multigpu':True,
        'n_bps': 512,

        'reg_coef': 5e-4,

        'base_lr': 5e-4,

        'best_model_fname': None, # trained without betas before
        'log_every_epoch': 2,
        'expr_code': expr_code,
        'work_dir': work_dir,
        'n_epochs': 10000,
        'dataset_dir': '/ps/scratch/body_hand_object_contact/bhoc_network/data/V01_01_00',
    }

    supercap_trainer = SuperCapTrainer(work_dir, ps=Configer(default_ps_fname=default_ps_fname, **params))
    ps = supercap_trainer.ps

    ps.dump_settings(os.path.join(work_dir, 'TR%02d_%s' % (ps.try_num, os.path.basename(default_ps_fname))))

    expr_message = '\n[%s] %d H neurons, batch_size=%d, BPS_NUM=512\n'% (ps.expr_code, ps.n_neurons, ps.batch_size)
    expr_message += 'Given BPS representation will output pose_body, root_orient and betas.\n'
    expr_message += 'L1 loss on the body mesh.\n'
    expr_message += 'Training on subsample of AMASS that are hard sampled using vposer+ easy samples.\n'
    expr_message += 'Female only BM is used.\n'
    expr_message += '\n'

    supercap_trainer.logger(expr_message)
    supercap_trainer.perform_training()
    ps.dump_settings(os.path.join(work_dir, 'TR%02d_%s' % (ps.try_num, os.path.basename(default_ps_fname))))