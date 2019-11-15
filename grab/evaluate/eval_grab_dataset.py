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
# AMASS: Archive of Motion Capture as Surface Shapes <https://arxiv.org/abs/1904.03278>
#
#
# Code Developed by:
# Nima Ghorbani <https://www.linkedin.com/in/nghorbani/>
#
# 2019.05.28

import os
import json
from tqdm import tqdm

from grab.tools.model_loader import load_grab
from grab.data.dataloader import GRAB_DS
from grab.train.grab_net import GRABTrainer

from human_body_prior.tools.omni_tools import makepath
from human_body_prior.tools.omni_tools import copy2cpu as c2c

from human_body_prior.body_model.body_model import BodyModel

from torch.utils.data import DataLoader
import torch

def visualize_results(dataset_dir, grab_model, gb_ps, batch_size=5, save_upto_bnum=5):

    ds_name = dataset_dir.split('/')[-2]
    splitname = dataset_dir.split('/')[-1]

    assert splitname in ['test', 'train', 'vald']
    comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    grab_model.eval()
    grab_model = grab_model.to(comp_device)

    with torch.no_grad():
        bm = BodyModel(gb_ps.bm_path, batch_size=batch_size).to(comp_device)

    ds = GRAB_DS(dataset_dir=os.path.join(dataset_dir, splitname))
    ds = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    outpath = os.path.join(gb_ps.work_dir, 'evaluations', 'ds_%s'%ds_name, os.path.basename(gb_ps.best_model_fname).replace('.pt',''), 'eval_on_grab','%s_samples'%splitname)
    print('dumping to %s'%outpath)

    for bId, dorig in enumerate(ds):
        dorig = {k: dorig[k].to(comp_device) for k in dorig.keys()}

        imgpath = makepath(os.path.join(outpath, '%s-%03d.png' % (gb_ps.expr_code, bId)), isfile=True)
        GRABTrainer.vis_results(dorig, grab_model, bm, imgpath, view_angles=[0, 90, 180], show_gen=False)

        if bId> save_upto_bnum: break


def evaluate_error(dataset_dir, grab_model, gb_ps, batch_size=512):
    grab_model.eval()

    ds_name = dataset_dir.split('/')[-1]

    comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bm = BodyModel(gb_ps.bm_path, batch_size=batch_size).to(comp_device)
    grab_model = grab_model.to(comp_device)

    final_errors = {}
    for splitname in ['test', 'train', 'vald']:

        ds = GRAB_DS(dataset_dir=os.path.join(dataset_dir, splitname))
        # print('%s dataset size: %s'%(splitname,len(ds)))
        ds = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)#batchsize for bm is fixed so drop the last one

        loss_mean = []
        with torch.no_grad():
            for dorig in ds:
                dorig = {k: dorig[k].to(comp_device) for k in dorig.keys()}

                MESH_SCALER = 1000

                drec = grab_model(**dorig)
                verts_hand_mano = bm(**drec).v

                # from psbody.mesh import Mesh
                # a = Mesh(c2c(verts_hand_mano[0]), c2c(bm.f))
                # b = Mesh(c2c(dorig['verts_hand_mano'][0]), c2c(bm.f))
                # a.concatenate_mesh(b).show()

                # loss_mean.append(torch.mean(torch.sqrt(torch.pow((mesh_orig - mesh_rec)* MESH_SCALER, 2))))
                loss_mean.append(torch.mean(torch.abs(dorig['verts_hand_mano'] - verts_hand_mano)* MESH_SCALER))

        final_errors[splitname] = {'v2v_mae': float(c2c(torch.stack(loss_mean).mean()))}

    outpath = makepath(os.path.join(gb_ps.work_dir, 'evaluations', 'ds_%s'%ds_name, os.path.basename(gb_ps.best_model_fname).replace('.pt','.json')), isfile=True)
    with open(outpath, 'w') as f:
        json.dump(final_errors,f)

    return final_errors

if __name__ == '__main__':
    expr_code = 'V03_07_04'
    data_code = 'V01_07_00'

    expr_basedir = '/ps/scratch/body_hand_object_contact/grab_net/experiments'

    expr_dir = os.path.join(expr_basedir, expr_code)
    grab_model, gb_ps = load_grab(expr_dir)
    # dataset_dir = gb_ps.dataset_dir
    dataset_dir = '/ps/scratch/body_hand_object_contact/grab_net/data/%s'%data_code

    # for splitname in ['test', 'vald', 'train']:
    #    visualize_results(os.path.join(dataset_dir, splitname), grab_model, gb_ps, batch_size=3)

    final_errors = evaluate_error(dataset_dir, grab_model, gb_ps, batch_size=512)
    print('[%s] [DS: %s] -- %s' % (gb_ps.best_model_fname, dataset_dir,  ', '.join(['%s: %.2e'%(k, v['v2v_mae']) for k,v in final_errors.items()])))
