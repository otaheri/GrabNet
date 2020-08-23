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
import os
import argparse
import sys
sys.path.append('.')
sys.path.append('..')

import mano
from psbody.mesh import MeshViewers, Mesh
from psbody.mesh.colors import name_to_rgb
from grabnet.tools.vis_tools import vis_results
from grabnet.data.dataloader import LoadData
from grabnet.tools.cfg_parser import Config
from grabnet.train.trainer import Trainer


def inference(grabnet):

    grabnet.coarse_net.eval()
    grabnet.refine_net.eval()

    ds_name = 'test'
    mesh_base = '/ps/scratch/grab/data/object_meshes/contact_meshes'
    ds_test = LoadData(dataset_dir=grabnet.cfg.dataset_dir, ds_name=ds_name)
    n_samples = 5

    rh_model = mano.load(model_path=grabnet.cfg.rhm_path,
                         model_type='mano',
                         num_pca_comps=45,
                         batch_size=n_samples,
                         flat_hand_mean=True).to(grabnet.device)

    grabnet.refine_net.rhm_train = rh_model
    test_obj_names = np.unique(ds_test.frame_objs)

    grabnet.logger(f'################# \n'
                          f'Colors Guide:'
                          f'                   \n'
                          f'Red   --->  Reconstructed grasp - CoarseNet\n'
                          f'Green --->  Reconstructed grasp - Refinent\n'
                          f'Blue  --->  Ground Truth Grasp\n'
                          f'Pink  --->  Generated grasp - CoarseNet\n'
                          f'Gray  --->  Generated grasp - RefineNet\n')

    for obj in test_obj_names:

        obj_frames = np.where(ds_test.frame_objs == obj)[0]
        rnd_frames = np.random.choice(obj_frames.shape[0], n_samples)
        obj_data = ds_test[obj_frames[rnd_frames]]
        frame_data = {k: obj_data[k].to(grabnet.device) for k in obj_data.keys()}
        obj_meshes = []
        rotmats = []
        for frame in range(n_samples):
            rot_mat = frame_data['root_orient_obj_rotmat'][frame].cpu().numpy().reshape(3, 3).T
            transl  = frame_data['trans_obj'][frame].cpu().numpy()

            obj_mesh = Mesh(filename=os.path.join(mesh_base, obj + '.ply'), vc=name_to_rgb['yellow'])
            obj_mesh.rotate_vertices(rot_mat)
            obj_mesh.v += transl

            obj_meshes.append(obj_mesh)
            rotmats.append(rot_mat)
        frame_data['mesh_object'] = obj_meshes
        frame_data['rotmat'] = rotmats
        save_dir = os.path.join(grabnet.cfg.work_dir, 'test_grasp_results')
        grabnet.logger(f'#################\n'
                              f'                   \n'
                              f'Showing results for the {obj.upper()}'
                              f'                      \n')
        vis_results(dorig=frame_data,
                    coarse_net=grabnet.coarse_net,
                    refine_net=grabnet.refine_net,
                    rh_model=rh_model,
                    show_rec=True,
                    show_gen=True,
                    save=False,
                    save_dir=save_dir
                    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GrabNet-Testing')

    parser.add_argument('--data-path', default = None, type=str,
                        help='The path to the folder that contains GrabNet data')

    parser.add_argument('--rhm-path', default = None, type=str,
                        help='The path to the folder containing MANO_RIHGT model')

    parser.add_argument('--config-path', default = None, type=str,
                        help='The path to the confguration of the trained GrabNet model')

    args = parser.parse_args()

    cfg_path = args.config_path
    data_path = args.data_path
    rhm_path = args.rhm_path


    cwd = os.getcwd()

    best_cnet = 'grabnet/models/coarsenet.pt'
    best_rnet = 'grabnet/models/refinenet.pt'
    vpe_path  = 'grabnet/configs/verts_per_edge.npy'
    c_weights_path = 'grabnet/configs/rhand_weight.npy'
    work_dir = cwd + '/tests'

    if cfg_path is None:
        cfg_path = 'grabnet/configs/grabnet_cfg.yaml'

    config = {
        'work_dir':work_dir,
        'vpe_path': vpe_path,
        'c_weights_path': c_weights_path,

    }

    cfg = Config(default_cfg_path=cfg_path, **config)

    if data_path is not None:
        cfg['dataset_dir'] = data_path
    if rhm_path is not None:
        cfg['rhm_path'] = rhm_path
    if cfg.best_cnet is  None:
        cfg['best_cnet'] = best_cnet
    if cfg.best_rnet is None:
        cfg['best_rnet'] = best_rnet

    grabnet = Trainer(cfg=cfg, inference=True)
    inference(grabnet)

