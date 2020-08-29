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
import os
import argparse

import mano
from psbody.mesh import MeshViewers, Mesh
from grabnet.tools.vis_tools import vis_results

from grabnet.tools.utils import euler
from grabnet.tools.cfg_parser import Config
from grabnet.train.trainer import Trainer

from bps_torch.bps import bps_torch

def grab_new_objs(grabnet, objs_path, rot=True, n_samples=10, scale=1.):
    
    grabnet.coarse_net.eval()
    grabnet.refine_net.eval()

    rh_model = mano.load(model_path=grabnet.cfg.rhm_path,
                         model_type='mano',
                         num_pca_comps=45,
                         batch_size=n_samples,
                         flat_hand_mean=True).to(grabnet.device)

    grabnet.refine_net.rhm_train = rh_model

    grabnet.logger(f'################# \n'
                   f'Colors Guide:'
                   f'                   \n'
                   f'Pink  --->  CoarseNet generated grasp\n'
                   f'Gray  --->  RefineNet generated grasp\n')

    bps = bps_torch(custom_basis = grabnet.bps)

    if not isinstance(objs_path, list):
        objs_path = [objs_path]
        
    for new_obj in objs_path:

        rand_rotdeg = np.random.random([n_samples, 3]) * np.array([360, 360, 360])

        rand_rotmat = euler(rand_rotdeg)
        dorig = {'bps_object': [],
                 'verts_object': [],
                 'mesh_object': [],
                 'rotmat':[]}

        for samples in range(n_samples):

            verts_obj, mesh_obj, rotmat = load_obj_verts(new_obj, rand_rotmat[samples], rndrotate=rot, scale=scale)
            
            bps_object = bps.encode(verts_obj, feature_type='dists')['dists']

            dorig['bps_object'].append(bps_object.to(grabnet.device))
            dorig['verts_object'].append(torch.from_numpy(verts_obj.astype(np.float32)).unsqueeze(0))
            dorig['mesh_object'].append(mesh_obj)
            dorig['rotmat'].append(rotmat)
            obj_name = os.path.basename(new_obj)

        dorig['bps_object'] = torch.cat(dorig['bps_object'])
        dorig['verts_object'] = torch.cat(dorig['verts_object'])

        save_dir = os.path.join(grabnet.cfg.work_dir, 'grab_new_objects')
        grabnet.logger(f'#################\n'
                              f'                   \n'
                              f'Showing results for the {obj_name.upper()}'
                              f'                      \n')

        vis_results(dorig=dorig,
                    coarse_net=grabnet.coarse_net,
                    refine_net=grabnet.refine_net,
                    rh_model=rh_model,
                    show_rec=False,
                    show_gen=True,
                    save=False,
                    save_dir=save_dir
                    )

def load_obj_verts(mesh_path, rand_rotmat, rndrotate=True, scale=1000., n_sample_verts=5000):

    np.random.seed(100)
    obj_mesh = Mesh(filename=mesh_path, vscale=scale)

    obj_mesh.reset_normals()
    obj_mesh.vc = obj_mesh.colors_like('green')

    ## center and scale the object
    max_length = np.linalg.norm(obj_mesh.v, axis=1).max()
    if  max_length > .15:
        re_scale = max_length/.08
        print(f'The object is very large, down-scaling by {re_scale} factor')
        obj_mesh.v = obj_mesh.v/re_scale

    object_fullpts = obj_mesh.v
    maximum = object_fullpts.max(0, keepdims=True)
    minimum = object_fullpts.min(0, keepdims=True)

    offset = ( maximum + minimum) / 2
    verts_obj = object_fullpts - offset
    obj_mesh.v = verts_obj

    if rndrotate:
        obj_mesh.rotate_vertices(rand_rotmat)
        verts_obj = obj_mesh.v
    
    if verts_obj.shape[0] > n_sample_verts:
        verts_sample_id = np.random.choice(verts_obj.shape[0], n_sample_verts, replace=False)
    else:
        verts_sample_id = np.arange(verts_obj.shape[0])
    verts_sampled = verts_obj[verts_sample_id]

    return verts_sampled, obj_mesh, rand_rotmat

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GrabNet-Testing')

    parser.add_argument('--data-path', default=None, type=str,
                        help='The path to the folder that contains GrabNet data')

    parser.add_argument('--obj-path', required = True, type=str,
                        help='The path to the 3D object Mesh or Pointcloud')

    parser.add_argument('--rhm-path', default = None, type=str,
                        help='The path to the folder containing MANO_RIHGT model')

    parser.add_argument('--config-path', default= None, type=str,
                        help='The path to the confguration of the trained GrabNet model')

    args = parser.parse_args()

    cfg_path = args.config_path
    data_path = args.data_path
    obj_path = args.obj_path
    rhm_path = args.rhm_path

    cwd = os.getcwd()
    work_dir = cwd + '/test'

    best_cnet = 'grabnet/models/coarsenet.pt'
    best_rnet = 'grabnet/models/refinenet.pt'
    vpe_path = 'grabnet/configs/verts_per_edge.npy'
    c_weights_path = 'grabnet/configs/rhand_weight.npy'

    if cfg_path is None:
        cfg_path = 'grabnet/configs/grabnet_cfg.yaml'


    config = {
        'work_dir': work_dir,
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
    grab_new_objs(grabnet,obj_path)
