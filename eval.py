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
import os
import argparse
from grabnet.tools.cfg_parser import Config
from grabnet.train.trainer import Trainer


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GrabNet-Training')

    parser.add_argument('--data-path', default=None, type=str,
                        help='The path to the folder that contains GrabNet data')

    parser.add_argument('--rhm-path', default=None, type=str,
                        help='The path to the folder containing MANO_RIHGT model')

    parser.add_argument('--config-path', default=None, type=str,
                        help='The path to the confguration of the trained GrabNet model')

    args = parser.parse_args()

    cfg_path = args.config_path
    data_path = args.data_path
    rhm_path = args.rhm_path

    cwd = os.getcwd()

    best_cnet = 'grabnet/models/coarsenet.pt'
    best_rnet = 'grabnet/models/refinenet.pt'
    vpe_path = 'grabnet/configs/verts_per_edge.npy'
    c_weights_path = 'grabnet/configs/rhand_weight.npy'
    work_dir = cwd + '/eval'

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

    grabnet_trainer = Trainer(cfg=cfg)

    grabnet_trainer.eval()