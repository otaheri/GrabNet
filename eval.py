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

import os
import argparse
from GrabNet.grabnet.tools.cfg_parser import Config
from GrabNet.grabnet.train.trainer import Trainer


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GrabNet-Training')

    parser.add_argument('--data-path', default = None, type=str,
                        help='The path to the folder that contains GrabNet data')

    parser.add_argument('--rhm-path', default = None, type=str,
                        help='The path to the folder containing MANO_RIHGT model')

    parser.add_argument('--config-path', required=True, type=str,
                        help='The path to the confguration of the trained GrabNet model')

    args = parser.parse_args()

    cfg_path = args.config_path
    data_path = args.data_path
    rhm_path = args.rhm_path

    cwd = os.getcwd()

    congfig = {
        'dataset_dir': data_path,
        'rhm_path': rhm_path,
        'vpe_path': cwd + '/grabnet/configs/verts_per_edge.npy',
        'c_weights_path': cwd + '/grabnet/configs/rhand_weight.npy',
    }

    cfg = Config(default_cfg_path=cfg_path)

    if data_path is not None:
        cfg['dataset_dir'] = data_path
    if rhm_path is not None:
        cfg['rhm_path'] = rhm_path

    grabnet_trainer = Trainer(cfg=cfg)

    grabnet_trainer.eval()