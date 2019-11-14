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
#
# 2019.11.14

import os, glob


def expid2model(expr_dir):
    from configer import Configer

    if not os.path.exists(expr_dir): raise ValueError('Could not find the experiment directory: %s' % expr_dir)

    best_model_fname = sorted(glob.glob(os.path.join(expr_dir, 'snapshots', '*.pt')), key=os.path.getmtime)[-1]

    print(('Found GRAB Trained Model: %s' % best_model_fname))

    default_ps_fname = glob.glob(os.path.join(expr_dir,'*.ini'))[0]
    if not os.path.exists(
        default_ps_fname): raise ValueError('Could not find the appropriate grab_settings: %s' % default_ps_fname)
    ps = Configer(default_ps_fname=default_ps_fname, work_dir = expr_dir, best_model_fname=best_model_fname)

    return ps, best_model_fname

def load_grab(expr_dir, grab_model='snapshot'):
    '''

    :param expr_dir:
    :param grab_model: either 'snapshot' to use the experiment folder's code or a GRAB imported module, e.g.
    from grab.train.grab_smpl import GRAB, then pass GRAB to this function
    :param if True will load the model definition used for training, and not the one in current repository
    :return:
    '''
    import importlib
    import torch

    ps, trained_model_fname = expid2model(expr_dir)
    if grab_model == 'snapshot':

        grab_path = sorted(glob.glob(os.path.join(expr_dir, '*.py')), key=os.path.getmtime)[-1]

        spec = importlib.util.spec_from_file_location('grab', grab_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        grab_pt = getattr(module, 'BHOC')(**ps)
    else:
        grab_pt = grab_model(**ps)

    grab_pt.load_state_dict(torch.load(trained_model_fname, map_location='cpu'))
    grab_pt.eval()

    return grab_pt, ps


if __name__ == '__main__':
    expr_dir = '/ps/scratch/body_hand_object_contact/grab_net/experiments/V03_04_01'
    grab_pt, ps = load_grab(expr_dir, grab_model='snapshot')
