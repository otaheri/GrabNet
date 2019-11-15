'''
Using the same Z check out the grasp on different object conditions
'''
import os
from grab.tools.model_loader import load_grab
from human_body_prior.tools.omni_tools import makepath
import torch
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import colors
from grab.tools.vis_tools import points_to_spheres
from human_body_prior.body_model.body_model import BodyModel

import numpy as np
from grab.data.dataloader import GRAB_DS
from torch.utils.data import DataLoader

from human_body_prior.mesh import MeshViewer
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import trimesh
from human_body_prior.tools.omni_tools import colors
from human_body_prior.mesh.sphere import points_to_spheres
from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_

from human_body_prior.tools.visualization_tools import imagearray2file

def eval_constant_z_varying_cond(dataset_dir, grab_model, gb_ps, batch_size=5, save_upto_bnum=5):

    ds = GRAB_DS(dataset_dir=dataset_dir)
    ds = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    splitname = dataset_dir.split('/')[-1]
    ds_name = dataset_dir.split('/')[-2]

    outpath = os.path.join(gb_ps.work_dir, 'evaluations', 'ds_%s'%ds_name, os.path.basename(gb_ps.best_model_fname).replace('.pt',''), 'eval_constant_z_varying_cond','%s_samples'%splitname)

    print('dumping to %s' % outpath)

    bm = BodyModel(gb_ps.bm_path, batch_size=batch_size)

    imw, imh = 800, 800

    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    mv.render_wireframe = False

    rnd_normal = np.random.normal(0., 1., size=(1, gb_ps.latentD))
    Zgen = torch.tensor(np.repeat(rnd_normal, repeats=batch_size, axis=0), dtype=torch.float32)

    view_angles=[0, 90, 180]

    for bId, dorig in enumerate(ds):
        imgpath = makepath(os.path.join(outpath, '%s-%03d.png' % (gb_ps.expr_code, bId)), isfile=True)

        images = np.zeros([len(view_angles) + 1, len(dorig['delta_object']), 1, imw, imh, 3])

        verts_hand_orig = bm(**dorig).v

        gen_parms = grab_model.decode(Zgen, dorig['delta_object'])
        verts_hand_gen = bm(**gen_parms).v


        for cId in range(batch_size):

            object_mesh = points_to_spheres(c2c(dorig['verts_object'][cId]), radius=0.008, vc=colors['green'])

            hand_mesh_orig = trimesh.Trimesh(vertices=c2c(verts_hand_orig[cId]), faces=c2c(bm.f), vertex_colors=np.tile(colors['blue'], (6890, 1)))

            hand_mesh_gen = trimesh.Trimesh(vertices=c2c(verts_hand_gen[cId]), faces=c2c(bm.f), vertex_colors=np.tile(colors['orange'], (6890, 1)))

            all_meshes = [hand_mesh_orig, hand_mesh_gen] + object_mesh

            mv.set_meshes([hand_mesh_orig] + object_mesh, group_name='static')
            images[0, cId, 0] = mv.render()  # [:,:,:3]

            for rId, angle in enumerate(view_angles):
                if angle != 0: apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(angle), (0, 1, 0)))
                mv.set_meshes([hand_mesh_gen] + object_mesh, group_name='static')
                images[rId + 1, cId, 0] = mv.render()  # [:,:,:3]

                if angle != 0: apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(-angle), (0, 1, 0)))

        imagearray2file(images, imgpath)
        print('created to %s' % imgpath)
        if bId> save_upto_bnum: break


if __name__ == '__main__':
    expr_code = 'V03_07_05'
    data_code = 'V01_07_00'

    expr_basedir = '/ps/scratch/body_hand_object_contact/grab_net/experiments'
    expr_dir = os.path.join(expr_basedir, expr_code)
    dataset_dir = '/ps/scratch/body_hand_object_contact/grab_net/data/%s' % (data_code)

    grab_model, gb_ps = load_grab(expr_dir)

    for splitname in ['test']:
    # for splitname in ['test', 'vald', 'train']:
        eval_constant_z_varying_cond(os.path.join(dataset_dir, splitname), grab_model, gb_ps, batch_size=5)