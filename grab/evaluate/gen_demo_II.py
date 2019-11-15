'''
Using the same Z check out the grasp on different object conditions
'''
import os
from grab.tools.model_loader import load_grab
from human_body_prior.tools.omni_tools import makepath
import torch

from grab.tools.vis_tools import points_to_spheres
from human_body_prior.body_model.body_model import BodyModel
from psbody.mesh import Mesh

import numpy as np
from grab.data.dataloader import GRAB_DS
from torch.utils.data import DataLoader

from human_body_prior.mesh import MeshViewer
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import trimesh
from human_body_prior.tools.omni_tools import colors, id_generator
from human_body_prior.mesh.sphere import points_to_spheres
from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_

from grab.tools.object_model import ObjectModel

from human_body_prior.tools.visualization_tools import imagearray2file
import pickle
from grab.data.data_preparation import get_object_names

def eval_varying_cond_constant_z_different_view(dataset_dir, grab_model, gb_ps, samples_per_object=5):
    '''

    :param dataset_dir:
    :param grab_model:
    :param gb_ps:
    :param samples_per_object:
    :return:
    '''

    contacts_dir = '/ps/scratch/body_hand_object_contact/contact_results/17/03_thrshld_50e_6_final'
    object_names = get_object_names(contacts_dir)

    object_splits = {
        'test': ['mug', 'wineglass', 'camera', 'binoculars', 'fryingpan'],
        'vald': ['apple', 'toothpaste', 'elephant', 'hand']
    }
    object_splits['train'] = list(set(object_names).difference(set(object_splits['test'] + object_splits['vald'])))

    GRAB_ds = GRAB_DS(dataset_dir=dataset_dir)
    GRAB_loader = DataLoader(GRAB_ds, batch_size=1, shuffle=True, drop_last=False)

    splitname = dataset_dir.split('/')[-1]
    ds_name = dataset_dir.split('/')[-2]

    work_dir = os.path.join(gb_ps.work_dir, 'evaluations', 'ds_%s'%ds_name, os.path.basename(gb_ps.best_model_fname).replace('.pt',''), 'eval_varying_cond_constant_z_different_view','%s_samples'%splitname)

    print('dumping to %s' % work_dir)

    bm = BodyModel(gb_ps.bm_path, batch_size=1)
    hand_faces = c2c(bm.f)

    imw, imh = 800, 800

    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    mv.render_wireframe = False

    object_info_path = os.path.join(dataset_dir, '../object_infos.pkl')
    with open(object_info_path, 'rb') as f: object_info = pickle.load(f)

    view_angles=[0, 90, 180]

    for interested_object_name in object_splits[splitname]:
        # gather different verts objects of the same object

        object_v_template = object_info[interested_object_name]['verts_object']
        object_faces = object_info[interested_object_name]['faces_object']

        om = ObjectModel(v_template=object_v_template)
        imgpath = makepath(os.path.join(work_dir, '%s-%s.png' % (gb_ps.expr_code, interested_object_name)), isfile=True)

        Zgen = torch.tensor(np.random.normal(0., 1., size=(1, gb_ps.latentD)), dtype=torch.float32)

        images = np.zeros([len(view_angles), samples_per_object, 1, imw, imh, 3])
        cId = 0
        for bId, dorig in enumerate(GRAB_loader):
            frame_name = GRAB_ds.frame_names[dorig['idx']]
            object_name = frame_name.split('/')[-1].split('_')[0]
            if object_name != interested_object_name: continue

            gen_parms = grab_model.decode(Zgen, dorig['delta_object'])
            verts_hand_gen = c2c(bm(**gen_parms).v)

            object_mesh = points_to_spheres(c2c(dorig['verts_object'][0]), radius=0.008, vc=colors['green'])

            object_verts_highres = c2c(om(root_orient=dorig['root_orient_object'], trans=dorig['trans_object']).v[0])
            object_mesh_highres = Mesh(object_verts_highres, object_faces, vc=colors['blue'])

            pklpath = os.path.join(work_dir, '%s_%s.pkl'%(object_name, id_generator(6)))
            mesh_hand_gen = Mesh(verts_hand_gen[0], hand_faces, vc=colors['grey'])
            with open(pklpath, 'wb') as f:
                pickle.dump({'mesh_obj': object_mesh_highres, 'mesh_hand_gen':mesh_hand_gen}, f, protocol=2)

            object_mesh_highres.concatenate_mesh(mesh_hand_gen).write_obj(pklpath.replace('.pkl', '.obj'))
            object_mesh_highres.write_obj(pklpath.replace('.pkl', '.obj'))

            hand_mesh_gen = trimesh.Trimesh(vertices=verts_hand_gen[0], faces=hand_faces, vertex_colors=np.tile(colors['orange'], (6890, 1)))

            all_meshes = [hand_mesh_gen] + object_mesh

            for rId, angle in enumerate(view_angles):
                if angle != 0: apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(angle), (0, 1, 0)))
                mv.set_meshes([hand_mesh_gen] + object_mesh, group_name='static')
                images[rId, cId, 0] = mv.render()  # [:,:,:3]
                if angle != 0: apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(-angle), (0, 1, 0)))

            cId += 1
            if cId >= samples_per_object: break

        imagearray2file(images, imgpath)
        print('created to %s' % imgpath)

def eval_varying_cond_varying_z_different_view(dataset_dir, grab_model, gb_ps, samples_per_object=5):

    contacts_dir = '/ps/scratch/body_hand_object_contact/contact_results/17/03_thrshld_50e_6_final'
    object_names = get_object_names(contacts_dir)

    object_splits = {
        'test': ['mug', 'wineglass', 'camera', 'binoculars', 'fryingpan'],
        'vald': ['apple', 'toothpaste', 'elephant', 'hand']
    }
    object_splits['train'] = list(set(object_names).difference(set(object_splits['test'] + object_splits['vald'])))

    GRAB_ds = GRAB_DS(dataset_dir=dataset_dir)
    GRAB_loader = DataLoader(GRAB_ds, batch_size=1, shuffle=True, drop_last=False)

    splitname = dataset_dir.split('/')[-1]
    ds_name = dataset_dir.split('/')[-2]

    work_dir = os.path.join(gb_ps.work_dir, 'evaluations', 'ds_%s'%ds_name, os.path.basename(gb_ps.best_model_fname).replace('.pt',''), 'eval_varying_cond_varying_z_different_view','%s_samples'%splitname)

    print('dumping to %s' % work_dir)

    bm = BodyModel(gb_ps.bm_path, batch_size=1)
    hand_faces = c2c(bm.f)

    imw, imh = 800, 800

    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    mv.render_wireframe = False

    object_info_path = os.path.join(dataset_dir, '../object_infos.pkl')
    with open(object_info_path, 'rb') as f: object_info = pickle.load(f)

    view_angles=[0, 90, 180]

    for interested_object_name in object_splits[splitname]:
        # gather different verts objects of the same object

        object_v_template = object_info[interested_object_name]['verts_object']
        object_faces = object_info[interested_object_name]['faces_object']

        om = ObjectModel(v_template=object_v_template)
        imgpath = makepath(os.path.join(work_dir, '%s-%s.png' % (gb_ps.expr_code, interested_object_name)), isfile=True)

        images = np.zeros([len(view_angles), samples_per_object, 1, imw, imh, 3])
        cId = 0
        for bId, dorig in enumerate(GRAB_loader):
            frame_name = GRAB_ds.frame_names[dorig['idx']]
            object_name = frame_name.split('/')[-1].split('_')[0]
            if object_name != interested_object_name: continue

            Zgen = torch.tensor(np.random.normal(0., 1., size=(1, gb_ps.latentD)), dtype=torch.float32)

            gen_parms = grab_model.decode(Zgen, dorig['delta_object'])
            verts_hand_gen = c2c(bm(**gen_parms).v)

            object_mesh = points_to_spheres(c2c(dorig['verts_object'][0]), radius=0.008, vc=colors['green'])

            object_verts_highres = c2c(om(root_orient=dorig['root_orient_object'], trans=dorig['trans_object']).v[0])
            object_mesh_highres = Mesh(object_verts_highres, object_faces, vc=colors['blue'])

            pklpath = os.path.join(work_dir, '%s_%s.pkl'%(object_name, id_generator(6)))
            mesh_hand_gen = Mesh(verts_hand_gen[0], hand_faces, vc=colors['grey'])
            with open(pklpath, 'wb') as f:
                pickle.dump({'mesh_obj': object_mesh_highres, 'mesh_hand_gen':mesh_hand_gen}, f, protocol=2)

            object_mesh_highres.concatenate_mesh(mesh_hand_gen).write_obj(pklpath.replace('.pkl', '.obj'))
            object_mesh_highres.write_obj(pklpath.replace('.pkl', '.obj'))

            hand_mesh_gen = trimesh.Trimesh(vertices=verts_hand_gen[0], faces=hand_faces, vertex_colors=np.tile(colors['orange'], (6890, 1)))

            all_meshes = [hand_mesh_gen] + object_mesh

            for rId, angle in enumerate(view_angles):
                if angle != 0: apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(angle), (0, 1, 0)))
                mv.set_meshes([hand_mesh_gen] + object_mesh, group_name='static')
                images[rId, cId, 0] = mv.render()  # [:,:,:3]
                if angle != 0: apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(-angle), (0, 1, 0)))

            cId += 1
            if cId >= samples_per_object: break

        imagearray2file(images, imgpath)
        print('created to %s' % imgpath)

def eval_constant_cond_varying_z_same_view(dataset_dir, grab_model, gb_ps, batch_size=5, save_upto_bnum=5, dump_obj=False):

    GRAB_ds = GRAB_DS(dataset_dir=dataset_dir)
    GRAB_loader = DataLoader(GRAB_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    splitname = dataset_dir.split('/')[-1]
    ds_name = dataset_dir.split('/')[-2]

    work_dir = os.path.join(gb_ps.work_dir, 'evaluations', 'ds_%s'%ds_name, os.path.basename(gb_ps.best_model_fname).replace('.pt',''), 'eval_constant_cond_varying_z_same_view','%s_samples'%splitname)

    print('dumping to %s' % work_dir)

    bm = BodyModel(gb_ps.bm_path, batch_size=batch_size)

    imw, imh = 800, 800

    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    mv.render_wireframe = False


    object_info_path = os.path.join(dataset_dir, '../object_infos.pkl')
    with open(object_info_path, 'rb') as f:
        object_info = pickle.load(f)

    view_angles=[0, 90, 180]

    for bId, dorig in enumerate(GRAB_loader):
        imgpath = makepath(os.path.join(work_dir, '%s-%03d.png' % (gb_ps.expr_code, bId)), isfile=True)

        images = np.zeros([len(view_angles), len(dorig['delta_object']), 1, imw, imh, 3])

        hand_faces = c2c(bm.f)

        oId = np.random.choice(batch_size)
        constant_obj_delta = torch.repeat_interleave(dorig['delta_object'][oId:oId+1], repeats=batch_size, dim=0)
        Zgen = torch.tensor(np.random.normal(0., 1., size=(batch_size, gb_ps.latentD)), dtype=torch.float32)

        gen_parms = grab_model.decode(Zgen, constant_obj_delta)
        verts_hand_gen = c2c(bm(**gen_parms).v)

        object_mesh = points_to_spheres(c2c(dorig['verts_object'][oId]), radius=0.008, vc=colors['green'])

        if dump_obj:

            frame_names = GRAB_ds.frame_names[dorig['idx']]

            frame_name = frame_names[oId]
            object_name = frame_name.split('/')[-1].split('_')[0]
            object_v_template = object_info[object_name]['verts_object']
            object_faces = object_info[object_name]['faces_object']

            om = ObjectModel(v_template=object_v_template)
            object_verts_highres = c2c(om(root_orient=dorig['root_orient_object'][oId:oId + 1], trans=dorig['trans_object'][oId:oId + 1]).v[0])
            object_mesh_highres = Mesh(object_verts_highres, object_faces, vc=colors['blue'])

        for cId in range(batch_size):
            if dump_obj:
                pklpath = os.path.join(work_dir, '%s_%s.pkl'%(object_name, id_generator(6)))
                mesh_hand_gen = Mesh(verts_hand_gen[cId], hand_faces, vc=colors['grey'])
                with open(pklpath, 'wb') as f:
                    pickle.dump({'mesh_obj': object_mesh_highres, 'mesh_hand_gen':mesh_hand_gen}, f, protocol=2)
                mesh_hand_gen.concatenate_mesh(object_mesh_highres).write_obj(pklpath.replace('.pkl', '.obj'))
                mesh_hand_gen.write_obj(pklpath.replace('.pkl', '.obj'))

            hand_mesh_gen = trimesh.Trimesh(vertices=verts_hand_gen[cId], faces=hand_faces, vertex_colors=np.tile(colors['orange'], (6890, 1)))

            all_meshes = [hand_mesh_gen] + object_mesh

            for rId, angle in enumerate(view_angles):
                if angle != 0: apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(angle), (0, 1, 0)))
                mv.set_meshes([hand_mesh_gen] + object_mesh, group_name='static')
                images[rId, cId, 0] = mv.render()  # [:,:,:3]

                if angle != 0: apply_mesh_tranfsormations_(all_meshes, trimesh.transformations.rotation_matrix(np.radians(-angle), (0, 1, 0)))

        imagearray2file(images, imgpath)
        print('created to %s' % imgpath)
        if bId> save_upto_bnum: break

if __name__ == '__main__':
    # expr_code = 'V03_07_04' # Xube
    # expr_code = 'V03_07_08_64D'#
    expr_code = 'V03_07_04_TR03' # CVPR
    # expr_code = 'V03_07_04_TR05' #
    # expr_code = 'V03_07_04_TR07' #
    # expr_code = 'V03_07_04_TR03' # good one

    # expr_code = 'V03_07_10_128D' # not bad
    # expr_code = 'V03_07_09_64D' # no good finger poses but fails with the frying pan
    # expr_code = 'V03_07_14_64D' # 10 perce drop using only pick up data. the wine glass pickup does not touch the glass

    # expr_code = 'V03_07_13_64D' # 20 perce drop with 01_12 dataset. the wine g
    data_code = 'V01_11_00'
    # data_code = 'V01_12_00'
    # data_code = 'V01_07_00'

    expr_basedir = '/ps/scratch/body_hand_object_contact/grab_net/experiments'
    dataset_dir = '/ps/scratch/body_hand_object_contact/grab_net/data/%s' % (data_code)

    # for expr_code in ['V03_07_08_64D', 'V03_07_07_32D', 'V03_07_07_64D' ]:
    expr_dir = os.path.join(expr_basedir, expr_code)
    grab_model, gb_ps = load_grab(expr_dir)

    # for splitname in ['test', ]:
    for splitname in ['test', 'vald', 'train']:
        eval_varying_cond_constant_z_different_view(os.path.join(dataset_dir, splitname), grab_model, gb_ps, samples_per_object=10)
        eval_varying_cond_varying_z_different_view(os.path.join(dataset_dir, splitname), grab_model, gb_ps, samples_per_object=12)
        eval_constant_cond_varying_z_same_view(os.path.join(dataset_dir, splitname), grab_model, gb_ps, batch_size=5, dump_obj=True)