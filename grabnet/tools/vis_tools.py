
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
import torch
import numpy as np
from psbody.mesh import Mesh, MeshViewers
from psbody.mesh.sphere import Sphere
from psbody.mesh.colors import name_to_rgb
from grabnet.tools.train_tools import point2point_signed
from grabnet.tools.utils import aa2rotmat
from grabnet.tools.utils import makepath
from grabnet.tools.utils import to_cpu


def vis_results(dorig, coarse_net, refine_net, rh_model, show_gen=True, show_rec=True, save=False, save_dir = None):

    with torch.no_grad():
        imw, imh = 400, 1000
        cols = len(dorig['bps_object'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if show_rec:
            mvs = MeshViewers(window_width=imw * cols, window_height=imh, shape=[3, cols], keepalive=True)
            drec_cnet = coarse_net(**dorig)
            verts_rh_rec_cnet = rh_model(**drec_cnet).vertices

            _, h2o, _ = point2point_signed(verts_rh_rec_cnet, dorig['verts_object'])

            drec_cnet['trans_rhand_f'] = drec_cnet['transl']
            drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(drec_cnet['global_orient']).view(-1, 3, 3)
            drec_cnet['fpose_rhand_rotmat_f'] = aa2rotmat(drec_cnet['hand_pose']).view(-1, 15, 3, 3)
            drec_cnet['verts_object'] = dorig['verts_object']
            drec_cnet['h2o_dist']= h2o.abs()

            drec_rnet = refine_net(**drec_cnet)
            verts_rh_rec_rnet = rh_model(**drec_rnet).vertices

            for cId in range(0, len(dorig['bps_object'])):
                try:
                    from copy import deepcopy
                    meshes = deepcopy(dorig['mesh_object'])
                    obj_mesh = meshes[cId]
                except:
                    obj_mesh = points_to_spheres(points=to_cpu(dorig['verts_object'][cId]), radius=0.002, vc=name_to_rgb['green'])


                hand_mesh_orig = Mesh(v=to_cpu(dorig['verts_rhand'][cId]), f=rh_model.faces, vc=name_to_rgb['blue'])
                hand_mesh_rec_cnet= Mesh(v=to_cpu(verts_rh_rec_cnet[cId]), f=rh_model.faces, vc=name_to_rgb['green'])
                hand_mesh_rec_rnet = Mesh(v=to_cpu(verts_rh_rec_rnet[cId]), f=rh_model.faces, vc=name_to_rgb['red'])

                if 'rotmat' in dorig:
                    rotmat = dorig['rotmat'][cId].T
                    obj_mesh = obj_mesh.rotate_vertices(rotmat)
                    hand_mesh_orig.rotate_vertices(rotmat)
                    hand_mesh_rec_cnet.rotate_vertices(rotmat)
                    hand_mesh_rec_rnet.rotate_vertices(rotmat)

                hand_mesh_rec_cnet.reset_face_normals()
                hand_mesh_rec_rnet.reset_face_normals()
                hand_mesh_orig.reset_face_normals()

                mvs[0][cId].set_static_meshes([hand_mesh_orig, obj_mesh], blocking=True)
                mvs[1][cId].set_static_meshes([hand_mesh_rec_cnet, obj_mesh], blocking=True)
                mvs[2][cId].set_static_meshes([hand_mesh_rec_rnet, obj_mesh], blocking=True)

                if save:
                    save_path = os.path.join(save_dir, str(cId))
                    makepath(save_path)
                    hand_mesh_rec_rnet.write_ply(filename=save_path + '/rh_mesh_gen_%d.ply' % cId)
                    obj_mesh[0].write_ply(filename=save_path + '/obj_mesh_%d.ply' % cId)

        if show_gen:
            mvs = MeshViewers(window_width=imw * cols, window_height=imh, shape=[2, cols], keepalive=True)

            drec_cnet = coarse_net.sample_poses(dorig['bps_object'])
            verts_rh_gen_cnet = rh_model(**drec_cnet).vertices

            _, h2o, _ = point2point_signed(verts_rh_gen_cnet, dorig['verts_object'].to(device))

            drec_cnet['trans_rhand_f'] = drec_cnet['transl']
            drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(drec_cnet['global_orient']).view(-1, 3, 3)
            drec_cnet['fpose_rhand_rotmat_f'] = aa2rotmat(drec_cnet['hand_pose']).view(-1, 15, 3, 3)
            drec_cnet['verts_object'] = dorig['verts_object'].to(device)
            drec_cnet['h2o_dist']= h2o.abs()

            drec_rnet = refine_net(**drec_cnet)
            verts_rh_gen_rnet = rh_model(**drec_rnet).vertices


            for cId in range(0, len(dorig['bps_object'])):
                try:
                    from copy import deepcopy
                    meshes = deepcopy(dorig['mesh_object'])
                    obj_mesh = meshes[cId]
                except:
                    obj_mesh = points_to_spheres(to_cpu(dorig['verts_object'][cId]), radius=0.002, vc=name_to_rgb['green'])

                hand_mesh_gen_cnet = Mesh(v=to_cpu(verts_rh_gen_cnet[cId]), f=rh_model.faces, vc=name_to_rgb['pink'])
                hand_mesh_gen_rnet = Mesh(v=to_cpu(verts_rh_gen_rnet[cId]), f=rh_model.faces, vc=name_to_rgb['gray'])

                if 'rotmat' in dorig:
                    rotmat = dorig['rotmat'][cId].T
                    obj_mesh = obj_mesh.rotate_vertices(rotmat)
                    hand_mesh_gen_cnet.rotate_vertices(rotmat)
                    hand_mesh_gen_rnet.rotate_vertices(rotmat)

                hand_mesh_gen_cnet.reset_face_normals()
                hand_mesh_gen_rnet.reset_face_normals()

                mvs[0][cId].set_static_meshes([hand_mesh_gen_cnet, obj_mesh], blocking=True)
                mvs[1][cId].set_static_meshes([hand_mesh_gen_rnet, obj_mesh], blocking=True)

                if save:
                    save_path = os.path.join(save_dir, str(cId))
                    makepath(save_path)
                    hand_mesh_gen_rnet.write_ply(filename=save_path + '/rh_mesh_gen_%d.ply' % cId)
                    obj_mesh[0].write_ply(filename=save_path + '/obj_mesh_%d.ply' % cId)



def points_to_spheres(points, radius=0.1, vc=name_to_rgb['blue']):

    spheres = Mesh(v=[], f=[])
    for pidx, center in enumerate(points):
        clr = vc[pidx] if len(vc) > 3 else vc
        spheres.concatenate_mesh(Sphere(center, radius).to_mesh(color=clr))
    return spheres

def cage(length=1,vc=name_to_rgb['black']):

    cage_points = np.array([[-1., -1., -1.],
                            [1., 1., 1.],
                            [1., -1., 1.],
                            [-1., 1., -1.]])
    c = Mesh(v=length * cage_points, f=[], vc=vc)
    return c



def create_video(path, fps=30,name='movie'):
    import os
    import subprocess

    src = os.path.join(path,'%*.png')
    movie_path = os.path.join(path,'%s.mp4'%name)
    i = 0
    while os.path.isfile(movie_path):
        movie_path = os.path.join(path,'%s_%02d.mp4'%(name,i))
        i +=1


    cmd = 'ffmpeg -f image2 -r %d -i %s -b:v 6400k -pix_fmt yuv420p %s' % (fps, src, movie_path)
    subprocess.call(cmd.split(' '))
    while not os.path.exists(movie_path):
        continue
