import numpy as np
import cPickle as pickle
# from fit_rigid_object2mocap import RigidObjectModel
from psbody.mesh import Mesh, MeshViewers, MeshViewer
from experiments.hand.smpl_handpca_wrapper import load_model
import torch

from experiments.nima.tools.mocap_interface import MocapSession
from experiments.omid.object_interaction.fit_rigid_object2mocap import get_marker_data
from experiments.omid.object_interaction.mesh2mesh_contact import point2point
from experiments.omid.object_interaction.mesh2mesh_contact import get_first_contact_surface
from experiments.omid.tools.dir_utils import makepath

from experiments.nima.tools.visualization_helpers import spheres_for

from experiments.hand.fast_derivatives.smpl_HF_fastderivatives_WRAPPER import load_model as load_smplx
from experiments.hand.fast_derivatives.smpl_HF_fastderivatives import SmplModelLBS

from experiments.nima.tools_ch.object_model import RigidObjectModel
from psbody.mesh.colors import name_to_rgb
from experiments.omid.tools.run_parallel import perform_parallel_tasks
from body.geometry.angles import euler


import sys, os, glob
sys.path.append("/is/ps2/otaheri/frankengeist/experiments/omid/chamfer-extension")
import dist_chamfer as chdist

def create_video(path, fps=30,name='movie'):
    import imageio
    import os
    import glob
    import subprocess

    fileList = []

    filename = os.path.join(path,'*.png')

    fileList = [img for img in sorted(glob.glob(filename))]


    # writer = imageio.get_writer(os.path.join(path,'test.mp4'), fps=fps)

    src = os.path.join(path,'%05d.png')
    movie_path = os.path.join(path,'..','%s.mp4'%name)

    cmd = 'ffmpeg -f image2 -r %d -i %s -b:v 6400k -pix_fmt yuv420p %s' % (fps, src, movie_path)
    # cmd = 'ffmpeg -y -threads 16 -framerate %d -i %s -vcodec h264 -pix_fmt yuv420p -an -b:v 5000k %s'% (fps, src, movie_path)
    subprocess.call(cmd.split(' '))
    while not os.path.exists(movie_path):
        continue
    import shutil
    shutil.rmtree(path, ignore_errors=True)
    # os.mkdir(path)


    # for im in fileList:
    #     writer.append_data(imageio.imread(im))
    # writer.close()

def run_parallel_contact():

    mosh_expr_id = '16_omid'
    # contact_expr_id = '01'
    mosh_subject_id = '191001_00158'
    # mosh_subject_id = '191002_00177'
    # mosh_subject_id = '191004_03331'
    # mosh_subject_id = '191011_00178'
    # mosh_subject_id = '191023_00034'
    # gender = 'male'

    # mrk_radius = 0.0032
    # show_markers = False
    # save_video = False
    # contact_threshold = 3e-3

    mosh_results_dir = '/ps/scratch/body_hand_object_contact/mosh_results/%s' % mosh_expr_id
    # contact_results_dir = '/ps/scratch/body_hand_object_contact/contact_results/%s' % mosh_expr_id


    subject_results_dir = os.path.join(mosh_results_dir,'mosh_subject', mosh_subject_id)
    # object_results_dir = os.path.join(mosh_results_dir,'mosh_object', mosh_subject_id)
    # object_mesh_dir = '/ps/scratch/body_hand_object_contact/data/object_meshes/contact_meshes'


    # model_file = os.path.join('/ps/scratch/common/moshpp/smplx/unlocked_head',gender,'model.pkl')
    # v_template_file = os.path.join('/ps/scratch/body_hand_object_contact/data/subject_meshes',gender,'%s.ply'%mosh_subject_id)
    # print v_template_file
    # v_template = Mesh(filename=v_template_file).v
    # smplx = load_smplx(fname_or_dict=model_file, ncomps= 24, v_template=v_template)

    subject_result_files = sorted(glob.glob(os.path.join(subject_results_dir,'*_stageII.pkl')))
    # subject_result_files = glob.glob(os.path.join(subject_results_dir,'*_poses.pkl'))


    total_jobs = []
    for s_result in subject_result_files:
        # args = {'smplx':smplx,
        #         's_result':s_result,
        #         'v_template_file':v_template_file,
        #         'model_file':model_file,
        #         'contact_results_dir':contact_results_dir,
        #         'mosh_subject_id':mosh_subject_id,
        #         'mosh_expr_id':mosh_expr_id,
        #         'contact_expr_id':contact_expr_id,
        #         'objects_results_dir':object_results_dir,
        #         'object_mesh_dir':object_mesh_dir,
        #         }
        total_jobs.append({'s_result':s_result})

    # total_jobs = [total_jobs[9]]
    for item in total_jobs:
        if 'cup_drink' in item['s_result']:
            total_jobs = [item]
            break
    # total_jobs = total_jobs[62:67] + total_jobs[108:113] + [total_jobs[160]] + total_jobs[170:174]
    # total_jobs = [total_jobs[160]]
    print('# total_jobs = %d'%len(total_jobs))
    perform_parallel_tasks(total_jobs, run_get_contact, poolsize=1,priority=1000,cpu=1,gpu=1, mem=10, username='otaheri')

def run_get_contact(ds):
    import numpy as np
    import cPickle as pickle
    from psbody.mesh import Mesh, MeshViewers, MeshViewer
    from experiments.omid.object_interaction.mesh2mesh_contact import get_first_contact_surface
    from experiments.omid.tools.dir_utils import makepath
    from experiments.nima.tools.visualization_helpers import spheres_for
    from experiments.hand.fast_derivatives.smpl_HF_fastderivatives_WRAPPER import load_model as load_smplx
    from experiments.nima.tools_ch.object_model import RigidObjectModel
    from psbody.mesh.colors import name_to_rgb
    from experiments.omid.tools.run_parallel import create_video
    from body.geometry.angles import euler



    import sys, os, glob
    sys.path.append("/is/ps2/otaheri/frankengeist/experiments/omid/chamfer-extension")
    import dist_chamfer as chdist

    mosh_expr_id = '16_omid'
    contact_expr_id = '01_thrshld_15e_6_final'
    mosh_subject_id = '191001_00158'
    # mosh_subject_id = '191002_00177'
    # mosh_subject_id = '191004_03331'
    # mosh_subject_id = '191011_00178'
    # mosh_subject_id = '191023_00034'

    gender = 'male'
    # gender = 'female'

    table = 'False'

    mrk_radius = 0.0032
    show_markers = False
    contact_threshold = 1.5e-5


    visualize_contact=True
    save_video = False
    Rxyz = [-90, 0, 0]
    Rxyz = euler(Rxyz)


    mosh_results_dir = '/ps/scratch/body_hand_object_contact/mosh_results/%s' % mosh_expr_id
    contact_results_dir = '/ps/scratch/body_hand_object_contact/contact_results/%s' % mosh_expr_id

    # subject_results_dir = os.path.join(mosh_results_dir,'mosh_subject', mosh_subject_id)
    Table_results_dir = os.path.join(mosh_results_dir+'_Table','mosh_object','objects_stageII', mosh_subject_id)
    object_results_dir = os.path.join(mosh_results_dir,'mosh_object','objects_stageII', mosh_subject_id)
    # object_results_dir = os.path.join(mosh_results_dir,'mosh_object', mosh_subject_id)
    object_mesh_dir = '/ps/scratch/body_hand_object_contact/data/object_meshes/contact_meshes'

    model_file = os.path.join('/ps/scratch/common/moshpp/smplx/unlocked_head',gender,'model.pkl')
    v_template_file = os.path.join('/ps/scratch/body_hand_object_contact/data/subject_meshes',gender,'%s.ply'%mosh_subject_id)

    v_template = Mesh(filename=v_template_file).v
    smplx = load_smplx(fname_or_dict=model_file, ncomps= 24, v_template=v_template)
    # subject_result_files = glob.glob(os.path.join(subject_results_dir,'*_poses.pkl'))

    if visualize_contact:
        mvs = MeshViewers(window_width=2400, window_height=1600, shape=[1, 2])
        mvs[0][0].background_color = np.ones(3)

    # for s_result in subject_result_files:
    s_result = ds['s_result']

    contact_result_fname = os.path.join(contact_results_dir,mosh_subject_id,contact_expr_id,os.path.basename(s_result).replace('_poses','_contact'))
    contact_video_fname = os.path.join(contact_results_dir,mosh_subject_id,contact_expr_id+'_videos')
    contact_video_file = os.path.join(contact_video_fname,os.path.basename(contact_result_fname))


    if os.path.exists(contact_result_fname):
        print 'contact results exist for file %s' % s_result

        if visualize_contact :#and (not os.path.exists(contact_video_file+'.mp4')):

            o_result = os.path.join(object_results_dir, os.path.basename(s_result))
            o_name = os.path.basename(s_result).split('_')[0]
            o_mesh = os.path.join(object_mesh_dir, '%s.ply' % o_name)
            # supporting Table for objects
            T_result = os.path.join(Table_results_dir, os.path.basename(s_result))
            T_mesh = os.path.join(object_mesh_dir, 'Table.ply')

            with open(s_result, 'rb') as f:
                s_mosh_result = pickle.load(f)
            with open(o_result, 'rb') as f:
                o_mosh_result = pickle.load(f)
            # with open(T_result, 'rb') as f:
            #     T_mosh_result = pickle.load(f)

            o_model = RigidObjectModel(model_plypath=o_mesh)
            # T_model = RigidObjectModel(model_plypath=T_mesh)

            body_pose = s_mosh_result['pose_est_poses']
            body_trans = s_mosh_result['pose_est_trans']

            object_pose = o_mosh_result['pose_est_poses']
            object_trans = o_mosh_result['pose_est_trans']

            # Table_pose = T_mosh_result['pose_est_poses']
            # Table_trans = T_mosh_result['pose_est_trans']

            with open(contact_result_fname, 'rb') as f:
                contact_result = pickle.load(f)
            contact_vertices_body = contact_result['body_contact_vertices']
            contact_vertices_object = contact_result['object_contact_vertices']

            if show_markers:
                obs_mrk = np.concatenate([s_mosh_result['pose_est_obmrks'], o_mosh_result['pose_est_obmrks']],
                                         axis=1)
                simul_mrk = np.concatenate(
                    [s_mosh_result['pose_est_simmrks'], o_mosh_result['pose_est_simmrks']], axis=1)



            if save_video:
                movie_name = os.path.basename(contact_result_fname)
                tmp_file = makepath(os.path.join(contact_video_fname,'tmp_%s'%(movie_name.split('.')[0])))

            last_frame = body_pose.shape[0]
            last_frame = body_pose.shape[0]/4

            for fCntr in range(last_frame):
                # fCntr +=335
                fCntr =fCntr*4
                print 'frame %4d / %4d' % (fCntr/4, last_frame)
                smplx.pose[:] = body_pose[fCntr]
                # smplx.pose[90:102] = body_pose[fCntr][90:102]
                # smplx.pose[:3] = body_pose[fCntr][:3]
                # smplx.pose[63:66] = body_pose[fCntr][63:66]
                # smplx.pose[57:60] = body_pose[fCntr][57:60]
                smplx.trans[:] = body_trans[fCntr]
                body_mesh = Mesh(v=smplx.r, f=smplx.f, vc=name_to_rgb['white']).rotate_vertices(Rxyz)

                o_model.pose[:] = object_pose[fCntr]
                o_model.trans[:] = object_trans[fCntr]
                object_mesh = Mesh(v=o_model.r, f=o_model.f, vc=name_to_rgb['yellow']).rotate_vertices(Rxyz)

                # T_model.pose[:] = Table_pose[fCntr]
                # T_model.trans[:] = Table_trans[fCntr]
                # Table_mesh = Mesh(v=T_model.r, f=T_model.f, vc=name_to_rgb['blue']).rotate_vertices(Rxyz)

                body_first_contact = np.where(contact_vertices_body[fCntr])[0]
                object_first_contact = np.where(contact_vertices_object[fCntr])[0]

                # body_first_contact = body_mesh.set_vertex_colors(vc='red', vertex_indices=contact_vertices_body[fCntr])
                # object_first_contact = object_mesh.set_vertex_colors(vc='red',
                #                                                      vertex_indices=contact_vertices_object[fCntr])

                body_first_contact = body_mesh.set_vertex_colors(vc='red', vertex_indices=body_first_contact)
                object_first_contact = object_mesh.set_vertex_colors(vc='red',
                                                                     vertex_indices=object_first_contact)



                if show_markers:
                    nonan_ids = ~np.isnan(obs_mrk)

                    ob_nonan = obs_mrk[nonan_ids].reshape(-1, 3)
                    sm_nonan = simul_mrk[nonan_ids].reshape(-1, 3)

                    # ob_nonan = ob_nonan if Rxyz is None else rotateXYZ(ob_nonan, Rxyz)
                    # sm_nonan = sm_nonan if Rxyz is None else rotateXYZ(sm_nonan, Rxyz)

                    obmrks = spheres_for(ob_nonan, radius=mrk_radius, vc=name_to_rgb['blue'])
                    simrks = spheres_for(sm_nonan, radius=mrk_radius, vc=name_to_rgb['red'])


                ####### save for Joachim #########
                base_dir = '/ps/project/COBRA_RENDER'
                seq_dir = os.path.join(base_dir,os.path.basename(contact_result_fname).replace('_stageII.pkl',''))
                makepath(seq_dir)
                body_first_contact.write_obj(os.path.join(seq_dir,'%04d_body.obj'%(fCntr/4)))
                object_first_contact.write_obj(os.path.join(seq_dir,'%04d_object.obj'%(fCntr/4)))

                ##################################

                # mvs[0][0].set_static_meshes([body_first_contact] + [object_first_contact] , blocking=True)
                # # mvs[0][0].set_static_meshes([body_first_contact]  , blocking=True)
                # mvs[0][1].set_static_meshes([object_first_contact], blocking=True)
                if save_video:
                    mvs[0][0].titlebar = 'frame %d/%d - %s'%((fCntr+1), last_frame, movie_name)
                    mvs[0][0].save_snapshot(tmp_file+'/%05d.png'%(fCntr+1))

            if save_video:
                create_video(tmp_file,name=movie_name)

        print 'skipping the file'
    else:
        print 'proccessing contact results for file %s' % s_result
        makepath(contact_result_fname, isfile=True)

        # res = {'hi': 5}
        # pickle.dump(res, open(contact_result_fname, 'w'), -1)

        o_result = os.path.join(object_results_dir, os.path.basename(s_result))
        o_name = os.path.basename(s_result).split('_')[0] #if not table else 'Table'
        o_mesh = os.path.join(object_mesh_dir, '%s.ply' % o_name)

        with open(s_result, 'rb') as f:
            s_mosh_result = pickle.load(f)
        with open(o_result, 'rb') as f:
            o_mosh_result = pickle.load(f)

        o_model = RigidObjectModel(model_plypath=o_mesh)

        body_pose = s_mosh_result['pose_est_poses']
        body_trans = s_mosh_result['pose_est_trans']

        object_pose = o_mosh_result['pose_est_poses']
        object_trans = o_mosh_result['pose_est_trans']

        contact_vertices_body = []
        contact_vertices_object = []

        last_frame = body_pose.shape[0]

        for fCntr in range(last_frame):
            # fCntr +=999
            print 'frame %4d / %4d' % (fCntr, last_frame)
            smplx.pose[:] = body_pose[fCntr]
            smplx.trans[:] = body_trans[fCntr]
            body_mesh = Mesh(v=smplx.r, f=smplx.f, vc=name_to_rgb['pink'])
            o_model.pose[:] = object_pose[fCntr]
            o_model.trans[:] = object_trans[fCntr]
            object_mesh = Mesh(v=o_model.r, f=o_model.f, vc=name_to_rgb['pink'])

            # missing_markers = (marker_data[frame_counter]==marker_data[frame_counter])[:,0]
            # mocap_markers = spheres_for(marker_data[frame_counter][missing_markers].numpy(), radius=0.003,  vc=np.array((0., 0., 1.)))
            # body, object = point2point(body_mesh,object_mesh,contact_threshold=1e-5)

            body_first_contact, object_first_contact = get_first_contact_surface(body_mesh, object_mesh, contact_threshold=contact_threshold)

            if visualize_contact:

                # labels_color = []
                # for i in range(751, 0, -752 / 55):
                #     labels_color.append(name_to_rgb.values()[i])
                #
                # unique_labels = np.unique(body_first_contact)
                # if unique_labels.size>1:
                #     for item in unique_labels[1:]:
                #         body_contact = np.where(body_first_contact==item)[0]
                #         object_contact = np.where(object_first_contact==item)[0]
                #
                #         body_mesh.set_vertex_colors(vc=labels_color[item], vertex_indices=body_contact)
                #         object_mesh.set_vertex_colors(vc=labels_color[item], vertex_indices=object_contact)

                body_contact = np.where(body_first_contact)[0]
                object_contact = np.where(object_first_contact)[0]

                body_mesh.set_vertex_colors(vc=name_to_rgb['red'], vertex_indices=body_contact)
                object_mesh.set_vertex_colors(vc=name_to_rgb['red'], vertex_indices=object_contact)

                mvs[0][0].set_static_meshes([body_mesh] + [object_mesh], blocking=True)
                mvs[0][1].set_static_meshes([object_mesh], blocking=True)
                if save_video:
                    mvs[0][0].save_snapshot(
                        '/ps/scratch/body_hand_object_contact/arxiv/snapshots/tmp/%05d.png' % fCntr)

            contact_vertices_body.append(body_first_contact)
            contact_vertices_object.append(object_first_contact)

        result = {'v_template': v_template_file,
                  'body_model': model_file,
                  'object_mesh': o_mesh,
                  'subject_mosh_file': s_result,
                  'object_mosh_file': o_result,
                  'body_contact_vertices': contact_vertices_body,
                  'object_contact_vertices': contact_vertices_object,
                  'contact_threshold': contact_threshold}

        pickle.dump(result, open(contact_result_fname, 'w'), -1)

    print 'end'
def vis_contact():
    pass


def main():

    mosh_expr_id = '05'
    contact_expr_id = '05'
    mosh_subject_id = '190920_00174'
    gender = 'male'

    mrk_radius = 0.0032
    show_markers = False
    return_mesh = False
    save_video = False
    contact_threshold = 1e-5
    visualize_contact = True

    mosh_results_dir = '/ps/scratch/body_hand_object_contact/mosh_results/%s' % mosh_expr_id
    contact_results_dir = '/ps/scratch/body_hand_object_contact/contact_results/%s' % mosh_expr_id


    subject_results_dir = os.path.join(mosh_results_dir,'mosh_subject', mosh_subject_id)
    object_results_dir = os.path.join(mosh_results_dir,'mosh_object', mosh_subject_id)
    object_mesh_dir = '/ps/scratch/body_hand_object_contact/data/object_meshes/contact_meshes'


    model_file = os.path.join('/ps/scratch/common/moshpp/smplx/unlocked_head',gender,'model.pkl')
    v_template_file = os.path.join('/ps/scratch/body_hand_object_contact/data/subject_meshes',gender,'%s.ply'%mosh_subject_id)

    v_template = Mesh(filename=v_template_file).v
    smplx = load_smplx(fname_or_dict=model_file, ncomps= 24, v_template=v_template)

    subject_result_files = glob.glob(os.path.join(subject_results_dir,'*_poses.pkl'))

    mvs = MeshViewers(window_width=2400, window_height=1600, shape=[1, 3])



    for s_result in subject_result_files:

        # for i, item in enumerate(subject_result_files):
        #     if 'binoculars' in item:
        #         print i
        #         print item
        # s_result = subject_result_files[58]


        contact_result_fname = os.path.join(contact_results_dir,mosh_subject_id,contact_expr_id,os.path.basename(s_result).replace('_poses','_contact'))

        if os.path.exists(contact_result_fname):
            print 'contact results exist for file %s'%s_result

            if visualize_contact:

                o_result = os.path.join(object_results_dir, os.path.basename(s_result))

                o_name = os.path.basename(s_result).split('_')[0]
                o_mesh = os.path.join(object_mesh_dir, '%s.ply' % o_name)


                with open(s_result, 'rb') as f:
                    s_mosh_result = pickle.load(f)
                with open(o_result, 'rb') as f:
                    o_mosh_result = pickle.load(f)

                o_model = RigidObjectModel(model_plypath=o_mesh)


                body_pose = s_mosh_result['pose_est_poses']
                body_trans = s_mosh_result['pose_est_trans']

                object_pose = o_mosh_result['pose_est_poses']
                object_trans = o_mosh_result['pose_est_trans']

                with open(contact_result_fname, 'rb') as f:
                    contact_result = pickle.load(f)
                contact_vertices_body = contact_result['body_contact_vertices']
                contact_vertices_object = contact_result['object_contact_vertices']

                contact_result_fname1='/ps/scratch/body_hand_object_contact/contact_results/05/190920_00174/02/cup_waterbottle_pour_into_cup_drink_contact.pkl'
                o_result1 = '/ps/scratch/body_hand_object_contact/mosh_results/05/mosh_object/190920_00174/cup_waterbottle_pour_into_cup_drink_poses.pkl'
                o_mesh1 = '/ps/scratch/body_hand_object_contact/data/object_meshes/final_scaled_and_cleaned_meshes/cup.ply'

                o_model1 = RigidObjectModel(model_plypath=o_mesh1)
                with open(o_result1, 'rb') as f:
                    o_mosh_result1 = pickle.load(f)

                object_pose1 = o_mosh_result1['pose_est_poses']
                object_trans1 = o_mosh_result1['pose_est_trans']

                with open(contact_result_fname1, 'rb') as f:
                    contact_result1 = pickle.load(f)
                contact_vertices_body1 = contact_result1['body_contact_vertices']
                contact_vertices_object1 = contact_result1['object_contact_vertices']

                # contact_vertices_body = np.concatenate([contact_vertices_body1,contact_vertices_body])




                if show_markers:
                    obs_mrk = np.concatenate([s_mosh_result['pose_est_obmrks'], o_mosh_result['pose_est_obmrks']],
                                             axis=1)
                    simul_mrk = np.concatenate(
                        [s_mosh_result['pose_est_simmrks'], o_mosh_result['pose_est_simmrks']], axis=1)

                last_frame = body_pose.shape[0]

                for fCntr in range(last_frame):
                    # fCntr +=335
                    print 'frame %4d / %4d' % (fCntr, last_frame)
                    smplx.pose[:] = body_pose[fCntr]
                    smplx.trans[:] = body_trans[fCntr]
                    body_mesh = Mesh(v=smplx.r, f=smplx.f, vc=name_to_rgb['pink'])
                    o_model.pose[:] = object_pose[fCntr]
                    o_model.trans[:] = object_trans[fCntr]
                    object_mesh = Mesh(v=o_model.r, f=o_model.f, vc=name_to_rgb['pink'])

                    o_model1.pose[:] = object_pose1[fCntr]
                    o_model1.trans[:] = object_trans1[fCntr]
                    object_mesh1 = Mesh(v=o_model1.r, f=o_model1.f, vc=name_to_rgb['pink'])

                    body_first_contact = body_mesh.set_vertex_colors(vc='red', vertex_indices=np.concatenate([contact_vertices_body[fCntr],contact_vertices_body1[fCntr]]))
                    object_first_contact = object_mesh.set_vertex_colors(vc='red',
                                                                         vertex_indices=contact_vertices_object[fCntr])

                    object_first_contact1 = object_mesh1.set_vertex_colors(vc='red',
                                                                         vertex_indices=contact_vertices_object1[fCntr])



                    if show_markers:
                        nonan_ids = ~np.isnan(obs_mrk)

                        ob_nonan = obs_mrk[nonan_ids].reshape(-1, 3)
                        sm_nonan = simul_mrk[nonan_ids].reshape(-1, 3)

                        # ob_nonan = ob_nonan if Rxyz is None else rotateXYZ(ob_nonan, Rxyz)
                        # sm_nonan = sm_nonan if Rxyz is None else rotateXYZ(sm_nonan, Rxyz)

                        obmrks = spheres_for(ob_nonan, radius=mrk_radius, vc=name_to_rgb['blue'])
                        simrks = spheres_for(sm_nonan, radius=mrk_radius, vc=name_to_rgb['red'])





                    mvs[0][0].set_static_meshes([body_first_contact] + [object_first_contact]+[object_first_contact1], blocking=True)
                    mvs[0][1].set_static_meshes([object_first_contact], blocking=True)
                    mvs[0][2].set_static_meshes([object_first_contact1], blocking=True)

                    # rotation_xyz = (90, 0, 0)
                    # r = euler(rotation_xyz)
                    # view1 = Mesh(v=object_first_contact.rotate_vertices(r).v, f=object_first_contact.f)
                    #
                    # mvs[0][2].set_static_meshes([view1], blocking=True)
                    #
                    # rotation_xyz = (0, 0, 90)
                    # r = euler(rotation_xyz)
                    # view2 = Mesh(v=object_first_contact.rotate_vertices(r).v, f=object_first_contact.f)
                    # mvs[0][3].set_static_meshes([view2], blocking=True)

                    if save_video:
                        mvs[0][0].save_snapshot('/ps/scratch/body_hand_object_contact/arxiv/snapshots/tmp/%05d.png' % fCntr)

                if save_video: create_video('/ps/scratch/body_hand_object_contact/arxiv/snapshots/tmp',
                                            name=os.path.basename(s_result))

            print 'skipping the file'
        else:
            print 'contact results for file %s'%s_result
            makepath(contact_result_fname,isfile=True)

            res = {'hi': 5}
            pickle.dump(res, open(contact_result_fname, 'w'), -1)

            o_result = os.path.join(object_results_dir, os.path.basename(s_result))
            o_name = os.path.basename(s_result).split('_')[0]
            o_mesh = os.path.join(object_mesh_dir, '%s.ply'%o_name)

            with open(s_result, 'rb') as f:
                s_mosh_result= pickle.load(f)
            with open(o_result, 'rb') as f:
                o_mosh_result= pickle.load(f)

            o_model = RigidObjectModel(model_plypath=o_mesh)

            body_pose = s_mosh_result['pose_est_poses']
            body_trans = s_mosh_result['pose_est_trans']

            object_pose = o_mosh_result['pose_est_poses']
            object_trans = o_mosh_result['pose_est_trans']

            contact_vertices_body = []
            contact_vertices_object = []



            last_frame = body_pose.shape[0]

            for fCntr in range(last_frame):
                # fCntr +=335
                print 'frame %4d / %4d'%(fCntr,last_frame)
                smplx.pose[:] = body_pose[fCntr]
                smplx.trans[:] = body_trans[fCntr]
                body_mesh = Mesh(v=smplx.r,f=smplx.f, vc=name_to_rgb['pink'])
                o_model.pose[:] = object_pose[fCntr]
                o_model.trans[:] = object_trans[fCntr]
                object_mesh = Mesh(v=o_model.r,f=o_model.f, vc=name_to_rgb['pink'])

                # missing_markers = (marker_data[frame_counter]==marker_data[frame_counter])[:,0]
                # mocap_markers = spheres_for(marker_data[frame_counter][missing_markers].numpy(), radius=0.003,  vc=np.array((0., 0., 1.)))
                # body, object = point2point(body_mesh,object_mesh,contact_threshold=1e-5)




                body_first_contact, object_first_contact = get_first_contact_surface(body_mesh,object_mesh,contact_threshold=1e-5)

                if visualize_contact:



                    body_first_contact = body_mesh.set_vertex_colors(vc='red',
                                                                     vertex_indices=body_first_contact)
                    object_first_contact = object_mesh.set_vertex_colors(vc='red',
                                                                         vertex_indices=object_first_contact)

                    mvs[0][0].set_static_meshes([body_first_contact] + [object_first_contact], blocking=True)
                    mvs[0][1].set_static_meshes([object_first_contact], blocking=True)
                    if save_video:
                        mvs[0][0].save_snapshot(
                            '/ps/scratch/body_hand_object_contact/arxiv/snapshots/tmp/%05d.png' % fCntr)

                contact_vertices_body.append(body_first_contact)
                contact_vertices_object.append(object_first_contact)

            result = {'v_template':v_template_file,
                      'body_model':model_file,
                      'object_mesh':o_mesh,
                      'subject_mosh_file':s_result,
                      'object_mosh_file':o_result,
                      'body_contact_vertices':contact_vertices_body,
                      'object_contact_vertices':contact_vertices_object,
                      'contact_threshold':contact_threshold}


            pickle.dump(result, open(contact_result_fname, 'w'), -1)


        print 'end'

if __name__ == '__main__':
    run_parallel_contact()
    # main()


    ########## to save mocap file names to a text file ###########
    # subject_results_dir = '/ps/scratch/vicondata/ViconDataCaptures/OfficialCaptures/HandsObjectContact/HandsObjectContact_191001_00158_ML/VDF_MCP_X2D'
    # my_list = glob.glob(os.path.join(subject_results_dir, '*.mcp'))
    # my_list = [name for name in sorted(my_list)]
    # with open('/home/otaheri/Desktop/seq_names.txt', 'w') as f:
    #     for item in my_list:
    #         f.write("%s\n" % os.path.basename(item).replace('.mcp', ''))




