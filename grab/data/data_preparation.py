
import glob
import os
import shutil
import sys
from datetime import datetime
from tqdm import tqdm
import numpy as np

import torch

from human_body_prior.tools.omni_tools import makepath, log2file

from human_body_prior.body_model.body_model import BodyModel
from grab.tools.object_model import ObjectModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from psbody.mesh import Mesh
from human_body_prior.tools.omni_tools import colors
from grab.tools.vis_tools import colors, points_to_spheres

from basis_point_sets.bps import generate_bps, convert_to_bps, reconstruct_from_bps
from basis_point_sets.bps_torch import convert_to_bps_batch, convert_to_bps_chamfer

from psbody.mesh import Mesh, MeshViewer
import pickle
from smplx.lbs import batch_rodrigues
from human_body_prior.train.vposer_smpl import VPoser  # matrot2aa
from grab.tools.rotations import rotateXYZ, local2global_pose, rotateZ


def get_object_names(contact_results_dir):
    object_names = []
    contact_fnames = glob.glob(os.path.join(contact_results_dir, '*/*_stageII.pkl'))
    for contact_fname in contact_fnames:
        object_name = os.path.basename(contact_fname).split('_')[0]
        object_names.append(object_name)
    return list(set(object_names))

def get_frame_contact_mask(data_c, data_o, include_joints=range(41, 56), exclude_joints = range(26, 41)):
    in_contact_frames = []
    table_height = data_o['pose_est_trans'][0, 2]
    for fId, fdata in enumerate(data_c['object_contact_vertices']):
        # if item.size!=0: # for contact without the labels
        if np.unique(fdata).size > 1:  # for labeled contacts
            if (data_o['pose_est_trans'][fId, 2] > table_height + .002) or (data_o['pose_est_trans'][fId, 2] < table_height - .002):
                data_mask = np.logical_and(fdata != 0, np.isin(fdata, include_joints))
                for j in include_joints:
                    if (fdata == j).sum() < 50:
                        data_mask[fdata == j] = False

                if np.isin(fdata, include_joints).any() and not np.isin(fdata, exclude_joints).any() and len(list(set(fdata[data_mask])))>1:

                    # skipping frames where less than two include_joints are in contact
                    # print(list(set(fdata[np.logical_and(fdata != 0 , np.isin(fdata, include_joints))])))
                    # it might happen that a joint which is not in the included joints is also in contact

                    in_contact_frames.append(True)
                else:
                    in_contact_frames.append(False)
            else:
                in_contact_frames.append(False)
        else:
            in_contact_frames.append(False)
    in_contact_frames = np.asarray(in_contact_frames)
    # print('Contact labels:', set((np.asarray(data_c['object_contact_vertices'])[in_contact_frames].reshape(-1)).tolist()))
    return in_contact_frames

def load_object_verts(object_info_path, max_num_object_verts = 10000):
    if os.path.exists(object_info_path):
        with open(object_info_path, 'rb') as f:
            object_infos = pickle.load(f)
    else:
        object_infos = {}

    def get_verts(object_name, data_c=None, subsample = True):
        if object_name not in object_infos:
            np.random.seed(100)  # this makes it possible to reproduce the sample ids
            object_mesh = Mesh(filename=data_c['object_mesh'])
            verts_object = object_mesh.v
            object_verts_dsampl_ids = np.random.choice(verts_object.shape[0], max_num_object_verts, replace=False)
            object_verts_dsampl = verts_object[object_verts_dsampl_ids]
            object_infos[object_name] = {'verts_object': verts_object,
                                         'faces_object': object_mesh.f,
                                         'object_verts_dsampl_ids': object_verts_dsampl_ids,
                                         'object_verts_dsampl': object_verts_dsampl}
        else:
            object_verts_dsampl_ids = object_infos[object_name]['object_verts_dsampl_ids']
            object_verts_dsampl = object_infos[object_name]['object_verts_dsampl']

        get_verts.object_infos = object_infos

        return object_verts_dsampl, object_verts_dsampl_ids

    get_verts.object_infos = object_infos
    return get_verts

def prepare_grab_dataset(data_workdir, contacts_dir, object_splits, logger=None):
    BPS_N_POINTS = 1024
    FPS_DS_RATE = 1
    MAX_NUM_OBJECT_VERTS = 5000 # equal to the number of hand verts
    NUM_AUGMENTATION = 1

    include_joints = list(range(41, 56))
    exclude_joints = list(range(26, 41))

    bps_basis_fname = makepath(os.path.join(data_workdir, 'bps_basis_%d.npz'%BPS_N_POINTS), isfile=True)
    if os.path.exists(bps_basis_fname):
        basis = np.load(bps_basis_fname)['basis']
    else:
        basis = generate_bps(n_points=BPS_N_POINTS, radius=0.18, n_dims=3, kd_ordering=True)
        np.savez(bps_basis_fname, basis=basis)

    part2vids = np.load('/ps/scratch/body_hand_object_contact/grab_net/part2vids.npz')

    hand_vids = part2vids['handR']
    hand_model_path = '/ps/scratch/body_hand_object_contact/body_models/models/models_unchumpy/MANO_RIGHT.npz'
    mano_smplxhand_offset = torch.from_numpy(np.array([0.0957, 0.0064, 0.0062]).reshape(1,-1).astype(np.float32))

    starttime = datetime.now().replace(microsecond=0)

    if logger is None:
        logger = log2file()
        logger('Starting data preparation at %s'%datetime.strftime(starttime, '%Y%m%d_%H%M'))

    shutil.copy2(sys.argv[0], os.path.join(data_workdir, os.path.basename(sys.argv[0]).replace('.py', '_%s.py' % datetime.strftime(starttime, '%Y%m%d_%H%M'))))

    logger('Starting with data preparation')
    logger('Num objects per datasplit: %s'%(', '.join(['%s: %d'%(k, len(v)) for k,v in object_splits.items()])))
    logger('Will augment train split %d times' % (NUM_AUGMENTATION))

    object_info_path = makepath(os.path.join(data_workdir, 'object_infos.pkl'), isfile=True)
    object_loader = load_object_verts(object_info_path, max_num_object_verts=MAX_NUM_OBJECT_VERTS)

    subject_infos = {}

    for split_name in object_splits.keys():
        outfname = makepath(os.path.join(data_workdir, split_name, 'delta_hand_mano.pt'), isfile=True)

        if os.path.exists(outfname):
            logger('Results for %s split already exist.'%(split_name))
            continue
        else:
            logger('Processing data for %s split.'%(split_name))


        contact_fnames = glob.glob(os.path.join(contacts_dir, '*/*_stageII.pkl'))

        out_data = {'verts_hand_mano':[], 'verts_object':[], 'delta_object':[], 'delta_hand_mano':[], 'root_orient':[], 'pose_hand':[], 'trans':[], 'trans_object': [], 'root_orient_object':[]}
        frame_names = []
        # contact_maps_fnames = []
        for contact_fname in contact_fnames:
            object_name = os.path.basename(contact_fname).split('_')[0]
            if object_name not in object_splits[split_name]: continue
            subject_name = contact_fname.split('/')[-2]
            # action_name = '_'.join(os.path.basename(contact_fname).split('_')[1:-1])
            # action_names.append(action_name)
            with open(contact_fname, 'rb') as f: data_c = pickle.load(f, encoding='latin1')
            with open(data_c['subject_mosh_file'], 'rb') as f: data_s = pickle.load(f, encoding='latin1')
            with open(data_c['object_mosh_file'], 'rb') as f: data_o = pickle.load(f, encoding='latin1')

            frame_mask = get_frame_contact_mask(data_c, data_o, include_joints, exclude_joints)
            # frame_mask[::FPS_DS_RATE] = False
            
            T = frame_mask.sum()
            if T<1:continue

            verts_object, _ = object_loader(object_name, data_c)

            if subject_name in subject_infos:
                v_template_s = subject_infos[split_name]
            else:
                v_template_s = Mesh(filename=data_s['vtemplate_fname']).v
                subject_infos[split_name] = v_template_s

            bm = BodyModel(bm_path=data_s['shape_est_templatemodel'].replace('.pkl', '.npz'), batch_size=T, v_template=v_template_s)
            bm_mano = BodyModel(bm_path=hand_model_path, batch_size=T)#, v_template=v_template_s[hand_vids])
            om = ObjectModel(v_template=verts_object, batch_size=T)

            object_parms = {'trans':data_o['pose_est_trans'][frame_mask], 'root_orient':data_o['pose_est_poses'][frame_mask]}

            object_parms = {k: torch.from_numpy(v).type(torch.float32) for k, v in object_parms.items()}

            verts_object = c2c(om(**object_parms).v)
            bps_offset = verts_object.mean(1)[:, None] # center both wrt to object

            body_fullpose = data_s['pose_est_fullposes'][frame_mask]
            body_params = {'root_orient': body_fullpose[:, :3],
                           'pose_body': body_fullpose[:, 3:66],
                           'pose_hand': body_fullpose[:, 75:],
                           'trans': data_s['pose_est_trans'][frame_mask],
                           }
            body_params = {k: torch.from_numpy(v).type(torch.float32) for k, v in body_params.items()}
            bm_eval = bm(**body_params)
            # verts_hand_smplx = c2c(bm_eval.v[:,hand_vids])

            ## get MANO hand params from smplx params
            #######compute global orientation of the hand
            fullpose = torch.from_numpy(body_fullpose).type(torch.float32)
            matrots = batch_rodrigues(fullpose.view(-1, 3)).view(T, -1, 3, 3)
            global_matrots = local2global_pose(matrots).view(T, 1, -1, 9)
            local_aa = VPoser.matrot2aa(global_matrots)
            rootorient_HR = local_aa[:, 0, 21]
            ######################### compute mano hand
            pose_HR = fullpose[:, 120:]
            trans_HR = bm_eval.Jtr[:, 21] - mano_smplxhand_offset
            bm_mano_eval = bm_mano(root_orient=rootorient_HR, pose_hand=pose_HR, trans=trans_HR)
            verts_hand_mano = c2c(bm_mano_eval.v)
            # ##########################
            
            # verts_object_hand_mano = np.concatenate([verts_object, verts_hand_mano], axis=1)

            verts_object -= bps_offset
            verts_hand_mano -= bps_offset
            trans_HR = c2c(trans_HR) - bps_offset[:,0]
            trans_OBJ = c2c(object_parms['trans']) - bps_offset[:,0]
            rootorient_OBJ = c2c(object_parms['root_orient'])

            # verts_hand_smplx -= bps_offset
            # verts_object_hand_mano -= bps_offset

            sys.path.append('/is/ps2/nghorbani/code-repos/basis_point_sets/chamfer-extension')

            _, delta_object = convert_to_bps_chamfer(verts_object, basis, return_deltas=True)
            _, delta_hand_mano = convert_to_bps_chamfer(verts_hand_mano, basis, return_deltas=True)

            out_data['verts_hand_mano'].append(verts_hand_mano)
            out_data['verts_object'].append(verts_object[:,np.random.choice(verts_object.shape[1], 500, replace=False)])
            out_data['delta_object'].append(delta_object)
            out_data['delta_hand_mano'].append(delta_hand_mano)
            out_data['root_orient'].append(rootorient_HR)
            out_data['pose_hand'].append(pose_HR)
            out_data['trans'].append(trans_HR)
            out_data['trans_object'].append(trans_OBJ)
            out_data['root_orient_object'].append(rootorient_OBJ)

            frame_names.extend(['%s_%s'%(contact_fname, fId) for fId in np.arange(len(data_o['pose_est_trans']))[frame_mask]])

            # #################
            # mv = MeshViewer(keepalive=False)
            # id = 0
            # obj_mesh = points_to_spheres(verts_object[id], radius=0.001, color=colors['blue'])
            #
            # bm_mano_eval = bm_mano(root_orient=rootorient_HR, pose_hand=pose_HR, trans=pose_HR.new(trans_HR))
            # verts_hand_mano = c2c(bm_mano_eval.v)
            #
            # hand_mano_mesh = Mesh(verts_hand_mano[id], c2c(bm_mano_eval.f), vc=colors['red'])
            # trans_HR_mesh = points_to_spheres(trans_HR[id:id+1]+ c2c(mano_smplxhand_offset), radius=0.01, color=colors['red'])
            #
            # mv.set_static_meshes([obj_mesh, hand_mano_mesh])
            # print
            # # mv.set_dynamic_meshes([])

            # # ################################
            # if split_name == 'train':
            #     # mv = MeshViewer(keepalive=False)
            #     # id = 0
            #     for _ in range(NUM_AUGMENTATION):
            #         rotdeg_z = np.random.random([T])*360
            #         trans_HR_rotated, rotmat_z = rotateZ(trans_HR + c2c(mano_smplxhand_offset), rotdeg_z)
            #         trans_HR_rotated = trans_HR_rotated - c2c(mano_smplxhand_offset)
            # 
            #         trans_OBJ_rotated, _ = rotateZ(trans_OBJ , rotdeg_z)
            # 
            #         verts_object_rotated, _ = rotateZ(verts_object, rotdeg_z)
            #         verts_hand_mano_rotated, _ = rotateZ(verts_hand_mano, rotdeg_z)
            # 
            #         ################# rotate root orient of hand
            # 
            #         rootorient_HR_matrot = VPoser.aa2matrot(rootorient_HR.view(T,1,-1,3)).view(T,3,3)#Nx1xnum_jointsx3
            #         rootorient_HR_matrot_rotated = torch.matmul(torch.from_numpy(rotmat_z.astype(np.float32)),rootorient_HR_matrot)
            #         rootorient_HR_rotated = VPoser.matrot2aa(rootorient_HR_matrot_rotated.view(T,1,-1,9)).view(T,3)
            # 
            #         _, delta_object_rotated = convert_to_bps_chamfer(verts_object_rotated, basis, return_deltas=True)
            #         _, delta_hand_mano_rotated = convert_to_bps_chamfer(verts_hand_mano_rotated, basis, return_deltas=True)
            # 
            # 
            #         ################ rotate root orient of object
            # 
            #         rootorient_OBJ_matrot = VPoser.aa2matrot(pose_HR.new(rootorient_OBJ).view(T,1,-1,3)).view(T,3,3)#Nx1xnum_jointsx3
            #         rootorient_OBJ_matrot_rotated = torch.matmul(torch.from_numpy(rotmat_z.astype(np.float32)),rootorient_OBJ_matrot)
            #         rootorient_OBJ_rotated = VPoser.matrot2aa(rootorient_OBJ_matrot_rotated.view(T,1,-1,9)).view(T,3)
            # 
            #         # bm_mano_eval_rotated = bm_mano(root_orient=rootorient_HR_rotated, pose_hand=pose_HR, trans=pose_HR.new(trans_HR_rotated))
            #         #
            #         # obj_mesh = points_to_spheres(verts_object[id], radius=0.001, color=colors['blue'])
            #         # hand_mano_mesh = Mesh(verts_hand_mano[id], c2c(bm_mano_eval.f), vc=colors['red'])
            #         # trans_HR_mesh = points_to_spheres(trans_HR[id:id+1]+ c2c(mano_smplxhand_offset), radius=0.01, color=colors['red'])
            #         # trans_HR_rotated_mesh = points_to_spheres(trans_HR_rotated[id:id+1]+ c2c(mano_smplxhand_offset), radius=0.01, color=colors['green'])
            #         #
            #         # obj_mesh_rotated = points_to_spheres(verts_object_rotated[id], radius=0.001, color=colors['blue'])
            #         # hand_mano_mesh_rotated = Mesh(verts_hand_mano_rotated[id], c2c(bm_mano_eval.f), vc=colors['red'])
            #         # hand_mano_mesh_rotated2 = Mesh(c2c(bm_mano_eval_rotated.v[id]), c2c(bm_mano_eval.f), vc=colors['green'])
            #         #
            #         # mv.set_static_meshes([obj_mesh, hand_mano_mesh, trans_HR_mesh])
            #         # mv.set_dynamic_meshes([obj_mesh_rotated, hand_mano_mesh_rotated, hand_mano_mesh_rotated2, trans_HR_rotated_mesh])
            #         # print()
            # 
            #         out_data['verts_hand_mano'].append(verts_hand_mano_rotated)
            #         out_data['verts_object'].append(verts_object_rotated[:, np.random.choice(verts_object.shape[1], 500, replace=False)])
            #         out_data['delta_object'].append(delta_object_rotated)
            #         out_data['delta_hand_mano'].append(delta_hand_mano_rotated)
            #         out_data['root_orient'].append(rootorient_HR_rotated)
            #         out_data['pose_hand'].append(pose_HR)
            #         out_data['trans'].append(trans_HR_rotated)
            #         out_data['trans_object'].append(trans_OBJ_rotated)
            #         out_data['root_orient_object'].append(rootorient_OBJ_rotated)
            # 
            #         frame_names.extend(['%s_rotated_%s' % (contact_fname, fId) for fId in np.arange(len(data_o['pose_est_trans']))[frame_mask]])

        for k,v in out_data.items():

            outfname = makepath(os.path.join(data_workdir, split_name, '%s.pt' % k), isfile=True)
            if os.path.exists(outfname): continue
            out_data[k] = torch.from_numpy(np.concatenate(v))
            torch.save(out_data[k], outfname)

        np.savez(os.path.join(data_workdir, split_name, 'frame_names.npz'), frame_names=frame_names)

        logger('%d datapoints for %s'%(out_data['delta_object'].shape[0], split_name))

    with open(object_info_path, 'wb') as f:
        pickle.dump(object_loader.object_infos, f, protocol=2)

    # action_names = list(set(action_names))
    # print(len(action_names), action_names)


if __name__ == '__main__':

    msg = 'Replaced SMPLx hand with MANO hand in bps representation\n'
    msg += 'Include mano hand right parameters\n'
    msg += '1X data augmentation\n'
    msg += '1X down sampling\n'
    msg += 'Added frame information only.\n'
    msg += '\n'

    contacts_dir = '/ps/scratch/body_hand_object_contact/contact_results/17/03_thrshld_50e_6_final'
    # contacts_dir = '/ps/scratch/body_hand_object_contact/contact_results/16_omid/02_thrshld_20e_6_final'
    # contacts_dir = '/ps/scratch/body_hand_object_contact/contact_results/16_01_thrshld_15e_6_final'
    object_names = get_object_names(contacts_dir)

    object_splits = {
        'test': ['mug', 'wineglass','camera', 'binoculars', 'fryingpan'],
        'vald': ['apple', 'toothpaste', 'elephant', 'hand']
    }
    object_splits['train'] = list(set(object_names).difference(set(object_splits['test'] + object_splits['vald'])))

    expr_code = 'V01_11_00'

    data_workdir = os.path.join('/ps/scratch/body_hand_object_contact/grab_net/data', expr_code)
    logger = log2file(os.path.join(data_workdir, '%s.log' % (expr_code)))

    logger('expr_code: %s'%expr_code)
    logger("dataset_dir = '%s'"%data_workdir)

    logger(msg)

    final_dsdir = prepare_grab_dataset(data_workdir, contacts_dir, object_splits, logger=logger)
    
    
    
    