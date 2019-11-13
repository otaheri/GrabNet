
import glob
import os
import shutil
import sys
from datetime import datetime
from tqdm import tqdm
import numpy as np

import torch

# from human_body_prior.tools.omni_tools import colors
from human_body_prior.tools.omni_tools import makepath, log2file

from human_body_prior.body_model.body_model import BodyModel
from bhoc.tools.object_model import ObjectModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from psbody.mesh import Mesh
from human_body_prior.tools.omni_tools import colors

from basis_point_sets.bps import generate_bps, convert_to_bps, reconstruct_from_bps
from basis_point_sets.bps_torch import convert_to_bps_batch, convert_to_bps_chamfer

from psbody.mesh import Mesh, MeshViewer
import pickle


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

    def get_verts(object_name, data_c=None):
        if object_name not in object_infos:
            np.random.seed(100)  # this makes it possible to reproduce the sample ids
            object_verts = Mesh(filename=data_c['object_mesh']).v
            object_verts_dsampl_ids = np.random.choice(object_verts.shape[0], max_num_object_verts, replace=False)
            object_verts_dsampl = object_verts[object_verts_dsampl_ids]
            object_infos[object_name] = {'object_verts': object_verts,
                                         'object_verts_dsampl_ids': object_verts_dsampl_ids,
                                         'object_verts_dsampl': object_verts_dsampl}
        else:
            object_verts_dsampl_ids = object_infos[object_name]['object_verts_dsampl_ids']
            object_verts_dsampl = object_infos[object_name]['object_verts_dsampl']

        get_verts.object_infos = object_infos

        return object_verts_dsampl, object_verts_dsampl_ids

    get_verts.object_infos = object_infos
    return get_verts


def prepare_bhoc_dataset(data_workdir, contacts_dir, object_splits, logger=None):
    BPS_N_POINTS = 1024
    FPS_DS_RATE = 5
    MAX_NUM_OBJECT_VERTS = 5000 # equal to the number of hand verts

    include_joints = list(range(41, 56))
    exclude_joints = list(range(26, 41))

    bps_basis_fname = makepath(os.path.join(data_workdir, 'bps_basis_%d.npz'%BPS_N_POINTS), isfile=True)
    if os.path.exists(bps_basis_fname):
        basis = np.load(bps_basis_fname)['basis']
    else:
        basis = generate_bps(n_points=BPS_N_POINTS, radius=0.18, n_dims=3, kd_ordering=True)
        np.savez(bps_basis_fname, basis=basis)

    part2vids = np.load('/ps/scratch/body_hand_object_contact/bhoc_network/part2vids.npz')

    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hand_vids = part2vids['handR']

    starttime = datetime.now().replace(microsecond=0)

    if logger is None:
        logger = log2file()
        logger('Starting augmenting task at %s'%datetime.strftime(starttime, '%Y%m%d_%H%M'))

    shutil.copy2(sys.argv[0], os.path.join(data_workdir, os.path.basename(sys.argv[0]).replace('.py', '_%s.py' % datetime.strftime(starttime, '%Y%m%d_%H%M'))))

    logger('Starting with data preparation')
    logger('Num objects per datasplit: %s'%(', '.join(['%s: %d'%(k, len(v)) for k,v in object_splits.items()])))

    object_info_path = makepath(os.path.join(data_workdir, 'object_infos.pkl'), isfile=True)
    object_loader = load_object_verts(object_info_path, max_num_object_verts=MAX_NUM_OBJECT_VERTS)

    subject_infos = {}

    for split_name in object_splits.keys():
        outfname = makepath(os.path.join(data_workdir, split_name, 'os_delta.pt'), isfile=True)

        if os.path.exists(outfname):
            logger('Results for %s split already exist.'%(split_name))
            continue

        contact_fnames = glob.glob(os.path.join(contacts_dir, '*/*_stageII.pkl'))

        out_data = {'os_delta': [], 'o_delta':[], 's_delta':[], 'offset':[], 'pose_hand':[]}
        # contact_maps_fnames = []
        for contact_fname in tqdm(contact_fnames):
            object_name = os.path.basename(contact_fname).split('_')[0]
            if object_name not in object_splits[split_name]: continue
            subject_name = contact_fname.split('/')[-2]
            # action_name = '_'.join(os.path.basename(contact_fname).split('_')[1:-1])
            # action_names.append(action_name)
            with open(contact_fname, 'rb') as f: data_c = pickle.load(f, encoding='latin1')
            with open(data_c['subject_mosh_file'], 'rb') as f: data_s = pickle.load(f, encoding='latin1')
            with open(data_c['object_mosh_file'], 'rb') as f: data_o = pickle.load(f, encoding='latin1')

            frame_mask = get_frame_contact_mask(data_c, data_o, include_joints, exclude_joints)
            T = frame_mask.sum()
            if T<1:
                logger('%s has no in contact frame.'%contact_fname)
                continue

            object_verts, _ = object_loader(object_name, data_c)

            if subject_name in subject_infos:
                v_template_s = subject_infos[split_name]
            else:
                v_template_s = Mesh(filename=data_s['vtemplate_fname']).v
                subject_infos[split_name] = v_template_s

            bm = BodyModel(bm_path=data_s['shape_est_templatemodel'].replace('.pkl', '.npz'), batch_size=T, v_template=v_template_s)
            om = ObjectModel(v_template=object_verts, batch_size=T)

            object_parms = {'trans':data_o['pose_est_trans'][frame_mask], 'root_orient':data_o['pose_est_poses'][frame_mask]}

            object_parms = {k: torch.from_numpy(v).type(torch.float32) for k, v in object_parms.items()}

            verts_o = c2c(om(**object_parms).v)
            body_full_pose = data_s['pose_est_fullposes'][frame_mask]
            body_params = {'root_orient': body_full_pose[:, :3],
                           'pose_body': body_full_pose[:, 3:66],
                           'pose_hand': body_full_pose[:, 75:],
                           'trans': data_s['pose_est_trans'][frame_mask],
                           }
            body_params = {k: torch.from_numpy(v).type(torch.float32) for k, v in body_params.items()}

            verts_s = c2c(bm(**body_params).v[:,hand_vids])

            verts_os = np.concatenate([verts_o, verts_s], axis=1)

            offset = verts_o.mean(1)[:, None] # center both wrt to object

            # basis_batched = torch.from_numpy(np.repeat(basis[None], repeats=T, axis=0).astype(np.float32))

            # os_delta, o_delta, s_delta = [], [], []
            # for tId in range(T):
            #     os_delta.append(convert_to_bps(verts_os[tId]-offset[tId], basis, return_deltas=True)[1])
            #     o_delta.append(convert_to_bps(verts_o[tId]-offset[tId], basis, return_deltas=True)[1])
            #     s_delta.append(convert_to_bps(verts_s[tId]-offset[tId], basis, return_deltas=True)[1])

            sys.path.append('/is/ps2/nghorbani/code-repos/basis_point_sets/chamfer-extension')

            _, os_delta = convert_to_bps_chamfer(verts_os-offset, basis, return_deltas=True)
            _, o_delta = convert_to_bps_chamfer(verts_o-offset, basis, return_deltas=True)
            _, s_delta = convert_to_bps_chamfer(verts_s-offset, basis, return_deltas=True)

            # os_delta, o_delta, s_delta = torch.cat(os_delta), torch.cat(o_delta), torch.cat(s_delta)

            out_data['os_delta'].append(os_delta)
            out_data['o_delta'].append(o_delta)
            out_data['s_delta'].append(s_delta)
            out_data['offset'].append(offset)
            out_data['pose_hand'].append(body_full_pose[:, 75:])

        for k,v in out_data.items():

            outfname = makepath(os.path.join(data_workdir, split_name, '%s.pt' % k), isfile=True)
            if os.path.exists(outfname): continue
            out_data[k] = torch.from_numpy(np.concatenate(v))
            torch.save(out_data[k], outfname)

        logger('%d datapoints for %s'%(out_data['os_delta'].shape[0], split_name))

    with open(object_info_path, 'wb') as f:
        pickle.dump(object_loader.object_infos, f, protocol=2)

    # action_names = list(set(action_names))
    # print(len(action_names), action_names)


if __name__ == '__main__':

    msg = '\n'

    contacts_dir = '/ps/scratch/body_hand_object_contact/contact_results/17/03_thrshld_50e_6_final'
    # contacts_dir = '/ps/scratch/body_hand_object_contact/contact_results/16_omid/02_thrshld_20e_6_final'
    # contacts_dir = '/ps/scratch/body_hand_object_contact/contact_results/16_01_thrshld_15e_6_final'
    object_names = get_object_names(contacts_dir)
    
    
    object_splits = {
        'test': ['elephant', 'cubesmall', 'wineglass','toothbrush', 'binoculars'],
        'vald': ['banana', 'toothpaste', 'mug', 'waterbottle', 'fryingpan']
    }
    object_splits['train'] = list(set(object_names).difference(set(object_splits['test'] + object_splits['vald'])))

    expr_code = 'V01_01_00'

    data_workdir = os.path.join('/ps/scratch/body_hand_object_contact/bhoc_network/data', expr_code)
    logger = log2file(os.path.join(data_workdir, '%s.log' % (expr_code)))

    logger('expr_code: %s'%expr_code)
    logger("dataset_dir = '%s'"%data_workdir)

    logger(msg)

    final_dsdir = prepare_bhoc_dataset(data_workdir, contacts_dir, object_splits, logger=logger)