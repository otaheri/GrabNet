
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

from basis_point_sets.bps import generate_bps, convert_to_bps

from psbody.mesh import Mesh
import pickle


def get_object_names(contact_results_dir):
    object_names = []
    contact_fnames = glob.glob(os.path.join(contact_results_dir, '*/*_stageII.pkl'))
    for contact_fname in contact_fnames:
        object_name = os.path.basename(contact_fname).split('_')[0]
        object_names.append(object_name)
    return list(set(object_names))

def get_frame_contact_mask(data_c, data_o, object_verts_dsampl_ids, include_joints=range(41, 56), exclude_joints = range(26, 41)):
    in_contact_frames = []
    table_height = data_o['pose_est_trans'][0, 2]
    for fId, fdata in enumerate(data_c['object_contact_vertices']):
        fdata = fdata[object_verts_dsampl_ids]

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

def prepare_bhoc_dataset(data_workdir, contacts_dir, object_splits, logger=None):
    interested_action_names = ['pick_all']
    MAX_NUM_OBJECT_VERTS = 7000
    BPS_N_POINTS = 512
    FPS_DS_RATE = 5
    MIN_NUM_CONTACTS = 10

    include_joints = list(range(41, 56))
    exclude_joints = list(range(26, 41))

    bps_basis_fname = makepath(os.path.join(data_workdir, 'bps_basis_%d.npz'%BPS_N_POINTS), isfile=True)
    if os.path.exists(bps_basis_fname):
        basis = np.load(bps_basis_fname)['basis']
    else:
        basis = generate_bps(n_points=BPS_N_POINTS, radius=2, n_dims=3, kd_ordering=True)
        np.savez(bps_basis_fname, basis=basis)

    starttime = datetime.now().replace(microsecond=0)

    if logger is None:
        logger = log2file()
        logger('Starting augmenting task at %s'%datetime.strftime(starttime, '%Y%m%d_%H%M'))

    shutil.copy2(sys.argv[0], os.path.join(data_workdir, os.path.basename(sys.argv[0]).replace('.py', '_%s.py' % datetime.strftime(starttime, '%Y%m%d_%H%M'))))

    logger('Starting with data preparation')
    logger('Num objects per datasplit: %s'%(', '.join(['%s: %d'%(k, len(v)) for k,v in object_splits.items()])))

    for split_name in object_splits.keys():
        contact_map_outpath = makepath(os.path.join(data_workdir, split_name, 'contact_maps.pt'), isfile=True)
        # if os.path.exists(contact_map_outpath):
        #     logger('Results for %s split already exist.'%(split_name))
        #     continue

        contact_fnames = glob.glob(os.path.join(contacts_dir, '*/*_stageII.pkl'))

        object_infos = {}
        contact_maps = []
        contact_maps_fnames = []
        contact_maps_object_names = []
        for contact_fname in tqdm(contact_fnames):
            object_name = os.path.basename(contact_fname).split('_')[0]
            if object_name not in object_splits[split_name]: continue
            action_name = '_'.join(os.path.basename(contact_fname).split('_')[1:-1])
            if not np.any([True for a in interested_action_names if a in action_name]): continue
            # action_names.append(action_name)
            with open(contact_fname, 'rb') as f: data_c = pickle.load(f, encoding='latin1')
            # with open(data_c['subject_mosh_file'], 'rb') as f: data_s = pickle.load(f, encoding='latin1')
            with open(data_c['object_mosh_file'], 'rb') as f: data_o = pickle.load(f, encoding='latin1')

            if object_name not in object_infos:
                np.random.seed(100)#this makes it possible to reproduce the sample ids
                object_verts = Mesh(filename=data_c['object_mesh']).v
                object_verts_dsampl_ids = np.random.choice(object_verts.shape[0], MAX_NUM_OBJECT_VERTS, replace=False)
                object_verts_dsampl = object_verts[object_verts_dsampl_ids]
                object_verts_dsampl_bps, object_verts_dsampl_bps_deltas = convert_to_bps(object_verts_dsampl, basis=basis, return_deltas=True)

                object_infos[object_name] = {'object_verts': object_verts,
                    'object_verts_dsampl_ids': object_verts_dsampl_ids, 'object_verts_dsampl':object_verts_dsampl,
                                                        'object_verts_dsampl_bps':object_verts_dsampl_bps,  'object_verts_dsampl_bps_deltas':object_verts_dsampl_bps_deltas}
            else:
                object_verts_dsampl_ids = object_infos[object_name]['object_verts_dsampl_ids']
                # object_verts_dsampl = object_infos[object_name]['object_verts_dsampl']
                # object_verts_dsampl_bps = object_infos[object_name]['object_verts_dsampl_bps']
                # object_verts_dsampl_bps_deltas = object_infos[object_name]['object_verts_dsampl_bps_deltas']

            frame_contact_mask = get_frame_contact_mask(data_c, data_o, object_verts_dsampl_ids, include_joints, exclude_joints)

            object_contact_map = np.asarray(data_c['object_contact_vertices'])[frame_contact_mask]
            object_contact_map_fnames = np.asarray(['%s'%(contact_fname+'_%05d'%fId) for fId in range(len(frame_contact_mask))])[frame_contact_mask]

            object_contact_map_dsample = object_contact_map[:, object_verts_dsampl_ids]
            object_contact_map_dsample_binary = np.where(object_contact_map_dsample != 0, np.ones_like(object_contact_map_dsample), object_contact_map_dsample)

            unwanted_contact_labels = set((object_contact_map_dsample.reshape(-1)).tolist()).difference(set(include_joints + [0]))
            for j in unwanted_contact_labels:
                mask = object_contact_map_dsample == j
                object_contact_map_dsample_binary = np.where(mask, np.zeros_like(object_contact_map_dsample), object_contact_map_dsample_binary)
                object_contact_map_dsample = np.where(mask, np.zeros_like(object_contact_map_dsample), object_contact_map_dsample)

            # for t in range(object_contact_map_dsample.shape[0]):
            #     for j in include_joints:
            #         if (object_contact_map_dsample[t] == j).sum() < 10:
            #             mask = object_contact_map_dsample[t] == j
            #             object_contact_map_dsample[t, mask] = 0
            #             object_contact_map_dsample_binary[t, mask] = 0

            finger_class_id = 0
            for j in include_joints:
                finger_class_id += 1
                object_contact_map_dsample = np.where(object_contact_map_dsample == j, np.ones_like(object_contact_map_dsample)*finger_class_id, object_contact_map_dsample)

            mask = object_contact_map_dsample_binary.sum(1)>MIN_NUM_CONTACTS
            object_contact_map_dsample = object_contact_map_dsample[mask] # at least 10 points to be in contact
            object_contact_map_dsample = object_contact_map_dsample[::FPS_DS_RATE]

            object_contact_map_fnames = object_contact_map_fnames[mask]
            object_contact_map_fnames = object_contact_map_fnames[::FPS_DS_RATE]

            T = object_contact_map_dsample.shape[0]
            contact_maps.append(object_contact_map_dsample)
            contact_maps_fnames.append(object_contact_map_fnames)
            contact_maps_object_names.extend([object_name for _ in range(T)])

        contact_maps = np.concatenate(contact_maps, axis=0).astype(np.int32)

        torch.save(torch.tensor(contact_maps), contact_map_outpath)
        np.savez(contact_map_outpath.replace('contact_maps.pt', 'contact_maps_object_names.npz'), object_names = np.array(contact_maps_object_names))

        class_count = {k:(contact_maps == k).sum() for k in range(contact_maps.max())}

        with open(contact_map_outpath.replace('contact_maps.pt', 'data_info.pkl'), 'wb') as f:
            pickle.dump({'class_count': class_count, 'object_infos': object_infos, 'bps_basis': basis, 'contact_maps_fnames':np.concatenate(contact_maps_fnames, axis=0)}, f, protocol=2)

        logger('%d datapoints for %s'%(contact_maps.shape[0], split_name))
        logger('class_counts [class_id: count]: {%s}'%(', '.join(['%s: %d'%(k,v) for k,v in class_count.items()])))

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
        # 'vald': ['banana', 'toothpaste', 'mug', 'waterbottle', 'fryingpan']
    }
    # object_splits['train'] = list(set(object_names).difference(set(object_splits['test'] + object_splits['vald'])))

    expr_code = 'V01_02_00'

    data_workdir = os.path.join('/ps/scratch/body_hand_object_contact/bhoc_network/data', expr_code)
    logger = log2file(os.path.join(data_workdir, '%s.log' % (expr_code)))

    logger('expr_code: %s'%expr_code)
    logger("dataset_dir = '%s'"%data_workdir)

    logger(msg)

    final_dsdir = prepare_bhoc_dataset(data_workdir, contacts_dir, object_splits, logger=logger)