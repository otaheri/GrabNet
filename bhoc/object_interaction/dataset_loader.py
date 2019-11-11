import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import cPickle as pickle
import shutil
import os
import sys
import glob
import argparse
import random
import time
from collections import OrderedDict
from psbody.mesh import Mesh, MeshViewer, MeshViewers

import basis_point_sets.bps as bps
import basis_point_sets.normalization as bpsn

from psbody.smpl.rodrigues import Rodrigues

test_objects = ['elephant', 'cubesmall', 'wineglass','toothbrush', 'binoculars']


def contact_fname_loader(contact_parent_path, intent='all', train=True):
    object_filenames = OrderedDict()
    seq_names = []
    subjects = os.listdir(contact_parent_path)
    for subjectID in subjects:
        seq_names = glob.glob(os.path.join(contact_parent_path, subjectID, '*.pkl'))
        for seq_name in seq_names:
            basename = os.path.basename(seq_name)
            object_name = basename.split('_')[0]
            if 'all' in intent:
                pass
            elif intent not in basename:
                continue
            if train:
                if object_name in test_objects:
                    continue
            else:
                if object_name not in test_objects:
                    continue
            if object_name not in object_filenames:
                object_filenames[object_name] = [seq_name]
                seq_names.append(seq_name)
            else:
                object_filenames[object_name].append(seq_name)
                seq_names.append(seq_name)
    return object_filenames


class ContactDataSet(data.Dataset):
    def __init__(self,contact_parent_path,contact_expr_id, n_sample_points=5000, train=True,training_set=None,num_contact_labels=None, intent='all'):
        super(ContactDataSet, self).__init__()
        self.n_points = n_sample_points
        self.num_contact_labels = num_contact_labels
        self.contact_parent_path = contact_parent_path

        self.objectbased_filenames = OrderedDict()
        self.subjectbased_filenames = OrderedDict()
        self.seq_names = []
        self.train = train

        # if not self.train:
        #     try:
        #         self.generated_bps = training_set.generated_bps
        #     except:
        #         raise Exception('please provide the training set for bps')


        if os.walk(contact_parent_path).next()[1]:
            self.subjects = os.listdir(contact_parent_path)

        else:
            self.subjects = [contact_parent_path.split('/')[-1]]
            self.contact_parent_path = os.path.join(contact_parent_path, '..')

        for subjectID in self.subjects:
            self.subjectbased_filenames[subjectID]=[]
            seq_names = glob.glob(os.path.join(self.contact_parent_path,subjectID,contact_expr_id,'*.pkl'))
            for seq_name in seq_names:
                basename = os.path.basename(seq_name)
                object_name = basename.split('_')[0]
                if 'all' in intent:
                    pass
                elif intent not in basename:
                        continue
                if train:
                    if object_name in test_objects:
                        continue
                else:
                    if object_name not in test_objects:
                        continue
                if object_name not in self.objectbased_filenames:
                    self.objectbased_filenames[object_name] = [seq_name]
                    self.seq_names.append(seq_name)
                else:
                    self.objectbased_filenames[object_name].append(seq_name)
                    self.seq_names.append(seq_name)
                self.subjectbased_filenames[subjectID].append(seq_name)
        self.data_preprocessing()

    def get_contact_labels(self,y):
        labels = range(1, 56)[-self.num_contact_labels:]
        labels = [21,22,28,31,34,37,40,43,46,49,52,55]
        labels = [40,43,46,49,52,55]
        labels = sorted(labels)
        y=y.copy()
        filter_contact = np.isin(y,labels)
        y[~filter_contact]=0
        for idx, label in enumerate(labels):
            y[y==label] = idx+1
        return y

    def get_bps_per_frame(self,index):
        points = self.object_fullpts[self.frame_idx[index]].reshape(1, -1, 3)
        obj_pts_n, mean, scale = bpsn.unit_normalize_batch(points, return_scalers=True)
        object_pts, object_pts_choice = bps.convert_to_bps(obj_pts_n.reshape(1, -1, 3), self.generated_bps,
                                                           return_deltas=False)
        # self.object_pts[object_name] = np.append(object_pts,scale.reshape(-1,1))
        return  np.append(object_pts * scale, scale.reshape(-1, 1))


    def __len__(self):
        return len(self.frame_idx)

    def __getitem__(self, index):
        input_pose = self.list_input_data[index]
        input_pntcloud = self.object_pts[self.frame_idx[index]]
        ############
        # object_fullpts = self.object_fullpts[self.frame_idx[index]]
        # object_fullpts = np.dot(object_fullpts, Rodrigues(input_pose[3:]))
        # obj_pts_n, mean, scale = bpsn.unit_normalize_batch(object_fullpts.reshape(1, -1, 3), return_scalers=True)
        # object_pts, object_pts_choice, _ = bps.convert_to_bps(object_fullpts.reshape(1, -1, 3), self.generated_bps,
        #                                                    return_deltas=False)
        # input_pntcloud = np.append(object_pts,scale.reshape(-1,1))

        #####################
        # object_pts = self.object_pts[self.frame_idx[index]]
        # input_pntcloud = np.dot(object_pts, Rodrigues(input_pose[3:]))

        ################

        input = np.hstack([input_pntcloud.reshape(-1),input_pose])

        output_pose = self.list_output_data[index]
        output_contact = self.list_output_contact_data[index]
        # if self.num_contact_labels is not None:
        #     output_contact = self.get_contact_labels(output_contact)
        output = np.hstack([output_pose,output_contact])

        return input.astype(np.float32), output.astype(np.float32)


    def data_preprocessing(self,based_on='object'):

        self.object_pts = OrderedDict()
        self.object_pts_choice = OrderedDict()
        self.object_fullpts = OrderedDict()
        ##### for bps #####
        self.object_pts_bps = OrderedDict()
        self.object_pts_delta = OrderedDict()
        self.processed_data = OrderedDict()

        self.list_input_data = []
        self.list_output_data = []
        output_contact_data = []
        self.frame_idx = []  #### to know which object to use for each frame


        self.generated_bps = bps.generate_bps(n_points=self.n_points, radius=1.15)



        iter_on = self.objectbased_filenames if 'object' in based_on else self.subjectbased_filenames
        for object_name in iter_on.keys():

            self.processed_data[object_name]=[]

            for seq_name in iter_on[object_name]:

                with open(seq_name, 'rb') as f:
                    contact_res = pickle.load(f)
                print object_name
                # raw_input('please enter')
                self.v_template_file = contact_res['v_template']
                contact_frames_body = np.asarray(contact_res['body_contact_vertices'])
                contact_frames_object = np.asarray(contact_res['object_contact_vertices'])
                subject_pose_result = contact_res['subject_mosh_file']
                object_pose_result = contact_res['object_mosh_file']
                with open(subject_pose_result, 'rb') as f:
                    subject_pose_result = pickle.load(f)
                with open(object_pose_result, 'rb') as f:
                    object_pose_result = pickle.load(f)

                subject_pose_data = np.concatenate([subject_pose_result['pose_est_trans'], subject_pose_result['pose_est_poses']],
                                              axis=1)
                object_pose_data = np.concatenate([object_pose_result['pose_est_trans'], object_pose_result['pose_est_poses']],
                                             axis=1)

                ### to get in contact frames
                # in_contact_frames = np.asarray([False if item.size == 0 else True for item in contact_frames_body])

                in_contact_frames = []
                table_height = object_pose_data[0,2]
                right_hand_labels = range(41,56)
                left_hand_labels = range(26,41)
                just_right_hand = True
                if just_right_hand:
                    print 'ACHTUNG, THIS IS JUST FOR RIGHT HAND!!!'
                for counter, item in enumerate(contact_frames_body):
                    # if item.size!=0: # for contact without the labels
                    if np.unique(item).size>1: # for labeled contacts
                        if (object_pose_data[counter,2]>table_height+.0002) or (object_pose_data[counter,2]<table_height-.0002):
                            if just_right_hand:
                                if np.isin(item,right_hand_labels).any():
                                    in_contact_frames.append(True)
                                else:
                                    in_contact_frames.append(False)
                            else:
                                in_contact_frames.append(True)
                        else:
                            in_contact_frames.append(False)
                    else:
                        in_contact_frames.append(False)
                in_contact_frames = np.asarray(in_contact_frames)
                ##########
                contact_frames_body = contact_frames_body[in_contact_frames]
                contact_frames_object = contact_frames_object[in_contact_frames]
                subject_data = subject_pose_data[in_contact_frames]
                object_data = object_pose_data[in_contact_frames]

                ####### for body mesh ############
                if 'body' not in self.object_pts:
                    body_fullpts = Mesh(filename=self.v_template_file).v
                    self.object_pts['body'] = body_fullpts
                    self.object_fullpts['body'] = body_fullpts
                    self.object_pts_choice['body'] = np.arange(body_fullpts.shape[0])
                ##################################
                # object_mesh_fname = contact_res['object_mesh']
                # print  os.path.basename(object_mesh_fname)
                ####### sampling the object mesh ##############
                if object_name not in self.object_pts:
                    object_mesh_fname = contact_res['object_mesh']
                    print  os.path.basename(object_mesh_fname)
                    object_fullpts = Mesh(filename=object_mesh_fname).v
                    self.object_fullpts[object_name] = object_fullpts
                    if False:
                        offset = (object_fullpts.max(0, keepdims=True) + object_fullpts.min(0, keepdims=True)) / 2
                        object_fullpts -= offset
                        scale = max(object_fullpts.max(0) - object_fullpts.min(0)) / 2
                        # object_fullpts /= scale
                        # object_fullpts = np.vstack((object_fullpts, scale*np.ones(object_fullpts.shape[1])))
                        pts_choice = np.random.choice(object_fullpts.shape[0], size=self.n_points, replace=False)
                        object_pts = object_fullpts[pts_choice]
                        # self.object_pts[object_name] = np.append(object_pts,scale.reshape(-1,1))
                        self.object_pts[object_name] = object_pts
                        self.object_pts_choice[object_name] = pts_choice

                    else:
                        obj_pts_n, mean, scale = bpsn.unit_normalize_batch(object_fullpts.reshape(1, -1, 3), return_scalers=True)
                        # object_pts , object_pts_choice = bps.convert_to_bps(obj_pts_n.reshape(1, -1, 3), self.generated_bps, return_deltas=True)
                        object_pts , object_pts_choice, trgt2src = bps.convert_to_bps(obj_pts_n.reshape(1, -1, 3), self.generated_bps, return_deltas=False)
                        # self.object_pts[object_name] = np.append(object_pts,scale.reshape(-1,1))
                        # scale = max(object_fullpts.max(0) - object_fullpts.min(0)) / 2
                        self.object_pts[object_name] = np.append(object_pts,scale.reshape(-1,1))
                        self.object_pts_choice[object_name] = object_pts_choice.reshape(-1)
                        # mesh_reconstruct = bps.reconstruct_from_bps(self.obj_deltas_bps, self.generated_bps)

                # contact_frames_object = self.per_frame_contact(contact_frames_object=contact_frames_object,object_name=object_name)
                # contact_frames_body = self.per_frame_contact(contact_frames_object=contact_frames_body,object_name='body')

                contact_frames_object = contact_frames_object[:,self.object_pts_choice[object_name]]


                closest_bps2object = np.unique(trgt2src.reshape(-1)[self.object_pts_choice[object_name]])
                contact_bps = np.zeros_like(contact_frames_object[:, self.object_pts_choice[object_name]])
                contact_bps[:,closest_bps2object] = contact_frames_object[:,self.object_pts_choice[object_name][closest_bps2object]]
                contact_frames_object = contact_bps

                object_names = np.repeat(object_name, contact_frames_object.shape[0], axis=0)
                input = object_data
                output = subject_data
                output_contact = np.concatenate([contact_frames_object, contact_frames_body], axis=1).astype(np.int8)

                self.list_input_data.append(input)
                self.list_output_data.append(output)
                output_contact_data.append(output_contact)
                self.frame_idx.append(object_names)

        self.frame_idx = np.concatenate(self.frame_idx)
        self.list_output_contact_data = np.concatenate(output_contact_data).astype(np.int8)
        self.list_input_data = np.concatenate(self.list_input_data).astype(np.float32)
        self.list_output_data = np.concatenate(self.list_output_data).astype(np.float32)
        print 'finished'




    def per_frame_contact(self, contact_frames_object,object_name):
        contact_frames_object_choice = []
        for frame in contact_frames_object:
            vrts_contact = np.zeros(self.object_fullpts[object_name].shape[0], np.int32)
            vrts_contact[frame] = 1
            vrts_contact_choice = vrts_contact[self.object_pts_choice[object_name]]
            contact_frames_object_choice.append(vrts_contact_choice)
        return np.asarray(contact_frames_object_choice).astype(np.bool)


                # Mesh(v=object_pts, f=[]).show()


if __name__=='__main__':

    contact_parent_path = '/ps/project/body_hand_object_contact/contact_results/13_omid/190920_00174/02'

    params = {  'batch_size': 32,
                'shuffle': True}

    # a = contact_fname_loader(contact_parent_path=contact_parent_path)
    training_set = ContactDataSet(contact_parent_path, n_sample_points=5000, train=True, intent='all')
    a = training_set[1]
    training_generator = data.DataLoader(training_set, **params)

    for X,y in training_generator:
        print X.shape
        print y.shape

    # training_data[1]
    print 'finished'









