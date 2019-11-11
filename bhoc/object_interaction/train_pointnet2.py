import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import cPickle as pickle
import shutil
import os
import sys
import glob
import argparse
import random
import time
import warnings
from collections import Counter


from psbody.mesh import MeshViewers, MeshViewer, Mesh
from experiments.nima.tools.visualization_helpers import spheres_for
from psbody.mesh.lines import Lines
from configer import Configer

from experiments.omid.object_interaction.dataset_loader import ContactDataSet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from experiments.omid.object_interaction.models import models
from models.models import VAE
from models.models import VAEPointnet
import basis_point_sets.bps as bps
import basis_point_sets.normalization as bpsn

from psbody.mesh.colors import name_to_rgb
from experiments.hand.fast_derivatives.smpl_HF_fastderivatives_WRAPPER import load_model as load_smplx
from experiments.nima.tools_ch.object_model import RigidObjectModel
import torchgeometry as tgm
from copy import deepcopy
import smplx
from smplx.lbs import (
    lbs, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vertex_label_contact = np.load(
            '/ps/scratch/body_hand_object_contact/contact_results/vertex_label_contact.npy').astype(np.int8)
def loss_fn(x, reconstructed_x, mean, log_var):

    ########## reconstruction loss
    # weight = torch.tensor([1, 20], dtype=torch.float32).to(device)
    # loss_f = nn.CrossEntropyLoss(reduction='none', weight=weight)

    # reconstructed_x = reconstructed_x.unsqueeze(1)
    # x = x.unsqueeze(1)
    # reconstructed_x = torch.stack([1-reconstructed_x, reconstructed_x], dim=-1)
    # x = torch.stack([1-x, x], dim=-1)
    # loss = loss_f(reconstructed_x, x.to(torch.long))
    # RCL = torch.mean(loss)
    # RCL = F.binary_cross_entropy(reconstructed_x, x,weight=weight, size_average=False)
    ###########

    RCL = F.binary_cross_entropy(reconstructed_x, x, size_average=False)
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return RCL + KLD

def loss_fn_pose(input, reconstructed_input,vertices, joints, z_mu, z_var,body_model, loss_on=None):
    pose = input[:,-495:]
    contact_verts = input[:,:-495]

    pose_hat = reconstructed_input[:,-495:]
    contact_verts_hat = reconstructed_input[:,:-495]

    loss_pose = torch.mean(torch.pow((pose - pose_hat), 2))
    if loss_on is not None:

        body_model.use_pca = False
        pose_h = pose_hat.view(8, -1, 3, 3).clone()
        # pose_hat = tgm.rotation_matrix_to_angle_axis(
        #             tgm.convert_points_to_homogeneous(pose_hat.contiguous().view(-1,3,3))).contiguous().view(input.shape[0],-1)

        # global_orient = pose_hat[:, :3]
        # body_pose = pose_hat[:, 3:66]
        # left_hpose = pose_hat[:,75:120]
        # right_hpose = pose_hat[:,120:165]
        # body_model.flat_hand_mean = True
        # output = body_model(return_verts=True, return_full_pose=False,global_orient=global_orient, body_pose = body_pose, left_hand_pose= left_hpose, right_hand_pose = right_hpose)
        # body_model.flat_hand_mean = False

        vertices_hat, joints_hat = lbs(body_model.shape_components.detach(), pose_h, body_model.v_template,
                               body_model.shapedirs, body_model.posedirs,
                               body_model.J_regressor, body_model.parents,
                               body_model.lbs_weights, pose2rot=False,
                               dtype=body_model.dtype)

        # vertices_hat = output.vertices
        # joints_hat = output.joints
        if 'joint' in loss_on:
          loss_hands = 10*torch.mean(torch.pow((joints[:,21:23] - joints_hat[:,21:23]),2))
          # loss_joints = torch.mean(torch.pow((joints[:,:55] - joints_hat[:,:55]),2))
          loss_pose = loss_pose + loss_hands #+ loss_joints

        else:
            a = np.isin(vertex_label_contact,np.array(range(26,56))).astype(np.int32)
            idx = np.where(a==1)[0]
            loss_pose += 5*torch.mean(torch.pow((vertices - vertices_hat),2))
            # loss_pose += 2.5*torch.mean(torch.pow((vertices[:,torch.from_numpy(idx).to(torch.long)] - vertices_hat[:,torch.from_numpy(idx).to(torch.long)]),2))


    # from experiments.nima.tools.visualization_helpers import spheres_for
    # aa = joints[0, :55]
    # bb = spheres_for(aa.cpu().numpy(), radius=.01)
    # mvs = MeshViewers(shape=[1, 1])
    # mvs[0][0].set_static_meshes(bb)

    # if contact_verts_hat.min()<0 or contact_verts_hat.max()>1 or contact_verts.min()<0 or contact_verts.max()>1:
    #     pass
    weight = torch.tensor([1, 20], dtype=torch.float32).to(device)
    RCL = F.binary_cross_entropy(contact_verts_hat, contact_verts, size_average=True)
    # RCL = F.binary_cross_entropy(torch.stack([contact_verts_hat,1-contact_verts_hat],dim=-1),
    #                              torch.stack([contact_verts    ,1-contact_verts    ],dim=-1),weight=weight, size_average=True)
    # loss_fn = nn.CrossEntropyLoss(ignore_index=2, reduction='none')
    # RCL = torch.mean(torch.pow((contact_verts_hat - contact_verts),2))
    # RCL=0
    KLD = -0.5 * torch.sum(1 + z_var - z_mu.pow(2) - z_var.exp())

    return KLD + 30*RCL + 30*loss_pose


def local2global_pose(local_pose):

    local_pose = local_pose.view(local_pose.shape[0], -1, 3, 3)
    global_pose = local_pose.clone()
    global_pose[:, 1] = torch.matmul(global_pose[:, 0], global_pose[:, 1].clone())
    global_pose[:, 2] = torch.matmul(global_pose[:, 0], global_pose[:, 2].clone())
    global_pose[:, 3] = torch.matmul(global_pose[:, 0], global_pose[:, 3].clone())

    global_pose[:, 4] = torch.matmul(global_pose[:, 1], global_pose[:, 4].clone())
    global_pose[:, 5] = torch.matmul(global_pose[:, 2], global_pose[:, 5].clone())
    global_pose[:, 6] = torch.matmul(global_pose[:, 3], global_pose[:, 6].clone())

    global_pose[:, 7] = torch.matmul(global_pose[:, 4], global_pose[:, 7].clone())
    global_pose[:, 8] = torch.matmul(global_pose[:, 5], global_pose[:, 8].clone())
    global_pose[:, 9] = torch.matmul(global_pose[:, 6], global_pose[:, 9].clone())

    global_pose[:, 10] = torch.matmul(global_pose[:, 7], global_pose[:, 10].clone())
    global_pose[:, 11] = torch.matmul(global_pose[:, 8], global_pose[:, 11].clone())
    global_pose[:, 12] = torch.matmul(global_pose[:, 9], global_pose[:, 12].clone())

    global_pose[:, 13] = torch.matmul(global_pose[:, 9], global_pose[:, 13].clone())
    global_pose[:, 14] = torch.matmul(global_pose[:, 9], global_pose[:, 14].clone())
    global_pose[:, 15] = torch.matmul(global_pose[:, 12], global_pose[:, 15].clone())

    global_pose[:, 16] = torch.matmul(global_pose[:, 13], global_pose[:, 16].clone())
    global_pose[:, 17] = torch.matmul(global_pose[:, 14], global_pose[:, 17].clone())
    global_pose[:, 18] = torch.matmul(global_pose[:, 16], global_pose[:, 18].clone())

    global_pose[:, 19] = torch.matmul(global_pose[:, 17], global_pose[:, 19].clone())
    global_pose[:, 20] = torch.matmul(global_pose[:, 18], global_pose[:, 20].clone())
    global_pose[:, 21] = torch.matmul(global_pose[:, 19], global_pose[:, 21].clone())

    return global_pose.view(global_pose.shape[0],-1)
def global2local_pose(global_pose):

    global_pose = global_pose.view(global_pose.shape[0], -1, 3, 3)
    local_pose = deepcopy(global_pose)
    local_pose[:, 1] = torch.matmul(global_pose[:, 0].transpose(2,1), global_pose[:, 1])
    local_pose[:, 2] = torch.matmul(global_pose[:, 0].transpose(2,1), global_pose[:, 2])
    local_pose[:, 3] = torch.matmul(global_pose[:, 0].transpose(2,1), global_pose[:, 3])

    local_pose[:, 4] = torch.matmul(global_pose[:, 1].transpose(2,1), global_pose[:, 4])
    local_pose[:, 5] = torch.matmul(global_pose[:, 2].transpose(2,1), global_pose[:, 5])
    local_pose[:, 6] = torch.matmul(global_pose[:, 3].transpose(2,1), global_pose[:, 6])

    local_pose[:, 7] = torch.matmul(global_pose[:, 4].transpose(2,1), global_pose[:, 7])
    local_pose[:, 8] = torch.matmul(global_pose[:, 5].transpose(2,1), global_pose[:, 8])
    local_pose[:, 9] = torch.matmul(global_pose[:, 6].transpose(2,1), global_pose[:, 9])

    local_pose[:, 10] = torch.matmul(global_pose[:, 7].transpose(2,1), global_pose[:, 10])
    local_pose[:, 11] = torch.matmul(global_pose[:, 8].transpose(2,1), global_pose[:, 11])
    local_pose[:, 12] = torch.matmul(global_pose[:, 9].transpose(2,1), global_pose[:, 12])

    local_pose[:, 13] = torch.matmul(global_pose[:, 9].transpose(2,1), global_pose[:, 13])
    local_pose[:, 14] = torch.matmul(global_pose[:, 9].transpose(2,1), global_pose[:, 14])
    local_pose[:, 15] = torch.matmul(global_pose[:, 12].transpose(2,1), global_pose[:, 15])

    local_pose[:, 16] = torch.matmul(global_pose[:, 13].transpose(2,1), global_pose[:, 16])
    local_pose[:, 17] = torch.matmul(global_pose[:, 14].transpose(2,1), global_pose[:, 17])
    local_pose[:, 18] = torch.matmul(global_pose[:, 16].transpose(2,1), global_pose[:, 18])

    local_pose[:, 19] = torch.matmul(global_pose[:, 17].transpose(2,1), global_pose[:, 19])
    local_pose[:, 20] = torch.matmul(global_pose[:, 18].transpose(2,1), global_pose[:, 20])
    local_pose[:, 21] = torch.matmul(global_pose[:, 19].transpose(2,1), global_pose[:, 21])

    return local_pose.view(local_pose.shape[0],-1)

def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    C = labels.max()
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = Variable(target)

    return target
def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(y.shape[0],-1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

def prepare_input(body_model,x,y,num_labels=None, use_pca=True):

    # 5000 points + 1 scale + 3 trans + 3 pose of the object
    x = x.to(device)
    # 102 pose parameters + 5000 contact object + 10475 contact body
    # y = y[:, :5102].to(device)

    # for both body and object
    # get selected labels for contact
    #################################
    y = y[:,:5102].to(device)
    # y_one_hot = to_one_hot(y[:,102:],n_dims=13).to(device)

    x[:, 5001:5004] = x[:, 5001:5004] - y[:, :3]  # object location relative to the root
    object_pose = tgm.angle_axis_to_rotation_matrix(x[:, 5004:])[:, :3, :3].contiguous().view(x.shape[0], -1)

    # condition = x[:,5001:5004] # condition on the object location
    condition = torch.cat([x[:, :5004], object_pose], dim=1)

    body_model.use_pca = use_pca
    if use_pca:
        global_orient = y[:, 3:6]
        body_pose = y[:, 6:69]
        left_hpose = y[:,78:90]
        right_hpose = y[:,90:102]
    else:
        global_orient = y[:, :3]
        body_pose = y[:, 3:66]
        left_hpose = y[:,75:120]
        right_hpose = y[:,120:165]
    output = body_model(return_verts=True, return_full_pose=True,global_orient =global_orient, body_pose = body_pose, left_hand_pose= left_hpose, right_hand_pose = right_hpose)

    input_pose = output.full_pose.detach()
    input_pose = tgm.angle_axis_to_rotation_matrix(input_pose.contiguous().view(-1, 3))[:, :3, :3].contiguous().view(x.shape[0], -1)
    # input = torch.cat([y_one_hot, input_pose], dim=1)
    y[:, 102:] = y[:, 102:]>0
    input = torch.cat([y[:, 102:], input_pose], dim=1)

    vertices = output.vertices.detach()
    joints = output.joints.detach()
    return input,condition, vertices,joints

def train_pose(model, train_iterator, optimizer, body_model):
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for i, (x, y) in enumerate(train_iterator):


        input,condition, vertices,joints = prepare_input(body_model,x,y)

        optimizer.zero_grad()
        reconstructed_input, z_mu, z_var,_ = model(input, condition)
        loss = loss_fn_pose(input, reconstructed_input,vertices, joints, z_mu, z_var,body_model, loss_on='vertices')
        loss.backward()
        train_loss += loss.item()
        #
        if i % 100 == 0:
            print ('iteration %d' % i)
            print ('loss %.4f' % loss)
        #
        optimizer.step()


    return train_loss

def eval_pose(test_iterator,model,body_model):
    # set the evaluation mode
    # contact_vis_pose(test_set, model)
    model.eval()

    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, (x, y) in enumerate(test_iterator):
            # reshape the data
            input,condition, vertices,joints = prepare_input(body_model,x,y)
            # forward pass
            reconstructed_input, z_mu, z_var,_ = model(input, condition)

            # loss
            loss = loss_fn_pose(input, reconstructed_input,vertices, joints, z_mu, z_var,body_model)
            test_loss += loss.item()
            if i % 100 == 0:
                print ('iteration %d'%i)
                print ('loss %.4f'%loss)

    return test_loss

def train(model, train_iterator, optimizer):
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for i, (x, y) in enumerate(train_iterator):


        x = x[:, :n_points*3].view(-1,n_points,3).to(device)
        y = y[:, 102:102 + n_points].unsqueeze(dim=-1).to(device)
        y[y>0] = 1

        # convert y into one-hot encoding
        input = y
        condition = x
        # update the gradients to zero
        optimizer.zero_grad()
        # forward pass
        reconstructed_input, z_mu, z_var,_ = model(input, condition)
        # loss
        loss = loss_fn(input, reconstructed_input, z_mu, z_var)
        # backward pass
        loss.backward()
        train_loss += loss.item()
        #
        if i % 100 == 0:
            print ('iteration %d' % i)
            print ('loss %.4f' % loss)

        # update the weights
        optimizer.step()

    return train_loss

def eval(test_iterator,model):
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, (x, y) in enumerate(test_iterator):
            # reshape the data
            x = x[:, :n_points*3].view(-1,n_points,3).to(device)
            y = y[:, 102:102+n_points].to(device)
            y[y>0] = 1
            # y[y[:,102:102+n_points] > 0] = 1

            # convert y into one-hot encoding
            input = y
            condition = x

            # forward pass
            reconstructed_input, z_mu, z_var,_ = model(input, condition)

            # loss
            loss = loss_fn(input, reconstructed_input, z_mu, z_var)
            test_loss += loss.item()
            if i % 100 == 0:
                print ('iteration %d'%i)
                print ('loss %.4f'%loss)

    return test_loss

def contact_vis(test_set,model):
    from psbody.mesh import Mesh, MeshViewers
    mvs = MeshViewers(shape=[1,1])
    model_path = makepath(os.path.join(result_dir, 'snapshot'))
    for idx in np.random.randint(100, len(test_set), 10):
        a, b = test_set.__getitem__(idx)
        object_name = test_set.frame_idx[idx]
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)
        a = a[:n_points].view(1, -1).to(device)
        b = b[102:102 + n_points].view(1, -1).to(device)
        obj_pts_n, mean, scale = bpsn.unit_normalize_batch(test_set.object_fullpts[object_name].reshape(1, -1, 3),
                                                           return_scalers=True)
        object_pts, deltas = bps.convert_to_bps(obj_pts_n.reshape(1, -1, 3), test_set.generated_bps,
                                                return_deltas=True)
        mesh_reconstruct = bps.reconstruct_from_bps(deltas, test_set.generated_bps)

        z = torch.randn([1, 128]).to(device)
        for i in range(1):
            recon_x = model.decoder(z, a)
            cond = recon_x > 0.4
            cond = cond.detach().cpu().numpy()

            contact_verts = F.sigmoid(recon_x).view(1,-1,13)
            cond = (contact_verts[...,1:]>.4).any(2)
            cond = cond.detach().cpu().numpy()

            contact_verts = F.sigmoid(recon_x).view(1, -1, 13)
            cond = (contact_verts[..., 1:] > .4)
            cond = cond.detach().cpu().numpy()
            m = Mesh(v=mesh_reconstruct.reshape(-1, 3), f=[], vc=np.array([0, 0, 1]))
            for label in range(1,cond.shape[-1]):
                colors = ['red','blue','green','white','pink','purple','yellow','sky blue', 'LimeGreen', 'brown', 'grey','IndianRed']
                cond_label = cond[...,label]
                key = name_to_rgb.keys()
                m.set_vertex_colors(vc=name_to_rgb[colors[label-1]], vertex_indices=np.where(cond_label == 1)[1])
            # m.show()
            mvs[0][0].set_static_meshes([m])
            raw_input('hit enter')

            list = os.listdir(model_path)  # dir is your directory path
            j = len(list)
            path_snap = os.path.join(model_path,'epoch_%d_%d.png'%(e,j))
            mvs[0][0].save_snapshot(path_snap)

def contact_vis_pose(test_set,model):
    from psbody.mesh import Mesh, MeshViewers
    mvs = MeshViewers(shape=[1,3],window_height=1600,window_width=2400)

    object_mesh_dir = '/ps/project/body_hand_object_contact/data/object_meshes/contact_meshes'
    model_path = '/ps/project/body_hand_object_contact/body_models/models/'
    body_m = smplx.create(model_path=model_path, model_type='smplx', gender='male',num_pca_comps=12, use_face_contour=False, batch_size=1,flat_hand_mean=False)
    body_m_hat = smplx.create(model_path=model_path, model_type='smplx', gender='male',num_pca_comps=12, use_face_contour=False, batch_size=1,flat_hand_mean=True)
    body_m.use_pca = False
    body_m.to(device)
    body_m_hat.to(device)

    v_template_file = os.path.join('/ps/project/body_hand_object_contact/data/subject_meshes', 'male',
                                   '191001_00158.ply')
    v_template = Mesh(filename=v_template_file).v
    body_m.v_template = torch.from_numpy(v_template.astype(np.float32)).to(device)
    body_m_hat.v_template = torch.from_numpy(v_template.astype(np.float32)).to(device)

    snap_path = makepath(os.path.join(result_dir, 'snapshot'))
    for idx in np.random.randint(0, len(test_set), 10):
        a, b = test_set.__getitem__(idx)
        object_name = test_set.frame_idx[idx]
        o_mesh = os.path.join(object_mesh_dir, '%s.ply' % object_name)
        o_model = RigidObjectModel(model_plypath=o_mesh)

        obj_pts_n, mean, scale = bpsn.unit_normalize_batch(test_set.object_fullpts[object_name].reshape(1, -1, 3),
                                                           return_scalers=True)
        object_pts, deltas = bps.convert_to_bps(obj_pts_n.reshape(1, -1, 3), test_set.generated_bps,
                                                return_deltas=True)
        mesh_reconstruct = bps.reconstruct_from_bps(deltas, test_set.generated_bps)
        mesh_reconstruct = mesh_reconstruct.reshape(-1,3)

        z = torch.randn([1, 128]).to(device)
        input,condition, vertices,joints = prepare_input(body_m,torch.from_numpy(a).view(1,-1),torch.from_numpy(b).view(1,-1))

        y_hat = model.decoder(z, condition)

        pose_hat = y_hat[:,-495:]
        contact_verts_hat = y_hat[:,:-495]
        # contact_verts_hat = contact_verts_hat.view(1,-1,13)

        pose_hat = tgm.rotation_matrix_to_angle_axis(
                    tgm.convert_points_to_homogeneous(pose_hat.contiguous().view(-1,3,3))[:,:3]).contiguous().view(input.shape[0],-1)

        global_orient = pose_hat[:, :3]
        body_pose = pose_hat[:, 3:66]
        left_hpose = pose_hat[:,75:120]
        right_hpose = pose_hat[:,120:165]

        body_m_hat.use_pca = False
        body_m_hat.flat_hand_mean = True
        output = body_m_hat(return_verts=True, return_full_pose=False,global_orient=global_orient, body_pose = body_pose, left_hand_pose= left_hpose, right_hand_pose = right_hpose)
        vertices_hat = output.vertices.detach().cpu().numpy().squeeze()
        vertices = vertices.detach().cpu().numpy().squeeze()

        # a = a.detach().cpu().numpy()
        o_model.trans[:] = a[5001:5004]-b[:3]
        o_model.pose[:] = a[5004:]

        body_mesh_hat = Mesh(v=vertices_hat, f=body_m.faces, vc=name_to_rgb['pink'])
        object_mesh_hat = Mesh(v=mesh_reconstruct, f=[], vc=name_to_rgb['blue'])

        contact = contact_verts_hat.detach().cpu().numpy() > 0.3
        print contact.shape
        contact_object = contact[:,:5000]
        contact_body = contact[:,5000:]

        # b = contact > .3
        # np.where(b[:, :, 1:] == True)[1]

        object_mesh_hat.set_vertex_colors(vc='red', vertex_indices=np.where(contact_object == True)[1])

        body_mesh = Mesh(v=vertices, f=body_m.faces, vc=name_to_rgb['pink'])
        body_mesh_hat.set_vertex_colors(vc='red', vertex_indices=np.where(contact_body == True)[1])
        body_mesh.set_vertex_colors(vc='red', vertex_indices=np.where(contact_body == True)[1])
        object_mesh = Mesh(v=o_model.r, f=o_model.f, vc=name_to_rgb['pink'])

        mvs[0][0].set_static_meshes([body_mesh] + [object_mesh], blocking=True)
        mvs[0][1].set_static_meshes([body_mesh_hat]+ [object_mesh], blocking=True)
        mvs[0][2].set_static_meshes([object_mesh_hat], blocking=True)
        raw_input("please hit enter")

        list = os.listdir(snap_path)  # dir is your directory path
        j = len(list)
        path_snap = os.path.join(snap_path,'epoch_%d_%d.png'%(e,j))
        mvs[0][0].save_snapshot(path_snap)

        # from psbody.mesh import Mesh, MeshViewers
        # mvs = MeshViewers(shape=[1,2])
        # object_mesh_dir = '/ps/project/body_hand_object_contact/data/object_meshes/contact_meshes'
        # model_file = os.path.join('/ps/project/common/moshpp/smplx/unlocked_head', 'male', 'model.pkl')
        # v_template_file = os.path.join('/ps/project/body_hand_object_contact/data/subject_meshes', 'male',
        #                                '191001_00158.ply')
        # v_template = Mesh(filename=v_template_file).v
        # smplx_model = load_smplx(fname_or_dict=model_file, ncomps= 24, v_template=v_template)



def continuous_rotation_6D(module_input):
    reshaped_input = module_input.view(-1, 3, 2)

    b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

    dot_prod = torch.sum(b1 * reshaped_input[:, :, 1],
                         dim=1, keepdim=True)
    b2 = F.normalize(reshaped_input[:, :, 1] -
                     dot_prod * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=1)

    return tgm.convert_points_to_homogeneous(torch.stack([b1, b2, b3], dim=-1))

def pointnet_train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parameters = {
    #     'network_name': '{}'.format(model._get_name()),
    #     'train_ID': '34_subject',
    #     'model_type': 'MANO',
    #     'input_dim': input_dim,
    #     'batch_size': batch_size,
    #     'hidden_dim': hidden_dim,
    #     'num_layers': num_layers,
    #     'out_dim': out_dim,
    #     'num_epochs': num_epochs,
    #     'tc_file_adrress': tc_file_address,
    #     'mosh_file_adrress': mosh_file_address,
    #     'optimizer': 'adam',
    #     'lr': lr,
    #     'loss_on': ['pose', 'joints'],  # , 'verts'], 'joints', ' pose'
    #     'fullpose': fullpose,
    #     'best_model_fname': None,
    #     'log_every_epoch': 2,
    #     'out_path': out_dir,
    #     'dtype': torch.float32,
    #     'test_size': 0.3,
    #     'visualize': False,
    #     'abs_pose': ''
    # }

    params = {'batch_size': 4,
              'shuffle': True}

    lr = 1e-3
    n_points = 1024

    model = models.PointNetDenseCls(k=13)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=2, reduction='none')

    lr_scheduler = ReduceLROnPlateau(optimizer, factor=.2, threshold=1e-5,threshold_mode='rel',min_lr=1e-8, patience=3, verbose=True)

    contact_parent_path = '/ps/scratch/body_hand_object_contact/contact_results/14_omid/191001_00158/01_thrshld_15e_6_new_labels'
    training_set = ContactDataSet(contact_parent_path, n_sample_points=n_points, train=False,num_contact_labels=32, intent='all')
    train_dataset = data.DataLoader(training_set,drop_last=True, **params)

    test_set = ContactDataSet(contact_parent_path, n_sample_points=n_points, train=False,num_contact_labels=32, intent='all')
    test_dataset = data.DataLoader(test_set,drop_last=True, **params)

    def train():
        model.train()
        for X, y in train_dataset:
            x = X[:,:3*n_points].view(4,3,-1).to(device)
            scale = X[:,3*n_points].view(-1,1).repeat(1,5000).view(-1,1,5000).to(device)
            # loc = X[:,3*n_points+1:3*n_points+4].view(-1,3).repeat(1,5000).view(-1,3,5000).to(device)
            rotation = tgm.angle_axis_to_rotation_matrix(X[:,3*n_points+4:])[:,:3,:3].to(device)
            x = torch.bmm(x.transpose(1,2),rotation).transpose(1,2)
            # pose = X[:,3*n_points+4:].view(-1,3).repeat(1,5000).view(-1,3,5000).to(device)
            # x = torch.cat([x,scale,loc,pose],1)
            x = torch.cat([x,scale],1)
            y = y[:,102:102+n_points].to(device)
            optimizer.zero_grad()
            y_hat = model(x)[0]
            y_hat = y_hat.view(-1,13)
            print y.shape
            y = y.to(torch.long)
            a = to_one_hot(y,n_dims=13).view(-1,13).to(device)

            weight = torch.tensor([1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100], dtype=torch.float32).to(device)
            # loss = F.binary_cross_entropy(y_hat, a)#,weight=weight)
            loss_fn = nn.CrossEntropyLoss(ignore_index=2, reduction='none',weight=weight)
            loss = loss_fn(y_hat,y.view(-1).to(torch.long))
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            print loss
            # print X.shape
            # print y.shape

    def eval():
        model.eval()
        for X, y in test_dataset:
            x = X[:, :3 * n_points].view(4, 3, -1).to(device)
            scale = X[:, 3 * n_points].view(-1, 1).repeat(1, 5000).view(-1, 1, 5000).to(device)
            x = torch.cat([x, scale], 1)
            y = y[:, 102:102 + n_points].to(device)

            y_hat = model(x)[0]
            y_hat = y_hat.view(-1, 13)
            y = y.view(-1).to(torch.long)
            a = to_one_hot(y).to(device)

            # loss = loss_fn(y_hat.transpose(1,2),y.to(torch.long))
            # loss = F.nll_loss(y_hat, a)
            loss = F.binary_cross_entropy(y_hat.view(-1), a.view(-1))
            # loss = torch.mean(loss)

            print loss
            # print X.shape
            # print y.shape

    for epoch in range(10):
        train()
        eval()
    print 'finished'



if __name__ == '__main__':

    # pointnet_train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params = {'batch_size': 8,
              'shuffle': True
              }
    lr = 1e-3
    n_points = 5000

    # model = VAE(encoder_layer_sizes=[5495, 1024, 512, 256], latent_size=32,
    #             decoder_layer_sizes=[256, 512, 1024, 5495], conditional=True, condition_size=5013)
    model = VAEPointnet(latent_size=32,num_classes=1,n_points=3000, condition_size=5000)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # loss_fn = nn.CrossEntropyLoss(ignore_index=2, reduction='none')

    lr_scheduler = ReduceLROnPlateau(optimizer, factor=.2, threshold=1e-5, threshold_mode='rel', min_lr=1e-8,
                                     patience=3, verbose=True)

    from experiments.omid.tools.dir_utils import makepath
    import inspect

    model_name= (model._get_name())
    input='labeled_contact'
    input=raw_input('please enter the description for the model')

    # writer = SummaryWriter()
    trained_model_dir = '/is/ps2/otaheri/contact_trained_networks'

    result_dir = makepath(os.path.join(trained_model_dir,model_name,'input_%s_lr_%4f'%(input,lr)))
    shutil.copy2(os.path.basename(sys.argv[0]), result_dir)
    shutil.copy2(inspect.getfile(ContactDataSet), result_dir)
    shutil.copy2(inspect.getfile(models), result_dir)


    # contact_parent_path = '/ps/project/body_hand_object_contact/contact_results/13_omid/190920_00174/02'
    # contact_parent_path = '/ps/project/body_hand_object_contact/contact_results/14_omid/191001_00158/01_thrshld_15e_6'
    contact_parent_path = '/ps/scratch/body_hand_object_contact/contact_results/14_omid/191001_00158/01_thrshld_15e_6_new_labels'
    # contact_parent_path = '/ps/project_cifs/body_hand_object_contact/contact_results/14_omid/191001_00158/01_thrshld_15e_6'

    training_set = ContactDataSet(contact_parent_path, n_sample_points=n_points, train=False, num_contact_labels=32, intent='all')
    train_dataset = data.DataLoader(training_set,drop_last=True, **params)

    test_set = ContactDataSet(contact_parent_path, n_sample_points=n_points, train=False,num_contact_labels=32, intent='all')
    test_dataset = data.DataLoader(test_set,drop_last=True, **params)

    model_path = '/ps/project/body_hand_object_contact/body_models/models/'
    body_model = smplx.create(model_path=model_path, model_type='smplx', gender='male', use_face_contour=False,
                              num_pca_comps=12, batch_size=train_dataset.batch_size)
    body_model.to(device)

    v_template_file = os.path.join('/ps/project/body_hand_object_contact/data/subject_meshes', 'male',
                                   '191001_00158.ply')
    v_template = Mesh(filename=v_template_file).v
    body_model.v_template = torch.from_numpy(v_template.astype(np.float32)).to(device)


    N_EPOCHS = 10

    for e in range(N_EPOCHS):

        # train_loss = train_pose(model=model, optimizer=optimizer, train_iterator=train_dataset,body_model=body_model)
        # test_loss = eval_pose(model=model, test_iterator=test_dataset,body_model=body_model)

        train_loss = train(model=model, optimizer=optimizer, train_iterator=train_dataset)
        test_loss = eval(model=model, test_iterator=test_dataset)

        lr_scheduler.step(test_loss)

        train_loss /= len(train_dataset)
        test_loss /= len(test_dataset)

        print  'Epoch %d'%e + '  Train Loss : %.2f  '%train_loss + '  Test Loss : %.2f'%test_loss
        model_path = makepath(os.path.join(result_dir, 'model_snapshot'))
        torch.save(model, os.path.join(model_path, 'model_E_%d.pth' % (e + 1)))
    print ('finished')
