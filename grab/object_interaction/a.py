# import torch
# import numpy as np
# # import sys
# # sys.path.append("/ps/project/multimodalmocap/frankengeist/experiments/omid/chamfer-extension")
# # import dist_chamfer as ext
# # distChamfer = ext.chamferDist()
# #
# #
# # ptcloud1 = torch.from_numpy(np.array(
# #        [[[ 0.0000e+00,  0.0000e+00,  0.0000e+00],
# #          [-2.7037e-03, -2.8291e-03, -2.4257e-03],
# #          [ 1.3118e-03, -2.5432e-04, -4.4479e-04]],
# #         [[ 0.0000e+00,  0.0000e+00,  0.0000e+00],
# #          [-1.8294e-03, -1.8009e-03, -2.8596e-03],
# #          [-2.3429e-05, -1.2263e-04,  1.8216e-04]]])).float().cuda()
# # ptcloud2 = torch.from_numpy(np.array(
# #        [[[ 0.0769, -0.0875, -0.0054],
# #          [ 0.0769, -0.0048, -0.0054],
# #          [ 0.0769, -0.0076, -0.0160]],
# #
# #         [[ 0.0586, -0.0545,  0.0316],
# #          [ 0.0586, -0.0550, -0.0346],
# #          [ 0.0589, -0.0538,  0.0317]]])).float().cuda()
# #
# # dist3, dist4, idx1, idx2 = distChamfer(ptcloud1, ptcloud2)
# # print(torch.mean(dist3), torch.mean(dist4))
#
#
# import basis_point_sets.bps as bps
# import basis_point_sets.normalization as bpsn
# from psbody.mesh import Mesh, MeshViewers, MeshViewer
# import glob, os
#
#
# object_mesh_dir = '/ps/project/body_hand_object_contact/data/object_meshes/contact_meshes/*.ply'
# mesh_names = glob.glob(os.path.join(object_mesh_dir))
# gbps = bps.generate_bps(n_points=2500, radius=1.1)
# for meshname in mesh_names:
#     mesh = Mesh(filename=meshname)
#     mesh_points = mesh.v
#     mesh_points_normalized = bpsn.unit_normalize_batch(mesh_points.reshape(1,-1,3))
#     mesh_dist, mesh_deltas = bps.convert_to_bps(mesh_points_normalized.reshape(1,-1,3), gbps, return_deltas=True)
#     mesh_reconstruct = bps.reconstruct_from_bps(mesh_deltas, gbps)
#     re_mesh = Mesh(v=mesh_reconstruct, f=[])
#     re_mesh.show()
#     # raw_input("press a key to continue")

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
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

from experiments.omid.object_interaction.old_data_loader import ContactDataSet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from experiments.omid.object_interaction.models import models
from old_models import VAE
import basis_point_sets.bps as bps
import basis_point_sets.normalization as bpsn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_fn(x, reconstructed_x, mean, log_var):
    # reconstruction loss
    # reconstructed_x = reconstructed_x.unsqueeze(1)
    # # x = x.unsqueeze(1)
    # reconstructed_x = torch.cat([reconstructed_x,1-reconstructed_x],dim=1)
    # x = torch.cat([x,1-x],dim=1)
    # weight = torch.Tensor([1, 15]).to(device)
    RCL = F.binary_cross_entropy(reconstructed_x, x,size_average=False)
    # lossfn = nn.CrossEntropyLoss(weight=weight,reduction='none')
    # RCL = lossfn(reconstructed_x, x.to(torch.long)).mean()
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return RCL + KLD

def train(model, train_iterator, optimizer):
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for i, (x, y) in enumerate(train_iterator):


        x = x[:, :n_points].to(device)
        y = y[:, 102:102 + n_points].to(device)

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
            x = x[:, :n_points].to(device)
            y = y[:, 102:102 + n_points].to(device)

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




if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = {'batch_size': 8,
              'shuffle': True}

    lr = 1e-3
    n_points = 5000

    model = VAE(encoder_layer_sizes=[n_points,1024, 512, 256], latent_size=128,
                decoder_layer_sizes=[256, 512,1024, n_points], conditional=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # loss_fn = nn.CrossEntropyLoss(ignore_index=2, reduction='none')

    lr_scheduler = ReduceLROnPlateau(optimizer, factor=.2, threshold=1e-5, threshold_mode='rel', min_lr=1e-8,
                                     patience=3, verbose=True)

    # contact_parent_path = '/ps/project/body_hand_object_contact/contact_results/13_omid/190920_00174/02'
    contact_parent_path = '/ps/project/body_hand_object_contact/contact_results/14_omid/191001_00158/01_thrshld_15e_6'
    # contact_parent_path = '/ps/project_cifs/body_hand_object_contact/contact_results/14_omid/191001_00158/01_thrshld_15e_6'

    training_set = ContactDataSet(contact_parent_path, n_sample_points=n_points, train=True, intent='all')
    train_dataset = data.DataLoader(training_set, **params)

    test_set = ContactDataSet(contact_parent_path, n_sample_points=n_points, train=False, intent='all')
    test_dataset = data.DataLoader(test_set, **params)

    N_EPOCHS = 10

    for e in range(N_EPOCHS):
        train_loss = train(model=model, optimizer=optimizer, train_iterator=train_dataset)
        test_loss = eval(model=model, test_iterator=test_dataset)
        lr_scheduler.step(test_loss)

        train_loss /= len(train_dataset)
        test_loss /= len(test_dataset)

        torch.save(model,'/ps/project/body_hand_object_contact/models/model_epoch_%d.pt'%e)
        print  'Epoch %d'%e + '  Train Loss : %.2f  '%train_loss + '  Test Loss : %.2f'%test_loss
    print ('finished')

    for idx in np.random.randint(100, 18000, 10):
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
        from psbody.mesh import Mesh

        z = torch.randn([1, 128]).to(device)
        recon_x = model.decoder(z, a)
        cond = recon_x > 0.4
        cond = cond.detach().cpu().numpy()
        m = Mesh(v=mesh_reconstruct.reshape(-1, 3), f=[], vc=np.array([0, 0, 1]))
        m.set_vertex_colors(vc='red', vertex_indices=np.where(cond == 1)[1])
        m.show()
