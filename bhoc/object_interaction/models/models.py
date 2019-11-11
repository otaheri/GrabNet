from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from experiments.omid.object_interaction.dataset_loader import ContactDataSet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data

import basis_point_sets.bps as bps
import basis_point_sets.normalization as bpsn

from .pointnet2 import PointNet2ClsMsg
from .pointnet2 import PointNet2PartSeg_msg_one_hot
from .pointnet2 import PointNet2SemSeg

import torch
import torch.nn as nn

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 16)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        iden = np.eye(4).ravel().astype(np.float32)
        iden = torch.from_numpy(iden).view(1, 4*4).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 4, 4)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat

class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        # x = F.softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss


# from utils import idx2onehot

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)

    return onehot


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,conditional=False, condition_size=None):

        super(VAE,self).__init__()

        if conditional and (condition_size is None):
            # dimentionality of the condition
            num_labels = encoder_layer_sizes[0]
        else:
            num_labels = condition_size

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.input_size = encoder_layer_sizes[0]

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, self.input_size)

        batch_size = x.size(0)

        if c.dim()>2:
            c = c.view(-1,self.input_size)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size]).to(device)
        z = eps * std + means
        # z = means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):


        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).to(device)

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super(Encoder,self).__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += 5000

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):

            self.MLP.add_module( name="BN{:d}".format(i), module=nn.BatchNorm1d(in_size))
            self.MLP.add_module( name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            # c = idx2onehot(c, n=10)
            x = torch.cat((x, c[...,:5000]), dim=-1)
        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


# class Decoder(nn.Module):
#
#     def __init__(self, layer_sizes, latent_size, conditional, num_labels):
#
#         super(Decoder,self).__init__()
#
#         self.MLP_contact = nn.Sequential()
#         self.MLP_pose = nn.Sequential()
#
#         self.conditional = conditional
#         if self.conditional:
#             input_size = latent_size + 5000-32
#             input_size_pose = latent_size + 3-32
#         else:
#             input_size = latent_size
#
#         layer_sizes[-1] -=495
#         for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
#             self.MLP_contact.add_module(
#                 name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
#             if i+1 < len(layer_sizes):
#                 self.MLP_contact.add_module(name="A{:d}".format(i), module=nn.ReLU())
#             else:
#                 self.MLP_contact.add_module(name="sigmoid", module=nn.Sigmoid())
#         layer_sizes[-1] =495
#         for i, (in_size, out_size) in enumerate(zip([input_size_pose]+layer_sizes[:-1], layer_sizes)):
#             self.MLP_pose.add_module(
#                 name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
#             if i+1 < len(layer_sizes):
#                 self.MLP_pose.add_module(name="A{:d}".format(i), module=nn.ReLU())
#             else:
#                 self.MLP_pose.add_module(name="Tanh", module=nn.Tanh())
#
#         # self.last_layer_sig = nn.Sigmoid()
#         # self.last_layer_tanh = nn.Tanh()
#
#     def forward(self, z, c):
#
#         if self.conditional:
#             # c = idx2onehot(c, n=10)
#             # z = torch.cat((z, c), dim=-1)
#             z1 = torch.cat((z[...,:32], c[...,:5000]), dim=-1)
#             z2 = torch.cat((z[...,32:], c[...,5001:5004]), dim=-1)
#
#         x1 = self.MLP_contact(z1)
#         x2 = self.MLP_pose(z2)
#         # x = torch.cat((self.last_layer_sig(x[:,:-495]),self.last_layer_tanh(x[:,-495:])),dim=-1)
#         # x = torch.cat((x[:,:-495],self.last_layer_tanh(x[:,-495:])),dim=-1)
#         x = torch.cat((x1,x2),dim=-1)
#         return x
class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super(Decoder,self).__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="BN{:d}".format(i), module=nn.BatchNorm1d(in_size))
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            # else:
                # self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
        self.last_layer_sig = nn.Sigmoid()
        self.last_layer_tanh = nn.Tanh()

    def forward(self, z, c):

        if self.conditional:
            # c = idx2onehot(c, n=10)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)
        # x = torch.cat((self.last_layer_sig(x[:,:-495]),self.last_layer_tanh(x[:,-495:])),dim=-1)
        # x = torch.cat((x[:,:-495],self.last_layer_tanh(x[:,-495:])),dim=-1)

        return x


class VAEshape(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,conditional=False, condition_size=None):

        super(VAEshape,self).__init__()

        # if conditional and (condition_size is None):
        #     # dimentionality of the condition
        #     num_labels = encoder_layer_sizes[0]
        # else:
        #     num_labels = condition_size
        num_labels = 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.input_size = encoder_layer_sizes[0]

        self.encoder = Encodershape(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decodershape(
            decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, self.input_size)

        batch_size = x.size(0)

        # if c.dim()>2:
        #     c = c.view(-1,self.input_size)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size]).to(device)
        z = eps * std + means
        # z = means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):


        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).to(device)

        recon_x = self.decoder(z, c)

        return recon_x


class Encodershape(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super(Encodershape,self).__init__()

        # self.conditional = conditional
        # if self.conditional:
        #     layer_sizes[0] += 5000

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):

            self.MLP.add_module( name="BN{:d}".format(i), module=nn.BatchNorm1d(in_size))
            self.MLP.add_module( name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        # if self.conditional:
        #     # c = idx2onehot(c, n=10)
        #     x = torch.cat((x, c[...,:5000]), dim=-1)
        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decodershape(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super(Decodershape,self).__init__()

        self.MLP = nn.Sequential()

        # self.conditional = conditional
        # if self.conditional:
        #     input_size = latent_size + num_labels
        # else:
        input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="BN{:d}".format(i), module=nn.BatchNorm1d(in_size))
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
        self.last_layer_sig = nn.Sigmoid()
        self.last_layer_tanh = nn.Tanh()

    def forward(self, z, c):

        # if self.conditional:
        #     # c = idx2onehot(c, n=10)
        #     z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)
        # x = torch.cat((self.last_layer_sig(x[:,:-495]),self.last_layer_tanh(x[:,-495:])),dim=-1)
        # x = torch.cat((x[:,:-495],self.last_layer_tanh(x[:,-495:])),dim=-1)

        return x


class HandPoseFC(nn.Module):
    def __init__(self, input_size,layer_sizes,n_points, output_size):
        super(HandPoseFC,self).__init__()

        self.MLP = nn.Sequential()


        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes, layer_sizes+[output_size])):
            self.MLP.add_module(name="BN{:d}".format(i), module=nn.BatchNorm1d(in_size))
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
                self.MLP.add_module(name="DO{:d}".format(i), module=nn.Dropout(p=.4))
            else:
                self.MLP.add_module(name="Tanh", module=nn.Tanh())


    def forward(self, x):
        x = self.MLP(x)
        return x

class VAEPointnet(nn.Module):

    def __init__(self, latent_size,num_classes,n_points, condition_size=None):

        super(VAEPointnet,self).__init__()

        if condition_size is not None:
            self.condition_size = condition_size
        else:
            self.condition_size = 0

        # assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int


        self.latent_size = latent_size
        self.input_size = n_points

        self.encoder = EncoderPointnet(n_points,latent_size, condition_size)
        self.decoder = DecoderPointnet(latent_size, condition_size, num_classes,n_points)

    def forward(self, x, c=None):

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size]).to(device)
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).to(device)

        recon_x = self.decoder(z, c)

        return recon_x


class EncoderPointnet(nn.Module):

    def __init__(self, n_points,latent_size, condition_size):

        super(EncoderPointnet,self).__init__()

        self.n_points = n_points
        self.latent_size = latent_size
        self.condition_size = condition_size

        self.pointnet2 = PointNet2ClsMsg().to(device)

        self.linear_means = nn.Linear(40, latent_size)
        self.linear_log_var = nn.Linear(40, latent_size)

    def forward(self, x, c=None):

        x = x.transpose(1,2)
        c = c.transpose(1,2)
        # x = torch.cat((c, x), dim=-1)

        x,_ = self.pointnet2(c,x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class DecoderPointnet(nn.Module):

    def __init__(self, latent_size, condition_size, num_classes,n_points):

        super(DecoderPointnet,self).__init__()

        self.n_points = n_points
        self.latent_size = latent_size
        self.condition_size = condition_size

        self.pointnet2seg = PointNet2SemSeg(num_classes).to(device)


    def forward(self, z, c):

        z = z.unsqueeze(1).repeat([1,c.shape[1],1])
        # z = torch.cat((z, c), dim=-1)
        z = z.permute(0,2,1)
        c = c.permute(0,2,1)
        x = self.pointnet2seg(c,z)

        return x


def loss_fn(x, reconstructed_x, mean, log_var):
    # reconstruction loss
    RCL = F.binary_cross_entropy(reconstructed_x, x, size_average=False)
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return RCL + KLD

def train(model, train_iterator, optimizer):
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for i, (x, y) in enumerate(train_iterator):
        # reshape the data into [batch_size, 784]
        # x = x.view(-1, 28 * 28)

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

        if i % 100 == 0:
            print ('iteration %d' % i)
            print ('loss %.4f' % loss)

        # update the weights
        optimizer.step()

    return train_loss

def eval(model, test_iterator):
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, (x, y) in enumerate(test_iterator):
            # reshape the data
            x = x[:, :3 * n_points].to(device)
            y = y[:, 102:102 + n_points].to(device)

            # convert y into one-hot encoding
            input = y
            condition = x

            # forward pass
            reconstructed_input, z_mu, z_var = model(x, y)

            # loss
            loss = loss_fn(input, reconstructed_input, z_mu, z_var)
            test_loss += loss.item()
            if i%100==0:
                print ('iteration %d'%i)
                print ('loss .4f'%loss)

    return test_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    # sim_data = Variable(torch.rand(32,3,2500))
    # trans = STN3d()
    # out = trans(sim_data)
    # print('stn', out.size())
    # print('loss', feature_transform_regularizer(out))
    #
    # sim_data_64d = Variable(torch.rand(32, 64, 2500))
    # trans = STNkd(k=64)
    # out = trans(sim_data_64d)
    # print('stn64d', out.size())
    # print('loss', feature_transform_regularizer(out))
    #
    # pointfeat = PointNetfeat(global_feat=True)
    # out, _, _ = pointfeat(sim_data)
    # print('global feat', out.size())
    #
    # pointfeat = PointNetfeat(global_feat=False)
    # out, _, _ = pointfeat(sim_data)
    # print('point feat', out.size())
    #
    # cls = PointNetCls(k = 5)
    # out, _, _ = cls(sim_data)
    # print('class', out.size())
    #
    # seg = PointNetDenseCls(k = 3)
    # out, _, _ = seg(sim_data)
    # print('seg', out.size())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = {'batch_size': 8,
              'shuffle': True}

    lr = 1e-3
    n_points = 5000

    model = VAE(encoder_layer_sizes=[n_points, 1024, 512, 256], latent_size=128, decoder_layer_sizes=[256,512,1024,n_points],conditional=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)

    # loss_fn = nn.CrossEntropyLoss(ignore_index=2, reduction='none')

    lr_scheduler = ReduceLROnPlateau(optimizer, factor=.2, threshold=1e-5,threshold_mode='rel',min_lr=1e-8, patience=3, verbose=True)

    # contact_parent_path = '/ps/project/body_hand_object_contact/contact_results/13_omid/190920_00174/02'
    contact_parent_path = '/ps/project/body_hand_object_contact/contact_results/14_omid/191001_00158/01_thrshld_15e_6'
    # contact_parent_path = '/ps/project_cifs/body_hand_object_contact/contact_results/14_omid/191001_00158/01_thrshld_15e_6'

    training_set = ContactDataSet(contact_parent_path, n_sample_points=n_points, train=True, intent='all')
    train_dataset = data.DataLoader(training_set, **params)

    test_set = ContactDataSet(contact_parent_path, n_sample_points=n_points, train=False, intent='all')
    test_dataset = data.DataLoader(test_set, **params)

    N_EPOCHS=10

    for e in range(N_EPOCHS):
        print ('Epoch number %d'%e)
        train_loss = train(model=model, optimizer=optimizer, train_iterator=train_dataset)
        # test_loss = test(model=model, test_iterator=test_dataset)
        #
        # train_loss /= len(train_dataset)
        # test_loss /= len(test_dataset)
        #
        # # print  'Epoch f'%e + 'Train Loss : .2f'%train_loss + 'Test Loss : .2f'%test_loss

    print ('finished')


    for idx in np.random.randint(100,30000,10):
        a, b = training_set.__getitem__(idx)
        object_name = training_set.frame_idx[idx]
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)
        a = a[:n_points].view(1, -1).to(device)
        b = b[102:102 + n_points].view(1, -1).to(device)
        obj_pts_n, mean, scale = bpsn.unit_normalize_batch(training_set.object_fullpts[object_name].reshape(1, -1, 3),
                                                           return_scalers=True)
        object_pts, deltas = bps.convert_to_bps(obj_pts_n.reshape(1, -1, 3), training_set.generated_bps, return_deltas=True)
        mesh_reconstruct = bps.reconstruct_from_bps(deltas, training_set.generated_bps)
        from psbody.mesh import Mesh
        z = torch.randn([1, 128]).to(device)
        recon_x = model.decoder(z, a)
        cond = recon_x>0.4
        cond = cond.detach().cpu().numpy()
        m = Mesh(v=mesh_reconstruct.reshape(-1,3),f=[], vc=np.array([0,0,1]))
        m.set_vertex_colors(vc='red', vertex_indices=np.where(cond == 1)[1])

        m.show().background_color = np.array([0,1,0])