

# import open3d as o3d

import numpy as np
import torch
import torch.nn as nn
from psbody.mesh import Mesh, MeshViewers, MeshViewer
from experiments.nima.tools.visualization_helpers import spheres_for
from experiments.nima.tools_torch.optimizers.lbfgs_ls import LBFGS as LBFGSLs
from experiments.nima.tools.mocap_interface import MocapSession

import torchgeometry as tgm
import copy

import sys
sys.path.append("/is/ps2/otaheri/frankengeist/experiments/omid/chamfer-extension")
import dist_chamfer as chdist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RigidObjectModel(nn.Module):

    def __init__(self,
                model_plypath,
                params =None,
                dtype = torch.float32):

        super(RigidObjectModel, self).__init__()

        '''
        
        '''
        # Todo:

        self.dtype = dtype
        from psbody.mesh import Mesh
        rigid_mesh = Mesh(filename=model_plypath)

        self.register_buffer('f', torch.tensor(rigid_mesh.f.astype(np.int32), dtype=torch.int32))
        self.register_buffer('v', torch.tensor(rigid_mesh.v.astype(np.float32)*.001, dtype=dtype))

        trans = torch.tensor(np.zeros((3)), dtype=dtype, requires_grad = True)
        self.register_parameter('trans', nn.Parameter(trans, requires_grad=True))

        root_orient = torch.tensor(np.zeros((3)), dtype=dtype, requires_grad=True)
        self.register_parameter('root_orient', nn.Parameter(root_orient, requires_grad=True))

        markers_loc = torch.tensor(np.zeros((6,3)), dtype=dtype, requires_grad=True)
        self.register_parameter('markers_loc', nn.Parameter(markers_loc, requires_grad=True))



    def forward(self, root_orient = None, trans = None, **kwargs):

        if root_orient is None:
            root_orient = self.root_orient
        if trans is None:
            trans = self.trans

        # verts = torch.matmul(self.v, tgm.angle_axis_to_rotation_matrix(root_orient.view(1,-1))[:,:3,:3]) + trans
        verts = torch.matmul(self.v, batch_rodrigues(root_orient.view(1, -1))[:, :3, :3]) + trans
        # verts = torch.matmul(self.v, root_orient.transpose(1,0)) + trans

        self.verts = verts

        return verts
    def show(self):
        self.get_mesh().show()

    def get_mesh(self, vc=None,vscale=None):
        with torch.no_grad():
            verts = self.forward()
            mesh = Mesh(v=verts.view(-1,3).cpu().numpy(), f=self.f.view(-1,3).cpu().numpy(), vc=vc,vscale=vscale)
        return mesh

    def get_optimized_marker_loc(self, marker_optimized_loc):
        root_orient = self.root_orient
        trans = self.trans
        verts = torch.matmul(marker_optimized_loc, batch_rodrigues(root_orient.view(1, -1))[:, :3, :3]) + trans
        return verts

from experiments.nima.tools_torch.lbs import batch_rodrigues

def get_marker_data(mocap_fname, markers=None, dtype=torch.float32):

    mp = MocapSession(mocap_fname=mocap_fname, units='mm')
    if markers is None:
        markers = mp.marker_labels()
    data = np.zeros([mp._markers_per_frame.shape[0], len(markers), 3])
    for fidx, frame in enumerate(mp.markers_asdict()):
        data[fidx, :, :] = np.asarray([frame[k] for k in markers]).reshape((-1, 3))
    data = torch.tensor(data, dtype = dtype)
    return data

def frame_picker(data=None,markers=None,num_frames = 20):
        # if markers is None:
        #     markers = self.markers

        idxs_to_shuffle = range(data.shape[0])
        np.random.shuffle(idxs_to_shuffle)

        picked_frames = []

        for idx in idxs_to_shuffle:
            frame = data[idx]
            nan_checker = (frame==frame).sum()

            if nan_checker==(frame.shape[0]*frame.shape[1]):
                picked_frames.append(frame)
            if len(picked_frames)>= num_frames:
                break

        return torch.stack(picked_frames)

# class MocapMarkerData:
#
#     def __init__(self,
#                 mocap_fname,
#                 markers =None,
#                 dtype = torch.float32):
#
#         super(MocapMarkerData, self).__init__()
#
#         self.mocap_fname = mocap_fname
#
#         mp = MocapSession(mocap_fname=mocap_fname, units='mm')
#         if markers is None:
#             markers = mp.marker_labels()
#
#         self.markers = markers
#
#         data = np.zeros([mp._markers_per_frame.shape[0], len(markers), 3])
#         for fidx, frame in enumerate(mp.markers_asdict()):
#             data[fidx, :, :] = np.asarray([frame[k] for k in markers]).reshape((-1, 3))
#         data = torch.tensor(data, dtype = dtype)
#         self.data = data
#
#
#     def frame_picker(self,markers=None,num_frames = 20):
#         if markers is None:
#             markers = self.markers
#
#         idxs_to_shuffle = range(self.data.shape[0])
#         np.random.shuffle(idxs_to_shuffle)
#
#         data = self.data
#
#         picked_frames = []
#
#         for idx in idxs_to_shuffle:
#             frame = data[idx]
#             nan_checker = (frame==frame).sum()
#
#             if nan_checker==(frame.shape[0]*frame.shape[1]):
#                 picked_frames.append(frame)
#             if len(picked_frames)>= num_frames:
#                 break
#
#         return torch.stack(picked_frames)





def object_mesh_fitting(optimizer, rigid_mesh, markers_data,missing_markers,label2vertices, on_step=None, create_graph=True, gstep=0):

    def fit(backward = True, visualize=True):

        fit.gstep += 1
        if backward:
            optimizer.zero_grad()

        verts = rigid_mesh.forward()

        loss_total = torch.mean(torch.pow((markers_data[missing_markers] - verts.view(-1, 3)[label2vertices][missing_markers]),2))
        # loss_total = torch.mean(torch.sqrt(torch.einsum('ij->i',torch.pow((markers_data[missing_markers] - verts.view(-1, 3)[label2vertices][missing_markers]), 2))))

        if backward:
            loss_total.backward(create_graph=create_graph,retain_graph=True)

        if on_step is not None and visualize:
            # message = 'it %.5d -- %s' % (fit.gstep, ' | '.join(['%s = %2.2e' % (k, np.sum(v)) for k, v in opt_objs.iteritems()]))
            message = 'it %.5d --' % (fit.gstep)
            print(message)
            on_step(object_mesh= rigid_mesh.get_mesh(),label2vertices=label2vertices)

        return loss_total
    fit.gstep = gstep
    return fit

def load_multi_rigid_objects(model_filename, num_models=20):
    rigid_mesh = RigidObjectModel(model_plypath=model_plypath).to(device)
    multi_models = []
    for idx_model in range(num_models):
        multi_models.append(RigidObjectModel(model_plypath=model_plypath).to(device))
    return multi_models, rigid_mesh


def marker_location_optimization(optimizer, static_model, multi_models,weights, markers_data,label2vertices, on_step=None, create_graph=True, gstep=0):

    def fit(backward = True, visualize=True):

        fit.gstep += 1
        if backward:
            optimizer.zero_grad()


        object_pcld = static_model.forward().to(device)
        markers_pcld = (object_pcld.view(-1,3)[label2vertices] + static_model.markers_loc.to(device)).to(device).view(1,-1,3)

        distChamfer = chdist.chamferDist()
        obj2mrk, mrk2obj, idx_o2m, idx_m2o = distChamfer(object_pcld, markers_pcld)

        loss_keep_marker_on_surface = mrk2obj.mean()
        loss = []
        for i, model in enumerate(multi_models):
            missing_markers = (markers_data[i]==markers_data[i])[:,0]
            loss_model = torch.mean(torch.pow((markers_data[i][missing_markers] - model.get_optimized_marker_loc(markers_pcld).view(-1, 3)[missing_markers]),2))
            loss.append(loss_model)

        loss = torch.stack(loss).mean()
        loss_keep_init_loc = torch.abs(static_model.markers_loc.mean())

        # loss_total = torch.mean(torch.sqrt(torch.einsum('ij->i',torch.pow((markers_data[missing_markers] - verts.view(-1, 3)[label2vertices][missing_markers]), 2))))
        if backward:
            loss.backward(create_graph=create_graph,retain_graph=True)
            loss_keep_init_loc.backward(create_graph=create_graph,retain_graph=True)

        loss_total = weights[0]*loss + weights[1]*loss_keep_init_loc + weights[2]*loss_keep_marker_on_surface
        if on_step is not None and visualize:
            # message = 'it %.5d -- %s' % (fit.gstep, ' | '.join(['%s = %2.2e' % (k, np.sum(v)) for k, v in opt_objs.iteritems()]))
            message = 'it %.5d --' % (fit.gstep)
            print(message)
            on_step(models=multi_models, picked_frames=markers_data,rigid_mesh= static_model.get_mesh(), orig_markers=object_pcld.view(-1,3)[label2vertices].cpu(), optimized_markers=markers_pcld.view(-1,3).cpu())

        # print 'it %.5d --' % (fit.gstep)
        # print loss_total

        return loss_total
    fit.gstep = gstep
    return fit



def vis_object(markers,i,mvs=None):
    if mvs is None: mvs = MeshViewers(window_width=2000, window_height=2000, shape=[1, 1])
    print 'iteration %d'%i

    def on_step(object_mesh,label2vertices):

        verts_markers = object_mesh.v.reshape(-1,3)[label2vertices]
        orig_spheres = spheres_for(markers.numpy(), radius=0.0016, vc=np.array((0., 0., 1.)))  # blue
        vert_spheres = spheres_for(verts_markers, radius=0.0016, vc=np.array((1., 0., 0.)))  # r
        # line_v_idxs2 = np.arange(len(l_hand))

        mvs[0][0].set_static_meshes([object_mesh] + orig_spheres+vert_spheres, blocking=True)

    return on_step


def vis_objects(mvs=None):
    if mvs is None:
        mvs = MeshViewers(window_width=3000, window_height=2000, shape=[4, 5])
        mvs1 = MeshViewers(window_width=400, window_height=500, shape=[1, 1])

    def on_step(models,picked_frames,rigid_mesh,orig_markers,optimized_markers):

        with torch.no_grad():
            for counter, model in enumerate(models):
                mesh = model.get_mesh()
                mocap_spheres = spheres_for(picked_frames[counter].cpu().numpy(), radius=0.0016, vc=np.array((0., 0., 1.)))
                mvs[counter/5][counter%5].set_static_meshes([mesh]+mocap_spheres, blocking=True)
            orig_spheres = spheres_for(orig_markers.detach().numpy(), radius=0.0016, vc=np.array((1., 0., 0.)))
            optimized_spheres = spheres_for(optimized_markers.detach().numpy(), radius=0.0016, vc=np.array((0., 0., 1.)))
            mvs1[0][0].set_static_meshes([rigid_mesh] + orig_spheres+optimized_spheres, blocking=True)

    return on_step



if __name__ == '__main__':

    # model_plypath='/home/otaheri/Desktop/Dice_mesh/knife_beab.ply'
    model_plypath='/home/otaheri/Desktop/Dice_mesh/flute_beab.ply'
    # model_plypath='/home/otaheri/Desktop/Dice_mesh/ball.ply'
    dtype = torch.float32


    rigid_mesh = RigidObjectModel(model_plypath=model_plypath).to(device)
    my_mesh = Mesh(filename=model_plypath)


    # mocap_fname = '/ps/project/vicondata/MasonIOI/20190828 Mason/labeled exports/pick knife 2.c3d'
    mocap_fname = '/ps/project/vicondata/MasonIOI/20190828 Mason/labeled exports/pick flute 1_both.c3d'
    mocap_fname = '/ps/data/IMU2Hands/IMU2Hands_190828_OT/labeled exports/pick flute 1_both.c3d'
    # mocap_fname = '/ps/project/nansense/IMU2Hands_190828_OT/labeled exports/kick_ball_full.c3d'
    # markers = {  'knife1':50224,
    #              'knife2':199123,
    #              'knife3':105937,
    #              'knife4':182837,
    #              'knife5':265727,
    #              'knife6':149185}

    markers = {'flute1': 152216,
               'flute2': 111086,
               'flute3': 58957,
               'flute4': 26626,
               'flute5': 13512,
               'flute6': 163}

    markers = {'flute1': 152216,
               'flute2': 111086,
               'flute3': 58957,
               'flute4': 26626,
               'flute5': 13512,
               'flute6': 363}


    # markers = {'ball1': 17314,
    #            'ball2': 9142,
    #            'ball3': 23586,
    #            'ball4': 30971,
    #            'ball5': 36425,
    #            'ball6': 30586,
    #            'ball7': 9469,
    #            'ball8': 23704,
    #            'ball9': 20237,
    #            'ball10': 20379,
    #            'ball11': 4191,
    #            'ball12': 16913
    #            }

    # marker_data = get_marker_data(mocap_fname=mocap_fname, markers=markers.keys(), dtype=dtype)
    marker_data = get_marker_data(mocap_fname=mocap_fname, markers=markers.keys(), dtype=dtype).to(device)
    # mp = MocapSession(mocap_fname=mocap_fname, units='mm')



    ########### optimize for marker position on the rigid mesh ###################
    multi_model, static_model = load_multi_rigid_objects(model_filename=model_plypath)
    picked_frames = frame_picker(data=marker_data)


    free_vars = [model.root_orient for model in multi_model]+[model.trans for model in multi_model]
    optimizer = LBFGSLs(free_vars, lr=1e-0, max_iter=5, line_search_fn='strong_Wolfe', tolerance_change=1e-15, tolerance_grad=1e-10)

    w = [1,0,0]

    on_step = vis_objects()
    closure = marker_location_optimization(optimizer=optimizer, static_model=static_model, multi_models=multi_model,weights=w, markers_data=picked_frames, label2vertices=markers.values(),on_step=on_step)
    optimizer.step(closure)

    free_vars = [model.root_orient for model in multi_model]+[model.trans for model in multi_model] + [static_model.markers_loc]
    optimizer = LBFGSLs(free_vars, lr=1e-0, max_iter=300, line_search_fn='strong_Wolfe', tolerance_change=1e-15, tolerance_grad=1e-10)

    w = [10000,0,5]
    on_step = vis_objects()
    closure = marker_location_optimization(optimizer=optimizer, static_model=static_model, multi_models=multi_model,weights=w, markers_data=picked_frames, label2vertices=markers.values(),on_step=on_step)
    optimizer.step(closure)

    ##############################################################################

    free_vars = [rigid_mesh.root_orient, rigid_mesh.trans]
    optimizer = LBFGSLs(free_vars, lr=1e-0, max_iter=1500, line_search_fn='strong_Wolfe', tolerance_change=1e-15, tolerance_grad=1e-10)

    object_poses = []

    mvs = MeshViewers(window_width=2000, window_height=800, shape=[1, 2])

    for i in range(marker_data.shape[0]):
        # i = i+500
        object_markers = marker_data[i]
        missing_markers = (object_markers==object_markers)[:,0]

        on_step = vis_object(markers=object_markers[missing_markers],i = i,mvs=mvs)
        # on_step = None

        optimizer.zero_grad()
        gstep = 0

        closure = object_mesh_fitting(optimizer= optimizer, rigid_mesh=rigid_mesh ,markers_data= object_markers, missing_markers=missing_markers, label2vertices= markers.values(), on_step=on_step, gstep=gstep)
        optimizer.step(closure)

        gstep = closure.gstep

        if False:
            with torch.no_grad():
                verts_orig = rigid_mesh.forward().cpu().view(-1,3).numpy()

            points_mocap = object_markers[missing_markers]

            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(verts_orig)

            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(points_mocap)

            threshold = 0.02
            trans_init = np.eye(4)

            reg_p2p = o3d.registration.registration_icp(
                    pcd2, pcd1, threshold, trans_init,
                    o3d.registration.TransformationEstimationPointToPlane())

            R = reg_p2p.transformation

            verts = np.asarray(pcd1.transform(R).points).reshape(-1,3)
            verts_markers = verts[markers.values()]

            orig_spheres = spheres_for(points_mocap.numpy(), radius=0.003, vc=np.array((0., 0., 1.)))  # blue
            vert_spheres = spheres_for(verts_markers, radius=0.003, vc=np.array((1., 0., 0.)))  # blue
            # line_v_idxs2 = np.arange(len(l_hand))

            mesh = Mesh(v=verts, f=rigid_mesh.f)

            mvs[0][1].set_static_meshes([mesh] + orig_spheres+vert_spheres, blocking=True)


        object_poses.append([rigid_mesh.root_orient.clone(), rigid_mesh.trans.clone()])
    torch.save(object_poses,'/home/otaheri/Desktop/flute_poses.pt')


