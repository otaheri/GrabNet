import numpy as np
import torch
import torch.nn as nn
from psbody.mesh import Mesh, MeshViewers, MeshViewer
from experiments.nima.tools.visualization_helpers import spheres_for
from experiments.nima.tools_torch.optimizers.lbfgs_ls import LBFGS as LBFGSLs
from experiments.nima.tools.mocap_interface import MocapSession

from copy import deepcopy




from fit_rigid_object2mocap import RigidObjectModel
import trimesh
import networkx as nx
import os
import time
import torch.autograd as autograd

import sys
sys.path.append('/is/ps2/otaheri/frankengeist/experiments/vchoutas/pytorch_mesh_self_isect')
from mesh_intersection.bvh_search_tree import BVH
sys.path.append("/is/ps2/otaheri/frankengeist/experiments/omid/chamfer-extension")
import dist_chamfer as chdist


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def point2point(source_mesh, target_mesh, contact_threshold, contact_method='dist', dtype=torch.float32):
    '''
    :param source_mesh:
    :param target_mesh:
    :param contact_method:
    :param contact_method:
    :return:
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source_verts = torch.from_numpy(source_mesh.v.astype(np.float32)).to(device)
    target_verts = torch.from_numpy(target_mesh.v.astype(np.float32)).to(device)

    # source_mesh = deepcopy(source_mesh)
    # target_mesh = deepcopy(target_mesh)

    # source_f = torch.tensor(source_mesh.f.astype(np.int32), dtype=torch.int32)
    # target_f = torch.tensor(target_mesh.f.astype(np.int32), dtype=torch.int32)

    distChamfer = chdist.chamferDist()

    src2trgt, trgt2src, src2trgt_idx, trgt2src_idx = distChamfer(source_verts.view(1,-1,3).to(device),target_verts.view(1,-1,3).to(device))

    source_normals = torch.from_numpy(source_mesh.estimate_vertex_normals().astype(np.float32)).to(device)

    source2target_correspond = source_verts[trgt2src_idx.data.view(-1).long()]
    distance_vector = target_verts - source2target_correspond
    source2target_normal_correspondence = source_normals[trgt2src_idx.data.view(-1).long()]

    in_out = torch.bmm(source2target_normal_correspondence.view(-1,1,3),distance_vector.view(-1,3,1)).view(-1).sign()

    if contact_method=='in_out': trgt2src = trgt2src*in_out
    trgt2src_in_out = trgt2src * in_out
    # src2trgt=src2trgt*in_out[src2trgt_idx.long()]



    src_verts_contact_idx = np.where(src2trgt.data.view(-1).cpu().numpy()<=contact_threshold)[0]
    # src_verts_contact = source_verts[:,src_verts_contact_idx]

    trgt_verts_contact_idx = np.where(trgt2src.data.view(-1).cpu().numpy() <= contact_threshold)[0]
    trgt_verts_contact_idx_in_out = np.where(trgt2src_in_out.data.view(-1).cpu().numpy() <= contact_threshold)[0]
    # trgt_verts_contact = target_verts[:, trgt_verts_contact_idx]

    # if not hasattr(source_mesh, 'vc'):
    #     source_mesh.vc = source_mesh.colors_like(color='pink')
    # source_mesh.set_vertex_colors(vc='red', vertex_indices=src_verts_contact_idx)
    #
    # if not hasattr(target_mesh, 'vc'):
    #     target_mesh.vc = target_mesh.colors_like(color='pink')
    # target_mesh.set_vertex_colors(vc='red', vertex_indices=trgt_verts_contact_idx)

    return src_verts_contact_idx, trgt_verts_contact_idx, trgt_verts_contact_idx_in_out, trgt2src_idx


def get_first_contact_surface(body,object, object_penetrating_verts=None, max_collisions = 5, batch_size = 1, contact_threshold=1e-6, show_collision=True):


    # object = Mesh(filename='/home/otaheri/Desktop/Dice_mesh/flute_beab_new.ply')
    # body = Mesh(filename='/home/otaheri/Desktop/Dice_mesh/knife_beab_new.ply')

    new_object = deepcopy(object)
    new_body = deepcopy(body)

    # object = deepcopy(object)
    # body = deepcopy(body)
    #
    # if not hasattr(new_object, 'vc'):
    #     new_object.vc = new_object.colors_like(color='pink')
    #     object.vc = object.colors_like(color='pink')
    #
    # if not hasattr(new_body, 'vc'):
    #     new_body.vc = new_body.colors_like(color='pink')
    #     body.vc = body.colors_like(color='pink')


    ####### concatenating meshes to be able to use collisions terms #############
    new_object.concatenate_mesh(new_body)
    new_object.vn = new_object.estimate_vertex_normals()
    new_object.vc = new_object.colors_like(color='pink')

    vertices = torch.tensor(new_object.v, dtype=torch.float32, device=device)
    faces = torch.tensor(new_object.f.astype(np.int64), dtype=torch.long,
                         device=device)

    # new_body.concatenate_mesh(new_object)
    # new_body.vn = new_body.estimate_vertex_normals()
    # new_body.vc = new_body.colors_like(color='pink')
    #
    # vertices = torch.tensor(new_body.v, dtype=torch.float32, device=device)
    # faces = torch.tensor(new_body.f.astype(np.int64), dtype=torch.long,
    #                      device=device)

    triangles = vertices[faces].unsqueeze(dim=0)

    m = BVH(max_collisions=max_collisions)

    torch.cuda.synchronize()
    start = time.time()
    outputs = m(triangles)
    torch.cuda.synchronize()
    print('Elapsed time for collision detection', time.time() - start)

    outputs = outputs.detach().cpu().numpy().squeeze()

    collisions_self = outputs[outputs[:, 0] >= 0, :]

    object_collision = collisions_self[(collisions_self < object.f.shape).any(axis=1)]

    ##### check only for body and object collision, not object and object or body and body ########
    a = object_collision<object.f.shape[0]
    a = np.logical_xor(a[:,0], a[:,1]).astype(np.int32)
    object_collision = object_collision[np.where(a==1)]

    number_of_colsn = object_collision.shape[0]

    print('Number of body object collisions = ', number_of_colsn)

    if number_of_colsn>0:

        ############ detaching two meshes >>> change the face index #################

        collisions_meshes = deepcopy(object_collision)
        collisions_meshes[:, 1] = collisions_meshes[:, 1] - object.f.shape[0]


        object_colliding_faces = object.f[collisions_meshes[:, 0]]
        object_colliding_verts = object.f[collisions_meshes[:, 0]].reshape(-1)
        object_colliding_edges = np.vstack(
            [object_colliding_faces[:, :2], object_colliding_faces[:, 1:], object_colliding_faces[:, [0, 2]]])

        body_colliding_faces = body.f[collisions_meshes[:, 1]]
        body_colliding_verts = body.f[collisions_meshes[:, 1]].reshape(-1)
        # body_colliding_edges = np.vstack(
        #     [body_colliding_faces[:, :2], body_colliding_faces[:, 1:], body_colliding_faces[:, [0, 2]]])


        ########### to show collision vertices ###########

        # object.set_vertex_colors(vc='green', vertex_indices=object_colliding_verts)
        # body.set_vertex_colors(vc='green', vertex_indices=body_colliding_verts)

        ###### creating a graph for intersecting edges to find number of collisions #########
        # G = nx.Graph()
        # G.add_edges_from(body_colliding_edges)
        #
        # intersections_body = []
        # for c in nx.connected_components(G):
        #     intersections_body.append(c)

        ########## Since object mesh resolution is much higher than body, it its better to show number of collisions on object #######################

        G = nx.Graph()
        G.add_edges_from(object_colliding_edges)

        intersections_object = []
        for c in nx.connected_components(G):
            intersections_object.append(c)

        intersections_body = [set(body_colliding_verts[np.where(np.in1d(object_colliding_verts, np.asarray(list(intersections_object[idx]))))]) for
                                idx in range(len(intersections_object))]

        ###### check if there are more than one intersection on the body, if so, check which one is the first one, if not, skip  >>>>>> NO, dont skip !!!! #####
        # if len(intersections_body)>1:

        # object.set_vertex_colors(vc='green', vertex_indices=np.asarray(list(intersections_object[0])))
        # object.show()
        # new_object.write_ply('/home/otaheri/Desktop/name.ply')

        body_edges = np.unique(np.vstack([body.f[:, :2], body.f[:, 1:], body.f[:, [0, 2]]]),axis=0)

        ######### for each intersection on the body, find the subset triangles(faces) to find the first intersection/s. ( the first intersection is the one which is a superset of others) ###########
        body_subset_faces_list = []

        # body_mesh_graph = nx.Graph()
        # body_mesh_graph.add_edges_from(body_edges)
        # num_body_parts = nx.number_connected_components(body_mesh_graph)

        num_body_parts = 3  # --> we can ignore eyes and use the rest of the body since human barely touch their eyes with objects ###########

        s = time.time()
        for intersection in intersections_body:
            is_in_mesh = np.zeros(body_edges.shape)

            for node in intersection:
                is_in_mesh = np.logical_or((body_edges == node), is_in_mesh).astype(np.int32)
            is_in_mesh = is_in_mesh.all(axis=1).astype(np.int32)

            keeped_edges = body_edges[np.where(is_in_mesh == 0)]
            # removed_nodes = set(body_edges[np.where(is_in_mesh == 1)].reshape(-1))
            body_mesh_graph = nx.Graph()
            body_mesh_graph.add_edges_from(keeped_edges)

            ###### check if the intersection divide the body into two parts ############

            body_parts = [c for c in nx.connected_components(body_mesh_graph)]
            if len(body_parts) > num_body_parts:
                # if len(body_parts) > num_body_parts+1:
                    for c in sorted(body_parts, key=len, reverse=True)[num_body_parts:]:
                        intersection=intersection.union(c)
                    body_subset_faces_list.append(intersection)
                # else:
                #     body_subset_faces_list.append(intersection.union(min(body_parts, key=len)))
            else:
                body_subset_faces_list.append(set(intersection))

        print time.time() - s

        ########### To find number of first intersections, it is possible that we have two fingers peneterating --> two areas of first peneteration   ###################
        sorted_index = sorted(range(len(body_subset_faces_list)), key=lambda k: len(body_subset_faces_list[k]),
                              reverse=True)
        body_subset_faces_list = sorted(body_subset_faces_list, key=len, reverse=True)

        #### Remove the intersections which are subset of other intersections ##############

        superset_flag = np.ones(len(body_subset_faces_list))
        for i in range(len(body_subset_faces_list)):
            flag = superset_flag[i]
            if flag:
                for j in range(i + 1, len(body_subset_faces_list)):
                    if body_subset_faces_list[i].issuperset(body_subset_faces_list[j]):
                        superset_flag[j] = 0

        body_first_intersection_idx = np.where(superset_flag == 1)[0]

        ############## to find corresponding intersections to the body intersections on the object  ##############
        # intersections_object = [object_colliding_verts[np.where(np.in1d(body_colliding_verts, np.asarray(list(intersections_body[sorted_index[idx]]))))] for idx in body_first_intersection_idx]

        intersections_object = [intersections_object[sorted_index[idx]] for idx in body_first_intersection_idx]


        object_edges = np.unique(np.vstack([object.f[:, :2], object.f[:, 1:], object.f[:, [0, 2]]]),axis=0)

        object_subset_faces_list = []

        s = time.time()
        intersections_object_keep = []
        for iIdx,intersection in enumerate(intersections_object):
            is_in_mesh = np.zeros(object_edges.shape)

            for node in intersection:
                is_in_mesh = np.logical_or((object_edges == node), is_in_mesh).astype(np.int32)
            is_in_mesh = is_in_mesh.all(axis=1).astype(np.int32)

            keeped_edges = object_edges[np.where(is_in_mesh == 0)]
            # removed_nodes = set(object_edges[np.where(is_in_mesh == 0)].reshape(-1))
            object_mesh_graph = nx.Graph()
            object_mesh_graph.add_edges_from(keeped_edges)

            ###### check if the intersection divide the body into two parts ############

            object_parts = [c for c in nx.connected_components(object_mesh_graph)]
            if len(object_parts) > 1:
                # if len(object_parts) > 2:
                    for c in sorted(object_parts, key=len, reverse=True)[1:]:
                        intersection=intersection.union(c)
                    object_subset_faces_list.append(intersection.union(sorted(object_parts, key=len, reverse=True)[1]))
                    intersections_object_keep.append(iIdx)
                    # object_subset_faces_list.append(sorted(object_parts, key=len, reverse=True)[1])
                # else:
                #     object_subset_faces_list.append(intersection.union(min(object_parts, key=len))) #.intersection(set(object_penetrating_verts)))
            else:
                object_subset_faces_list.append(set(intersection))
        print time.time() - s




        ###############

        contact_vrts_b = np.hstack([np.asarray(
            list(body_subset_faces_list[idx].union(intersections_body[idx]))) for idx in body_first_intersection_idx])

        contact_vrts_o = np.hstack([np.asarray(
            list(object_subset_faces_list[idx])) for idx in range(len(intersections_object))])

        contact_vrts_b_dist,  contact_vrts_o_dist, contact_vrts_o_dist_in_out, o2b_vrts_correspondence = point2point(body, object, contact_method='dist', contact_threshold=contact_threshold)

        ##### in case the whole body of the object is peneterating in the body, e.g. eyeglasses handle
        # --> this needs to be fixed, in case one finger is peneterating but another one is not,
        # this method ignores the other one as well  ############
        if contact_vrts_o.shape[0]>contact_vrts_o_dist_in_out.shape[0]:
            contact_vrts_o = contact_vrts_o_dist_in_out

        ############### to prevent adding threshold to the peneterating parts ######################
        body_subset_faces_list = [body_subset_faces_list[i] for i in intersections_object_keep]
        o2b_vrts_correspondence = o2b_vrts_correspondence.view(-1).cpu().numpy()
        if len(body_subset_faces_list)>0:
            all_body_contact_vertices = np.asarray(list(set.union(*body_subset_faces_list)))
            o2b_contact = o2b_vrts_correspondence[contact_vrts_o_dist_in_out]
            contact_vrts_o_dist = contact_vrts_o_dist_in_out[~np.isin(o2b_contact,all_body_contact_vertices)] # this ignore the vertices which are in contact with peneteration fingers

        ###########################################################################################

        contact_vertices_object = np.asarray(list(set(contact_vrts_o).union(set(contact_vrts_o_dist))))
        contact_vertices_body = np.asarray(list(set(contact_vrts_b).union(set(contact_vrts_b_dist))))
        ############################################################################################
        # to label each intersection with the finger that it has contact with
        vertex_label_contact = np.load(
            '/ps/scratch/body_hand_object_contact/contact_results/vertex_label_contact.npy').astype(np.int8)

        contact_labels_object_vrts = np.zeros(object.v.shape[0],dtype=np.int8)
        contact_labels_body_vrts = np.zeros(body.v.shape[0],dtype=np.int8)

        contact_labels_object_vrts[contact_vertices_object] = vertex_label_contact[o2b_vrts_correspondence[contact_vertices_object]]
        contact_labels_body_vrts[contact_vertices_body]   = vertex_label_contact[contact_vertices_body]

    else:
        contact_vertices_body,  contact_vertices_object, contact_vertices_object_in_out, o2b_vrts_correspondence = point2point(body, object, contact_method='dist', contact_threshold=contact_threshold)
        vertex_label_contact = np.load(
            '/ps/scratch/body_hand_object_contact/contact_results/vertex_label_contact.npy').astype(np.int8)
        o2b_vrts_correspondence = o2b_vrts_correspondence.view(-1).cpu().numpy()

        contact_labels_object_vrts = np.zeros(object.v.shape[0],dtype=np.int8)
        contact_labels_body_vrts = np.zeros(body.v.shape[0],dtype=np.int8)

        contact_labels_object_vrts[contact_vertices_object] = vertex_label_contact[o2b_vrts_correspondence[contact_vertices_object]]
        contact_labels_body_vrts[contact_vertices_body]   = vertex_label_contact[contact_vertices_body]
    # return contact_vertices_body, contact_vertices_object
    return contact_labels_body_vrts, contact_labels_object_vrts




def get_peneterating_triangles():
    pass

def get_visible_triangles():
    pass


def degree2radian(degree):
    return (degree*np.pi)/180.0

# for triangle i, the indices of three neighbouring triangles are stored in f_a_mat[i][0:3]
def build_triangle_adjacency_mat(mesh):
    adjacency = mesh.face_adjacency

    f_a_mat = np.zeros(mesh.faces.shape, dtype = np.int64)

    f_a_count = np.zeros(mesh.faces.shape[0], dtype = np.int64)

    for i in range(adjacency.shape[0]):
        f1 = adjacency[i][0]
        f2 = adjacency[i][1]

        f_a_mat[f1][f_a_count[f1]] = f2
        f_a_count[f1] += 1
        f_a_mat[f2][f_a_count[f2]] = f1
        f_a_count[f2] += 1

    return f_a_mat

def find_first(v, vec): # return index of first element == v in vec
    result = np.where(vec == v)
    if not result[0].shape[0] == 0:
        return result[0][0]
    else:
        return -1

def find_connected_components(f_a_mat, f_flag):
    components = []
    component_sizes = []
    f_index = find_first(1, f_flag)
    while not f_index == -1:
        component = []
        candidateList = []
        component.append(f_index)
        f_flag[f_index][0] = 0

        candidateList.append(f_a_mat[f_index][0])
        candidateList.append(f_a_mat[f_index][1])
        candidateList.append(f_a_mat[f_index][2])

        while not len(candidateList) == 0:
            f_index = candidateList.pop()
            if f_flag[f_index][0] == 1:
                component.append(f_index)
                f_flag[f_index][0] = 0
                candidateList.append(f_a_mat[f_index][0])
                candidateList.append(f_a_mat[f_index][1])
                candidateList.append(f_a_mat[f_index][2])
        components.append(component)

        f_index = find_first(1, f_flag)

    for component in components:
        component_sizes.append(len(component))
    return components, component_sizes






if __name__=='__main__':



    ######################################
    if False:
        mesh = trimesh.load('/home/otaheri/Desktop/Dice_mesh/knife_beab_new.ply', process=False)
        boundary_face = np.load('/home/otaheri/Desktop/knife_colliding_faces.npy')

        f_a_mat = build_triangle_adjacency_mat(mesh)

        f_flag = np.zeros([mesh.faces.shape[0], 1])
        for f_id in boundary_face:
            f_flag[f_id] = 1
        boundaries, boundary_sizes = find_connected_components(f_a_mat, f_flag)

        print "There are " + str(len(boundaries)) + " boundaries."

        for face in range(mesh.faces.shape[0]):
            mesh.visual.face_colors[face] = [255, 255, 0, 255]

        b_id = 0
        for boundary in boundaries:
            for f_id in boundary:
                mesh.visual.face_colors[f_id] = [b_id * 60 + 60, 0, 0, 255]
            b_id += 1
        mesh.show()

        f_flag = np.ones([mesh.faces.shape[0], 1])
        for face in boundary_face:
            f_flag[face][0] = 0

        components_, component_sizes_ = find_connected_components(f_a_mat, f_flag)
        print("There are " + str(len(components_)) + " components.")

        for i in range(len(components_)):
            for f_id in components_[i]:
                mesh.visual.face_colors[f_id] = [0, i * 60 + 60, 0, 255]
        mesh.visual.face_colors[components_[0][1]] = [0, 0, 255, 255]
        mesh.show()

        boundary_component_mat = np.zeros([len(boundaries), len(components_)])
        # For each boundary check if they contain other boundary:
        for b_id in range(len(boundaries)):
            # Reset face flag:
            f_flag = np.ones([mesh.faces.shape[0], 1])
            for f_id in boundaries[b_id]:
                f_flag[f_id] = 0
            components, component_sizes = find_connected_components(f_a_mat, f_flag)
            # check if components_ in components:
            for i in range(len(components_)):
                if components_[i][0] in components[0]:
                    boundary_component_mat[b_id][i] = -1
                elif components_[i][0] in components[1]:
                    boundary_component_mat[b_id][i] = 1

        print(boundary_component_mat)
        input('')


    ##############################
    ######################
    ############################
    ######################

    # object = Mesh(filename='/home/otaheri/Desktop/Dice_mesh/flute_beab_new.ply')
    # body = Mesh(filename='/home/otaheri/Desktop/Dice_mesh/knife_beab_new.ply')
    #
    # get_first_contact_surface(body=body,object=object)


    object_mesh_file = '/home/otaheri/Desktop/Dice_mesh/flute_beab.ply'
    rigid_mesh = RigidObjectModel(object_mesh_file)

    mvs = MeshViewers(shape=[1,1])

    camRRR = np.array( [[ 1.,  0.,  0.],
                        [ 0.,  1.,  0.],
                        [0.,  0.,  1.]])

    camTTT=np.array([-1.46137345,  0.96600509,  5.31522942])
    cams = np.asarray(camRRR.T.dot(-camTTT)).reshape((1, 3)).astype(np.float64)

    normal_threshold = .5
    from psbody.mesh.visibility import visibility_compute

    vis_final = np.zeros([1,rigid_mesh.v.shape[0]])

    mesh = rigid_mesh.get_mesh()

    for degree in range(10,360,5):
        rotation_matrix = torch.tensor([1,0,0],dtype=torch.float32)*degree2radian(degree)

        rigid_mesh.root_orient[:] = rotation_matrix



        arguments = {'cams': cams,
                     'v': rigid_mesh.verts.numpy(),
                     'f': mesh.f,
                     'n': rigid_mesh.get_mesh().estimate_vertex_normals()}  # 'n': algn2.estimate_vertex_normals()}  # nooo !!! overwrites .vn above

        vis, n_dot = visibility_compute(**arguments)

        if normal_threshold is not None:
            vis = np.logical_and(vis, (n_dot > normal_threshold)).astype(np.int32)

        vis_final = np.logical_or(vis,vis_final).astype(np.int32)




        mvs[0][0].set_static_meshes([rigid_mesh.get_mesh()], blocking=True)
        # time.sleep(1)

    mesh.remove_vertices(np.where(np.sum(vis, axis=0) < 1)[0])
    mvs[0][0].set_static_meshes([mesh], blocking=True)








