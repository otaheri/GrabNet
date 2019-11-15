import numpy as np

# Expects rotations in a list
# e.g. [90, 180, 270]
# --> rotateX by 90
# --> rotateY by 180
# --> rotateZ by 270
import torch

def get_z_rotmat(angle):
    rz = np.array([
        [np.cos(angle), -np.sin(angle), 0.],
        [np.sin(angle), np.cos(angle), 0.],
        [0., 0., 1.]
    ])
    return  rz

def rotateZ(mesh_v, Rz):
    '''

    :param mesh_v: Nxnum_vx3
    :param Rxyz: Nx3
    :return:
    '''

    mesh_v_rotated = []
    z_rots = []

    for fId in range(mesh_v.shape[0]):

        angle = np.radians(Rz[fId])
        rz = np.array([
            [np.cos(angle), -np.sin(angle), 0. ],
            [np.sin(angle), np.cos(angle), 0. ],
            [0., 0., 1. ]
        ])
        z_rots.append(rz)
        mesh_v_rotated.append(rz.dot(mesh_v[fId].T).T)

    return np.array(mesh_v_rotated), np.array(z_rots)

def rotateXYZ(mesh_v, Rxyz):
    '''

    :param mesh_v: Nxnum_vx3
    :param Rxyz: Nx3
    :return:
    '''

    mesh_v_rotated = []

    for fId in range(mesh_v.shape[0]):
        angle = np.radians(Rxyz[fId, 0])
        rx = np.array([
            [1., 0., 0.           ],
            [0., np.cos(angle), -np.sin(angle)],
            [0., np.sin(angle), np.cos(angle) ]
        ])

        angle = np.radians(Rxyz[fId, 1])
        ry = np.array([
            [np.cos(angle), 0., np.sin(angle)],
            [0., 1., 0.           ],
            [-np.sin(angle), 0., np.cos(angle)]
        ])

        angle = np.radians(Rxyz[fId, 2])
        rz = np.array([
            [np.cos(angle), -np.sin(angle), 0. ],
            [np.sin(angle), np.cos(angle), 0. ],
            [0., 0., 1. ]
        ])
        mesh_v_rotated.append(rz.dot(ry.dot(rx.dot(mesh_v[fId].T))).T)

    return np.array(mesh_v_rotated)


def local2global_pose(local_pose):

    local_pose = local_pose.view(local_pose.shape[0], -1, 3, 3)
    global_pose = local_pose.clone()
    # global_pose[:, 1] = torch.matmul(global_pose[:, 0], global_pose[:, 1].clone())
    # global_pose[:, 2] = torch.matmul(global_pose[:, 0], global_pose[:, 2].clone())
    global_pose[:, 3] = torch.matmul(global_pose[:, 0], global_pose[:, 3].clone())

    # global_pose[:, 4] = torch.matmul(global_pose[:, 1], global_pose[:, 4].clone())
    # global_pose[:, 5] = torch.matmul(global_pose[:, 2], global_pose[:, 5].clone())
    global_pose[:, 6] = torch.matmul(global_pose[:, 3], global_pose[:, 6].clone())

    # global_pose[:, 7] = torch.matmul(global_pose[:, 4], global_pose[:, 7].clone())
    # global_pose[:, 8] = torch.matmul(global_pose[:, 5], global_pose[:, 8].clone())
    global_pose[:, 9] = torch.matmul(global_pose[:, 6], global_pose[:, 9].clone())

    # global_pose[:, 10] = torch.matmul(global_pose[:, 7], global_pose[:, 10].clone())
    # global_pose[:, 11] = torch.matmul(global_pose[:, 8], global_pose[:, 11].clone())
    # global_pose[:, 12] = torch.matmul(global_pose[:, 9], global_pose[:, 12].clone())

    # global_pose[:, 13] = torch.matmul(global_pose[:, 9], global_pose[:, 13].clone())
    global_pose[:, 14] = torch.matmul(global_pose[:, 9], global_pose[:, 14].clone())
    # global_pose[:, 15] = torch.matmul(global_pose[:, 12], global_pose[:, 15].clone())

    # global_pose[:, 16] = torch.matmul(global_pose[:, 13], global_pose[:, 16].clone())
    global_pose[:, 17] = torch.matmul(global_pose[:, 14], global_pose[:, 17].clone())
    # global_pose[:, 18] = torch.matmul(global_pose[:, 16], global_pose[:, 18].clone())

    global_pose[:, 19] = torch.matmul(global_pose[:, 17], global_pose[:, 19].clone())
    # global_pose[:, 20] = torch.matmul(global_pose[:, 18], global_pose[:, 20].clone())
    global_pose[:, 21] = torch.matmul(global_pose[:, 19], global_pose[:, 21].clone())

    return global_pose.view(global_pose.shape[0],-1,3,3)