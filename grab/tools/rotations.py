import numpy as np

# Expects rotations in a list
# e.g. [90, 180, 270]
# --> rotateX by 90
# --> rotateY by 180
# --> rotateZ by 270
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
