

import torch
import torch.nn as nn

from experiments.nima.tools_torch.lbs import batch_rodrigues

class RigidObjectModel(nn.Module):

    def __init__(self,
                 model_plypath,
                 params=None,
                 dtype=torch.float32):

        super(RigidObjectModel, self).__init__()

        '''

        '''
        # Todo:

        self.dtype = dtype
        from psbody.mesh import Mesh
        rigid_mesh = Mesh(filename=model_plypath)

        self.register_buffer('f', torch.tensor(rigid_mesh.f.astype(np.int32), dtype=torch.int32))
        self.register_buffer('v', torch.tensor(rigid_mesh.v.astype(np.float32) * .001, dtype=dtype))

        trans = torch.tensor(np.zeros((3)), dtype=dtype, requires_grad=True)
        self.register_parameter('trans', nn.Parameter(trans, requires_grad=True))

        root_orient = torch.tensor(np.zeros((3)), dtype=dtype, requires_grad=True)
        self.register_parameter('root_orient', nn.Parameter(root_orient, requires_grad=True))

        marker_loc = torch.tensor(np.zeros((6, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('marker_loc', nn.Parameter(root_orient, requires_grad=True))

    def forward(self, root_orient=None, trans=None, **kwargs):

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

    def get_mesh(self):
        with torch.no_grad():
            verts = self.forward()
            mesh = Mesh(v=verts.cpu().numpy(), f=self.f.numpy())
        return mesh
