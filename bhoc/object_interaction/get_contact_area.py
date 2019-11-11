
import open3d as o3d

import numpy as np
import torch
import torch.nn as nn
from psbody.mesh import Mesh, MeshViewers, MeshViewer
from experiments.nima.tools.visualization_helpers import spheres_for
from experiments.nima.tools_torch.optimizers.lbfgs_ls import LBFGS as LBFGSLs
from experiments.nima.tools.mocap_interface import MocapSession

import torchgeometry as tgm
import copy

from body.alignment.objectives import sample_from_mesh

class GetContact():

    def __init__(self,
                body_plypath,
                object_plypath,
                dtype = torch.float32):

        super(GetContact, self).__init__()


        self.dtype = dtype


    def ScanToMesh(self,scan, mesh_verts, mesh_faces, rho=lambda x : x, scan_sampler=None, normalize=True, signed=False):
        """Returns a Ch object whose only dterm is 'mesh_verts'"""

        if scan_sampler is None:
            scan_sampler = scan

        sampler, n_samples = construct_sampler(scan_sampler, scan.v.size / 3)

        norm_const = np.sqrt(n_samples) if normalize else 1

        if signed:
            fn = lambda x : SignedSqrt(rho(x)) / norm_const
        else:
            fn = lambda x : ch.sqrt(rho(x)) / norm_const

        result = Ch(lambda mesh_verts : fn(MeshDistanceSquared(
            sample_verts=scan.v,
            sample_faces=scan.f,
            reference_verts=mesh_verts,
            reference_faces=mesh_faces,
            sampler=sampler,
            signed=signed
            )))

        result.mesh_verts = mesh_verts
        return result

    def construct_sampler(self,sampler_or_template, num_mesh_verts):
        if isinstance(sampler_or_template, dict):
            sampler = sampler_or_template
        else:
            sampler = sample_from_mesh(sampler_or_template, sample_type='uniformly-from-vertices', num_samples=1e+5)

        n_samples = sampler['point2sample'].shape[0] / 3 if 'point2sample' in sampler else num_mesh_verts
        return sampler, n_samples