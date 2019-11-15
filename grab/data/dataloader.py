
import os
import torch
from torch.utils.data import Dataset
import glob
import numpy as np

from grab.tools.rotations import local2global_pose


class GRAB_DS(Dataset):
    def __init__(self, dataset_dir, ):

        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt','')
            self.ds[k] = torch.load(data_fname).type(torch.float32)
            
        self.frame_names = np.load(os.path.join(dataset_dir, 'frame_names.npz'))['frame_names']

    def __len__(self):
        k = list(self.ds.keys())[0]
        return len(self.ds[k])


    def __getitem__(self, idx):

        data = {k: self.ds[k][idx] for k in self.ds.keys()}
    
        # data['root_orient_HR'] = data['fullpose'][63:66]
        # data['pose_HR'] = data['fullpose'][120:]
        # obj_info = self.data_info['object_infos'][self.object_names[idx]]
        # data['verts_dsampl'] = obj_info['object_verts_dsampl']
        # data['verts_dsampl_bps'] = obj_info['object_verts_dsampl_bps']
        # data['verts_dsampl_bps_deltas'] = obj_info['object_verts_dsampl_bps_deltas']
        data['idx'] = torch.from_numpy(np.array(idx, dtype=np.int32))
        #
        # data = {k: torch.from_numpy(v.reshape(-1)) if not isinstance(v, torch.Tensor) else v for k,v in data.items() }
        return data


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from basis_point_sets.bps import reconstruct_from_bps
    from psbody.mesh import MeshViewer, Mesh
    from grab.tools.vis_tools import colors, points_to_spheres
    from human_body_prior.body_model.body_model import BodyModel
    from smplx.lbs import batch_rodrigues
    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    from human_body_prior.train.vposer_smpl import VPoser#matrot2aa
    import time
    batch_size = 256
    dataset_dir = '/ps/scratch/body_hand_object_contact/grab_net/data/V01_07_00/train'

    ds = GRAB_DS(dataset_dir=dataset_dir)
    print('dataset size: %d'%len(ds))

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=False)

    mv = MeshViewer(keepalive=False)
    bps_basis = np.load(os.path.join(dataset_dir, '../bps_basis_1024.npz'))['basis']
    bm_path = '/ps/scratch/body_hand_object_contact/body_models/models/models_unchumpy/MANO_RIGHT.npz'

    bm = BodyModel(bm_path, batch_size=batch_size)

    for i_batch, dorig in enumerate(dataloader):
        # print({k:v.type() for k,v in dorig.items()})
        id = 0

        dorig_cpu = {k:c2c(v)[id] for k, v in dorig.items()}

        # print(dorig.keys())

        o_orig_pc = reconstruct_from_bps(dorig_cpu['delta_object'][None], bps_basis)[0]
        obj_mesh = points_to_spheres(o_orig_pc, radius=0.001, color=colors['blue'])
        hand_mesh = Mesh(dorig_cpu['verts_hand_mano'], [], vc=colors['grey'])

        bm_eval = bm(**dorig).v
        bm_mesh = Mesh(c2c(bm_eval[id]), c2c(bm.f), vc=colors['red'])
        print(ds.frame_names[dorig['idx']][id])

        mv.set_static_meshes([obj_mesh, hand_mesh, bm_mesh])
        time.sleep(2)

        # reconstruct object using the o_delta
        # visualize hand using the hand verts

        #
        # object_verts_dsampl_rec = reconstruct_from_bps(data['verts_dsampl_bps_deltas'].reshape(1,-1,3), ds.data_info['bps_basis'])[0]
        # class_id_color = {k:v for k,v in enumerate(colors.values())}
        #
        # obj_verts = data['verts_dsampl'].reshape(-1,3)
        # obj_mesh1 = points_to_spheres(obj_verts, radius=0.001, color=class_id_color[0])
        # mv.set_static_meshes([obj_mesh1])
        # meshes = []
        # for j in range(data['contact_maps'].max()):
        #     if j == 0: continue
        #     contact_verts = obj_verts[data['contact_maps'] == j]
        #     if len(contact_verts):
        #         obj_mesh2 = points_to_spheres(contact_verts, radius=0.001, color=class_id_color[j])
        #         meshes.append(obj_mesh2)
        #
        #         print('# contact for class %d = %d'%(j, (data['contact_maps'] == j).sum()))
        # # obj_mesh2 = points_to_spheres(object_verts_dsampl_rec, radius=0.001, color=colors['blue'])
        #
        # mv.set_dynamic_meshes(meshes)
        # mv.set_titlebar('%05d'%i_batch)
        # time.sleep(5)
