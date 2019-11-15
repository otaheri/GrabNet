
import os
import torch
from torch.utils.data import Dataset
import glob
import numpy as np


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
    dataset_dir = '/ps/scratch/body_hand_object_contact/grab_net/data/V01_11_00/test'
    # dataset_dir = '/ps/scratch/body_hand_object_contact/grab_net/data/V01_07_00/train'

    ds = GRAB_DS(dataset_dir=dataset_dir)
    print('dataset size: %d'%len(ds))

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=False)

    mv = MeshViewer(keepalive=False)
    bps_basis = np.load(os.path.join(dataset_dir, '../bps_basis_1024.npz'))['basis']
    bm_path = '/ps/scratch/body_hand_object_contact/body_models/models/models_unchumpy/MANO_RIGHT.npz'

    bm = BodyModel(bm_path, batch_size=batch_size)

    for i_batch, dorig in enumerate(dataloader):
        # print({k:v.type() for k,v in dorig.items()})
        found_one = False
        for id in range(batch_size):
            if 'fryingpan' in ds.frame_names[dorig['idx']][id]:
                found_one = True
                break
        if not found_one: continue

        dorig_cpu = {k:c2c(v)[id] for k, v in dorig.items()}

        # print(dorig.keys())

        o_orig_pc = reconstruct_from_bps(dorig_cpu['delta_object'][None], bps_basis)[0]
        obj_mesh = points_to_spheres(o_orig_pc, radius=0.001, color=colors['blue'])
        hand_mesh = Mesh(dorig_cpu['verts_hand_mano'], [], vc=colors['grey'])

        bm_eval = bm(**dorig).v
        bm_mesh = Mesh(c2c(bm_eval[id]), c2c(bm.f), vc=colors['red'])
        print(ds.frame_names[dorig['idx']][id])
        print(ds.frame_names[dorig['idx']])

        mv.set_static_meshes([obj_mesh, hand_mesh, bm_mesh])
        time.sleep(2)

