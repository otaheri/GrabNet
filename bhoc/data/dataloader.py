
import os
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import pickle

class BHOC_DS(Dataset):
    def __init__(self, dataset_dir, ):

        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt','')
            self.ds[k] = torch.load(data_fname).type(torch.float32)
            
        # with open(os.path.join(dataset_dir, 'data_info.pkl'), 'rb') as f: self.data_info = pickle.load(f)

    def __len__(self):
        k = list(self.ds.keys())[0]
        return len(self.ds[k])


    def __getitem__(self, idx):

        data = {k: self.ds[k][idx] for k in self.ds.keys()}
        # ToDo add BPS too

        #
        # obj_info = self.data_info['object_infos'][self.object_names[idx]]
        # data['verts_dsampl'] = obj_info['object_verts_dsampl']
        # data['verts_dsampl_bps'] = obj_info['object_verts_dsampl_bps']
        # data['verts_dsampl_bps_deltas'] = obj_info['object_verts_dsampl_bps_deltas']
        # data['idx'] = np.array(idx, dtype=np.int32)
        #
        # data = {k: torch.from_numpy(v.reshape(-1)) if not isinstance(v, torch.Tensor) else v for k,v in data.items() }
        return data

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from basis_point_sets.bps import reconstruct_from_bps
    from psbody.mesh import MeshViewer
    from bhoc.tools.vis_tools import colors, points_to_spheres
    import time

    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    
    batch_size = 256
    dataset_dir = '/ps/scratch/body_hand_object_contact/bhoc_network/data/V01_01_00/test'

    ds = BHOC_DS(dataset_dir=dataset_dir)
    print('dataset size: %d'%len(ds))

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=False)
    # mv = MeshViewer(keepalive=False)

    for i_batch, sample_batched in tqdm(enumerate(dataloader)):
        print(sample_batched.keys())
        print({k:v.type() for k,v in sample_batched.items()})
        break
        # print
        #
        # data = {k:c2c(v)[0] for k, v in sample_batched.items()}
        # idx = data['idx']
        # print('%d'%ds.data_info['contact_maps_fnames'][idx])
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
