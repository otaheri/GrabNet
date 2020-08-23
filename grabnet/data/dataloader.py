# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#

import numpy as np
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
import os

import time
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

class LoadData(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 ds_name='train',
                 dtype=torch.float32,
                 only_params = False,
                 load_on_ram = False):

        super().__init__()

        self.only_params = only_params

        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.ds = self._np2torch(os.path.join(self.ds_path,'grabnet_%s.npz'%ds_name))

        frame_names = np.load(os.path.join(dataset_dir,ds_name, 'frame_names.npz'))['frame_names']
        self.frame_names =np.asarray([os.path.join(dataset_dir, fname) for fname in frame_names])
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2].split('_')[0] for name in self.frame_names])

        self.sbjs = np.unique(self.frame_sbjs)
        self.obj_info = np.load(os.path.join(dataset_dir, 'obj_info.npy'), allow_pickle=True).item()
        self.sbj_info = np.load(os.path.join(dataset_dir, 'sbj_info.npy'), allow_pickle=True).item()

        ## bps_torch data

        bps_fname = os.path.join(dataset_dir, 'bps.npz')
        self.bps = torch.from_numpy(np.load(bps_fname)['basis']).to(dtype)
        ## Hand vtemps and betas

        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_vtemp'] for sbj in self.sbjs]))
        self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_betas'] for sbj in self.sbjs]))

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)

        self.load_on_ram = False
        if load_on_ram:
            self.ds = self[:]
            self.load_on_ram = True

    def _np2torch(self,ds_path):
        data = np.load(ds_path, allow_pickle=True)
        data_torch = {k:torch.tensor(data[k]) for k in data.files}
        return data_torch
    def load_disk(self,idx):

        if isinstance(idx, int):
            return self._np2torch(self.frame_names[idx])

        frame_names = self.frame_names[idx]
        from_disk = []
        for f in frame_names:
            from_disk.append(self._np2torch(f))
        from_disk = default_collate(from_disk)
        return from_disk

    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]
        # return len(self.frame_names)

    def __getitem__(self, idx):

        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        if not self.only_params:
            if not self.load_on_ram:
                form_disk = self.load_disk(idx)
                data_out.update(form_disk)
        return data_out

if __name__=='__main__':

    data_path = '/ps/scratch/grab/contact_results/omid_46/GrabNet/data'
    ds = LoadData(data_path, ds_name='val', only_params=False)

    dataloader = data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=10, drop_last=True)

    s = time.time()
    for i in range(320):
        a = ds[i]
    print(time.time() - s)
    print('pass')

    dl = iter(dataloader)
    s = time.time()
    for i in range(10):
        a = next(dl)
    print(time.time()-s)
    print('pass')

    # mvs = MeshViewers(shape=[1,1])
    #
    # bps_torch = test_ds.bps_torch
    # choice = np.random.choice(range(test_ds.__len__()), 30, replace=False)
    # for frame in choice:
    #     data = test_ds[frame]
    #     rhand = Mesh(v=data['verts_rhand'].numpy(),f=[])
    #     obj = Mesh(v=data['verts_object'].numpy(), f=[], vc=name_to_rgb['blue'])
    #     bps_p = Mesh(v=bps_torch, f=[], vc=name_to_rgb['red'])
    #     mvs[0][0].set_static_meshes([rhand,obj, bps_p])
    #     time.sleep(.4)
    #
    # print('finished')
