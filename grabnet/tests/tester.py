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

import os
import numpy as np
import torch

from grabnet.tools.utils import makepath, makelogger
from grabnet.models.models import CoarseNet, RefineNet



class Tester:

    def __init__(self, cfg):

        self.dtype = torch.float32

        makepath(cfg.work_dir, isfile=False)
        logger = makelogger(makepath(os.path.join(cfg.work_dir, 'V00.log'), isfile=True)).info
        self.logger = logger

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.coarse_net = CoarseNet().to(self.device)
        self.refine_net = RefineNet().to(self.device)

        self.cfg = cfg
        self.coarse_net.cfg = cfg

        if cfg.best_cnet is not None:
            self._get_cnet_model().load_state_dict(torch.load(cfg.best_cnet, map_location=self.device), strict=False)
            logger('Restored CoarseNet model from %s' % cfg.best_cnet)
        if cfg.best_rnet is not None:
            self._get_rnet_model().load_state_dict(torch.load(cfg.best_rnet, map_location=self.device), strict=False)
            logger('Restored RefineNet model from %s' % cfg.best_rnet)

        self.bps = torch.from_numpy(np.load(cfg.bps_dir)['basis']).to(self.dtype)

    def _get_cnet_model(self):
        return self.coarse_net.module if isinstance(self.coarse_net, torch.nn.DataParallel) else self.coarse_net

    def _get_rnet_model(self):
        return self.refine_net.module if isinstance(self.refine_net, torch.nn.DataParallel) else self.refine_net