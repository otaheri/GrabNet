

import numpy as np
import cPickle as pickle
from fit_rigid_object2mocap import RigidObjectModel
from psbody.mesh import Mesh, MeshViewers, MeshViewer
from experiments.hand.smpl_handpca_wrapper import load_model
import torch
import json

from experiments.nima.tools.mocap_interface import MocapSession
from experiments.omid.object_interaction.fit_rigid_object2mocap import get_marker_data
from experiments.omid.object_interaction.mesh2mesh_contact import point2point
from experiments.omid.object_interaction.mesh2mesh_contact import get_first_contact_surface

from experiments.nima.tools.visualization_helpers import spheres_for

from experiments.hand.fast_derivatives.smpl_HF_fastderivatives_WRAPPER import load_model as load_smplx
from experiments.hand.fast_derivatives.smpl_HF_fastderivatives import SmplModelLBS


if __name__=='__main__':

    mvs = MeshViewers(window_width=2400, window_height=1600, shape=[1, 1])
    object_mesh_file = '/ps/project/body_hand_object_contact/data/object_meshes/final_scaled_and_cleaned_meshes/wristwatch.ply'
    # object_mesh_file = '/ps/project/body_hand_object_contact/data/object_meshes/edited_meshes/cubesmall.ply'
    object_marker_setting = '/ps/project/body_hand_object_contact/data/object_settings/marker_settings_wristwatch.json'

    object_mesh = Mesh(filename=object_mesh_file)

    with open(object_marker_setting, 'r') as mrkr_setng:
        markers = json.load(mrkr_setng)

    marker = markers['markersets'][0]['indices'].values()
    b = markers['markersets'][0]['indices'].keys()
    print marker
    print b

    marker_location = object_mesh.v[marker]
    marker_location.reshape(-1,3)

    mocap_markers = spheres_for(marker_location.reshape(-1,3), radius=0.0016,  vc=np.array((0., 0., 1.)))

    mocap_markers = spheres_for(marker_location[0].reshape(-1,3), radius=0.0016,  vc=np.array((1., 0., 0.)))
    mocap_markers1 = spheres_for(marker_location[1].reshape(-1,3), radius=0.0016,  vc=np.array((0.,1., 0.)))
    mocap_markers2 = spheres_for(marker_location[2].reshape(-1,3), radius=0.0016,  vc=np.array((0., 0., 1.)))
    mocap_markers3 = spheres_for(marker_location[3].reshape(-1,3), radius=0.0016,  vc=np.array((1., 0., 1.)))
    mocap_markers4 = spheres_for(marker_location[4].reshape(-1,3), radius=0.0016,  vc=np.array((0., 1., 1.)))
    mocap_markers5 = spheres_for(marker_location[5].reshape(-1,3), radius=0.0016,  vc=np.array((1., 1., 0.)))
    mocap_markers6 = spheres_for(marker_location[6].reshape(-1,3), radius=0.0016,  vc=np.array((1.,1.,1.)))
    mocap_markers7 = spheres_for(marker_location[6].reshape(-1,3), radius=0.0016,  vc=np.array((0., 0., 0.)))
    mocap_markers8 = spheres_for(marker_location[6].reshape(-1,3), radius=0.0016,  vc=np.array((0., 0., 0.)))
    # mocap_markers9 = spheres_for(marker_location[9].reshape(-1,3), radius=0.0016,  vc=np.array((0., 0., 1.)))
    #
    mvs[0][0].set_static_meshes([object_mesh] + mocap_markers + mocap_markers1+ mocap_markers2+ mocap_markers3+ mocap_markers4+ mocap_markers5+ mocap_markers6+ mocap_markers7 + mocap_markers8)#+ mocap_markers9 )
    # mvs[0][0].set_static_meshes([object_mesh] + mocap_markers )

    print 'hi'