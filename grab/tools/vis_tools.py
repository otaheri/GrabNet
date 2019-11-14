import numpy as np
from psbody.mesh import Mesh

colors = {
    'pink': [.7, .7, .9],
    'purple': [.9, .7, .7],
    'cyan': [.7, .75, .5],
    'red': [1.0,0.0,0.0],

    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [.5, .7, .7],
    'blue': [.0, .0, 1.],

    'offwhite': [.8, .9, .9],
    'white': [1., 1., 1.],
    'orange': [.5, .65, .9],

    'grey': [.7, .7, .7],
    'xardali': [0.588, 0.608, 0],
    'banafsh_light': [0.969, 0.196, 0.906],
    'surati': [0.918, 0.702, 0.906],
    # 'black': np.zeros(3),
    'white': np.ones(3),

    'yellowg': [0.83,1,0],

}


def points_to_spheres(points, radius=0.2, color=np.ones(3) * .5):
    from psbody.mesh.sphere import Sphere

    spheres = Mesh(v=[], f=[])
    for pidx, center in enumerate(points):
        clr = color[pidx] if len(color) > 3 else color
        spheres.concatenate_mesh(Sphere(center, radius).to_mesh(color=clr))
    return spheres