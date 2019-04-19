# -*- coding: utf-8 -*-
# vispy: gallery 10
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

""" Demonstrates use of visual.Markers to create a point cloud with a
standard turntable camera to fly around with and a centered 3D Axis.
"""

import numpy as np
import vispy.scene
from vispy.scene import visuals
import src.obstacle as obs

canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# generate data
pos = np.random.normal(size=(100000, 3), scale=0.2)
colors = np.ones((len(pos), 4), dtype=float)

# Adjust transparency via the gm obstacle field.
mu = np.random.rand(5, 3)
cov = np.random.rand(5, 3, 3)
cov += np.transpose(cov, (0, 2, 1))
prec = np.linalg.inv(cov)
obs.np_gm_distance_field(pos, mu, cov)
colors[:, -1] = .3

# create scatter object and fill in the data
scatter = visuals.Markers()
scatter.set_data(pos, edge_color=None, face_color=colors, size=5)

view.add(scatter)

view.camera = 'turntable'  # or try 'arcball'

# add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()