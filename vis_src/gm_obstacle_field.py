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
from vis_src.ellipse import Ellipse

canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# generate data
pos = np.random.normal(size=(1000000, 3), scale=3.)
colors = np.ones((len(pos), 4), dtype=float)

# Adjust transparency via the gm obstacle field.
mu = np.random.normal(scale=3., size=(5, 3))
# cov = []
# for _ in range(len(mu)):
#     mat = np.random.rand(3, 3) - .5
#     mat += mat.T
#     mat = mat.dot(mat)
#     cov.append(mat)
# cov = np.array(cov)
# prec = np.linalg.inv(cov)
prec = np.repeat(np.diag([1., .1, .6])[None, :, :], repeats=5, axis=0)
cov = np.linalg.inv(prec)
ci = 0
# colors[:, ci] = .606 - np.max(obs.np_gm_obstacle_cost(pos, mu, prec), axis=1)
colors[:, ci] = .135 - np.max(obs.np_gm_obstacle_cost(pos, mu, prec), axis=1)
colors[:, ci] /= np.abs(colors[:, ci]) * -1.
colors[:, ci] += 1.
colors[:, ci] /= 2
# colors[:, -1] = .3
pos = pos[colors[:, ci].astype(bool), ...]
colors = colors[colors[:, ci].astype(bool), ...]

# TODO draw covariance ellipsoids, i.e. the obstacles.
for m, c in zip(mu, cov):
    Ellipse(mu=m, cov=c, parent=view.scene)

# create scatter object and fill in the data
scatter = visuals.Markers()
scatter.set_data(pos, edge_color=colors, face_color=colors, size=5)

view.add(scatter)

view.camera = 'turntable'  # or try 'arcball'

# add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()