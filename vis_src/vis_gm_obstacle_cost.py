import numpy as np
import vispy.scene
from vispy.scene import visuals
import src.obstacle as obs
from vis_src.ellipse import Ellipse

# Adapted from https://github.com/vispy/vispy/blob/master/examples/basics/scene/point_cloud.py by Andy Zane

# TODO clean this up...a lot.

canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# generate data
pos = np.random.normal(size=(1000000, 3), scale=3.)
colors = np.ones((len(pos), 4), dtype=float)

# Get some gaussians.
mu = np.random.normal(scale=3., size=(5, 3))
cov = []
for _ in range(len(mu)):
    mat = np.random.rand(3, 3) - .5
    mat += mat.T
    mat = mat.dot(mat)
    mat += np.diag(np.random.rand(3)*2.)
    cov.append(mat)
cov = np.array(cov)
prec = np.linalg.inv(cov)
# prec = np.repeat(np.diag([1., .1, .6])[None, :, :], repeats=5, axis=0)
# cov = np.linalg.inv(prec)

# Change coloring for those inside 2 stdevs.
ci = 0
# colors[:, ci] = .606 - np.max(obs.np_gm_obstacle_cost(pos, mu, prec), axis=1)
colors[:, ci] = .135 - np.max(obs.np_gm_obstacle_cost(pos, mu, prec), axis=1)
colors[:, ci] /= np.abs(colors[:, ci]) * -1.
colors[:, ci] += 1.
colors[:, ci] /= 2
# colors[:, -1] = .3

# Actually just only visualize those within 2 stdevs.
pos = pos[colors[:, ci].astype(bool), ...]
colors = colors[colors[:, ci].astype(bool), ...]

# Draw the std ellipses.
for m, c in zip(mu, cov):
    Ellipse(mu=m, cov=c, std=2., parent=view.scene)

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