import numpy as np
import vispy.scene
from vispy.scene import visuals
import src.obstacle as obs
from vis_src.ellipse import Ellipse

canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# generate some points around a few unit spheres.
x = np.random.normal(0., 1., size=(1000, 3))
x = x / np.linalg.norm(x, axis=1, keepdims=True)
x[330:] *= 1.75
x[660:] *= 2.1
colors = np.ones((len(x), 4), dtype=float)
colors[330:660, 0] = .1
colors[660:, 1] = .1

# Get some gaussians.
K = 3
D = 3
mu = np.random.normal(loc=0., scale=15., size=(K, D))
dd = np.random.normal(loc=0., scale=3., size=(K, D, D*2))
cov = []
for d in dd:
    cov.append(np.cov(d))
cov = np.array(cov)

A = np.linalg.cholesky(cov)
xg = np.matmul(A[None, ...], x[:, None, :, None])
xg = xg.squeeze() + mu[None, :, :]
xg = xg.reshape(-1, D)

# And reproject with slightly different mus (so we can differentiate the balls.)
# TODO
xb = obs.np_el_backproject_all(x=xg, mu=mu, Ainv=np.linalg.inv(A)).reshape(-1, D)
xb += np.ones((1, 3))*20.

# create scatter object and fill in the data
scatter = visuals.Markers()
scatter.set_data(x, edge_color=colors, face_color=colors, size=5)
scatter2 = visuals.Markers()
scatter2.set_data(xg, size=5, face_color='blue')
scatter3 = visuals.Markers()
scatter3.set_data(xb, size=5, face_color='red')

view.add(scatter)
view.add(scatter2)
view.add(scatter3)

view.camera = 'turntable'  # or try 'arcball'

# add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()