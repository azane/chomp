import numpy as np
import vispy.scene
from vispy.scene import visuals
import src.obstacle as obs
from vis_src.ellipse import Ellipse
import src.kinematics as kn
import theano as th
import theano.tensor as tt

canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# Make a weird L shaped robot.
Lsize = 5.
lx = np.random.multivariate_normal([Lsize, 0., 0.],
                                   [[Lsize**2., 0., 0.],
                                    [0., .1, 0.],
                                    [0., 0., .1]],
                                   10)
ly = np.random.multivariate_normal([0., Lsize, 0.],
                                   [[.1, 0., 0.],
                                    [0., Lsize**2., 0.],
                                    [0., 0., .1]],
                                   10)
u = np.vstack((lx, ly))
U = len(u)
D = 3

# Create a path through space.
qstart = np.ones(3) * -58.
qend = -qstart
Q = int(40)
qvec = (qend - qstart) / Q
qarange = np.arange(Q)[:, None]
qpath = np.hstack((qarange,)*D) * qvec + qstart
qpath = np.hstack((qpath, np.ones((Q, 3))*2*np.pi))

# Compute the body position over the full path.
q = tt.dmatrix('q')
xf = kn.th_6dof_rigid
f_xf = th.function(inputs=[q], outputs=xf(q, tt.constant(u)), mode=th.compile.FAST_COMPILE)
uu = f_xf(qpath)

# Get some gaussians.
K = 3
D = 3
mu = np.random.normal(loc=0., scale=28., size=(K, D))
dd = np.random.normal(loc=0., scale=10., size=(K, D, D*2))
cov = []
for d in dd:
    cov.append(np.cov(d))
cov = np.array(cov)

# Plot them.
for m, c in zip(mu, cov):
    Ellipse(mu=m, cov=c, std=.7, parent=view.scene, edge_color='blue')

# Project the paths into gaussian spherical space.
# This should yield three different paths, each bending around the origin in different ways.
xb = obs.np_el_backproject_all(x=uu.reshape(-1, D), mu=mu, Ainv=np.linalg.inv(np.linalg.cholesky(cov))).reshape(-1, D)

# create scatter object and fill in the data
scatter3 = visuals.Markers()
scatter3.set_data(xb, size=5, face_color='blue')
view.add(scatter3)

scatter4 = visuals.Markers()
scatter4.set_data(uu.reshape(-1, D), size=5, face_color='green')
view.add(scatter4)

view.camera = 'turntable'  # or try 'arcball'

# add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()