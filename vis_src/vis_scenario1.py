import src.kinematics as kn
import src.objective as obj
import src.obstacle as obs
import theano as th
import theano.tensor as tt
import numpy as np
import vispy
import vispy.scene
import vispy.visuals
from vis_src.ellipse import Ellipse
from scipy.optimize import minimize


"""
Build a gradient for the scenario where:
1. Our obstacles are just a bunch of ellipsoids.
2. Our robot is a random point cloud.
3. Our robot moves with translation only.
"""

D = 3

# <Robot>
q = tt.dmatrix('q')
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
u = tt.constant(np.vstack((lx, ly)))
# u = tt.constant(lx)
U = len(u.value)

SCALE = 1

# The boundary conditions and initial path.
qstart = np.ones(3) * -58. * SCALE
qend = -qstart
Q = int(40 * SCALE)
qvec = (qend - qstart) / Q
qarange = np.arange(Q)[:, None]
qpath = np.hstack((qarange,)*D) * qvec + qstart

qpath = np.hstack((qpath, np.ones((Q, 3))*2*np.pi))
# </Robot>

# <Kinematics>
xf = kn.th_6dof_rigid
f_xf = th.function(inputs=[q], outputs=xf(q, u), mode=th.compile.FAST_COMPILE)
# </Kinematics>

# <Obstacles>
K = int(10 * SCALE ** 2)
mu = np.random.normal(loc=0., scale=27.*SCALE, size=(K, D))
dd = np.random.normal(loc=0., scale=12., size=(K, D, 7))
cov = []
for x in dd:
    cov.append(np.cov(x))
cov = np.array(cov)
prec = np.linalg.inv(cov)
ttmu = tt.constant(mu)
ttprec = tt.constant(prec)

cf = obs.th_gm_closest_obstacle_cost_wrap(ttmu, ttprec)
f_cf = th.function(inputs=[q], outputs=cf(xf(q, u)), mode=th.compile.FAST_COMPILE)


def path_clear(qpath_):
    # Clear if the whole robot is outside ~2.1 stdevs for all robot points.
    # 2 stdevs is considered the "boundary" here.
    return np.all(np.less(f_cf(qpath_), .110))


covAinv = np.linalg.cholesky(cov)
def path_clear_meh(qpath_):
    raise NotImplementedError("This is broken.")
    # Clear if the path of closest approach between all adjacent qpath members
    #  is outside of the obstacle closest to that path.
    xx = f_xf(qpath_)
    x1 = xx[1:]
    x2 = xx[:-1]
    d = obs.np_el_nearestd(x1=x1.reshape(-1, D), x2=x2.reshape(-1, D), mu=mu, Ainv=covAinv, prec=prec)
    # Clear if no point is within 2.1 stdevs of an ellipse.
    return not np.any(d < 2.1)
# </Obstacles>


# <Gradient>
# For debugging purposes.
# th.config.compute_test_value = 'warn'
# q.tag.test_value = qpath

# Smoothness objective.
# Minimally value angular smoothness.
w = th.shared(np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4]))
smooth, _ = obj.th_smoothness(q, w)

# Obstacle objective.
obstac, _ = obj.th_obstacle(q=q, u=u,
                            cf=cf,
                            xf=xf)

y_obj = obstac + smooth
yp_obj = th.grad(y_obj, wrt=q)

f_obj = th.function(inputs=[q], outputs=y_obj, mode=th.compile.FAST_COMPILE)
fp_obj = th.function(inputs=[q], outputs=yp_obj, mode=th.compile.FAST_COMPILE)
# </Gradient>

# <Visualization>
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# Draw the ellpses at 2 stdevs.
for m, c in zip(mu, cov):
    Ellipse(mu=m, cov=c, std=2., parent=view.scene, edge_color=(0,0,0,1))

PlotTrajectory = vispy.scene.visuals.create_visual_node(vispy.visuals.LinePlotVisual)
p1 = PlotTrajectory(qpath[:, :3], width=.1, color='green',
                    edge_color='w', symbol='o', face_color=(0.2, 0.2, 1, 0.8),
                    parent=view.scene)

# The scatter of the robot's body.
scatter = vispy.scene.visuals.Markers()
scolor = (0,1,0,1)
scatter.set_data(np.squeeze(f_xf(qpath[0][None, ...])), edge_color=scolor, face_color=scolor)
view.add(scatter)

view.camera = 'turntable'
# </Visualization>

# <Update>
K_ = obj.slow_fdiff_1(len(qpath)-2)
Ainv = np.linalg.inv(K_.T.dot(K_) + np.diag(np.ones(K_.shape[1])*0.001))

clear = False
qi = 0

# <Manual>
maxiter = 75
mi = 0
last_obj = np.inf
step = np.array([0.01, 0.01, 0.01, 0.0005, 0.0005, 0.0005])


def update(ev):
    global clear
    global qi
    global mi
    global qpath
    global step
    global last_obj

    mi += 1
    if not clear and mi > maxiter:
        return

    if not clear:

        qpath_p = fp_obj(qpath)
        vispy.app.process_events()
        qpath[1:-1] -= step[None, :]*Ainv.dot(qpath_p[1:-1])
        vispy.app.process_events()

        # If any angle-axes of our 6dof angle-axes are close to zero,
        #  adjust them all away.
        # This breaks the transformation etc. etc.
        # We trigger this with enough "magnitude" to pull out a reasonable
        #  axis from the angle axis.
        if np.any(np.less(qpath[:, 3:].sum(axis=1), .2)):
            qpath = kn.unzero_6dof(qpath)

        this_obj = f_obj(qpath)
        print("Objective: ", this_obj)
        if (this_obj > last_obj):
            # step *= .7
            step *= .4
        else:
            # step *= 1./.98
            step *= 1. / .8
        print("Step: ", step)
        last_obj = this_obj

        p1.set_data(qpath[:, :3])
        p1.update()

        clear = path_clear(qpath)
        if clear:
            print("Clear!")

        scatter.set_data(np.reshape(f_xf(qpath), newshape=(-1, D)), edge_color=scolor, face_color=scolor)
        scatter.update()
    else:
        # Once complete, move the robot along the path acc. to the kinematics.
        scatter.set_data(np.squeeze(f_xf(qpath[qi%Q][None, ...])), edge_color=scolor, face_color=scolor)
        scatter.update()
        qi += 1

    if not clear and mi == maxiter:
        print(f"Failed after {mi} iterations!")

# </Manual>


timer = vispy.app.Timer(connect=update, interval=0.05)
timer.start()
# </Update>


if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()
