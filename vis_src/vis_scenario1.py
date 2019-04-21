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
from collections import deque


"""
Build a gradient for the scenario where:
1. Our obstacles are just a bunch of ellipsoids.
2. Our robot is a random point cloud.
3. Our robot moves with translation only.
"""

D = 3

# <Robot>
q = tt.dmatrix('q')
u = tt.constant(np.random.normal(loc=0., scale=1., size=(10, D)), name='u')

# The boundary conditions and initial path.
qstart = np.ones(3) * -40.
qend = -qstart
Q = 120
qvec = (qend - qstart) / Q
qarange = np.arange(Q)[:, None]
qpath = np.hstack((qarange,)*D) * qvec + qstart
# </Robot>

# <Obstacles>
K = 6
mu = np.random.normal(loc=0., scale=25., size=(K, D))
dd = np.random.normal(loc=0., scale=13., size=(K, D, 7))
cov = []
for x in dd:
    cov.append(np.cov(x))
cov = np.array(cov)
prec = np.linalg.inv(cov)
ttmu = tt.constant(mu)
ttprec = tt.constant(prec)

cf = obs.th_gm_obstacle_cost_wrap(ttmu, ttprec)
f_cf = th.function(inputs=[q], outputs=cf(q), mode=th.compile.FAST_COMPILE)


def path_clear(qpath_):
    # Clear if the whole robot is outside 2.02 stdevs for all robot points.
    return np.all(np.less(f_cf(qpath_), .130))
# </Obstacles>

# <Gradient>
# For debugging purposes.
th.config.compute_test_value = 'warn'
q.tag.test_value = qpath

# Smoothness objective.
smooth, _ = obj.th_smoothness(q)

# Obstacle objective.
xf = kn.th_translation_only
f_xf = th.function(inputs=[q], outputs=xf(q, u), mode=th.compile.FAST_COMPILE)
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
p1 = PlotTrajectory(qpath, width=.1, color='green',
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
Ainv = np.linalg.inv(K_.T.dot(K_))

clear = False
qi = 0


def update(ev):
    global clear
    global qi
    if not clear:
        qpath_p = fp_obj(qpath)
        vispy.app.process_events()
        qpath[1:-1] -= 0.01*Ainv.dot(qpath_p[1:-1])
        vispy.app.process_events()
        p1.set_data(qpath)
        p1.update()

        clear = path_clear(qpath)
        if clear:
            print("Clear!")
    else:
        # Once complete, move the robot along the path acc. to the kinematics.
        scatter.set_data(np.squeeze(f_xf(qpath[qi%Q][None, ...])), edge_color=scolor, face_color=scolor)
        scatter.update()
        qi += 1


timer = vispy.app.Timer(connect=update, interval=0.05)
timer.start()
# </Update>


if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()
