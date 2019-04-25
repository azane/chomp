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
import vis_src.vis_6dof_gm_helpers as h
import src.rrt as rrt

D = h.D
Q = h.Q
U = h.U

q = tt.dmatrix('q')
u = h.get_L_3d_robot()

# <Kinematics>
xf = kn.th_6dof_rigid
f_xf = th.function(inputs=[q], outputs=xf(q, u), mode=th.compile.FAST_COMPILE)
# </Kinematics>

# <Obstacles>
mu, cov = h.get_gm_obstacle_field()
covAinv = np.linalg.inv(np.linalg.cholesky(cov))

prec = np.linalg.inv(cov)
ttmu = tt.constant(mu)
ttprec = tt.constant(prec)

# cf = obs.th_gm_closest_obstacle_cost_wrap(ttmu, ttprec)
# f_cf = th.function(inputs=[q], outputs=cf(xf(q, u)), mode=th.compile.FAST_COMPILE)

qstart = np.ones(3) * -58.
qend = -qstart
qpath = np.vstack((qstart[None, :], qend[None, :]))
qpath = np.hstack((qpath, np.ones(qpath.shape)))
# </Obstacles>

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

planner = rrt.RRT_GM6DOF(mu=mu, cov=cov, u=u, start=qpath[0], goal=qpath[-1],
                         # AA is dynamical per distance. TODO erm does this make sense with aa not just a?
                         bounds=[(-58, 58)]*3 + [(-.006*np.pi, .006*np.pi)]*3)


def update(ev):
    if not planner.done:
        qpath = planner.plan()
        if qpath is None:
            return
        p1.set_data(qpath[:, :3])
        p1.update()

        uu = f_xf(qpath)
        scatter.set_data(uu.reshape(-1, D), edge_color=scolor, face_color=scolor)
        scatter.update()

        if planner.done:
            print("Clear!")



timer = vispy.app.Timer(connect=update, interval=0.05)
timer.start()

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()
