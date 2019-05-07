import experiments.helpers as h
import theano.tensor as tt
import theano as th
import src.kinematics as kn
import src.obstacle as obs
import src.objective as obj
import numpy as np
from typing import *


# TODO HACK needed for visualization so making global...
f_xf = None  # type: Callable


def exp1():

    # Retrieve a non-random robot body.
    ttu = h.get_L_3d_const_robot()

    # The path variable.
    ttq = tt.dmatrix('q')

    # Kinematic function.
    xf = kn.th_6dof_rigid
    global f_xf
    f_xf = th.function(inputs=[ttq], outputs=xf(ttq, ttu), mode=th.compile.FAST_RUN)

    # Obstacle function.
    ttmu = tt.dmatrix('mu')
    ttprec = tt.dtensor3('prec')
    cf = obs.th_gm_closest_obstacle_cost_wrap(ttmu, ttprec)
    # f_cf = th.function(inputs=[ttq, ttmu, ttprec], outputs=cf(xf(ttq, ttu)), mode=th.compile.FAST_RUN)

    # Smoothness objective.
    ttw = tt.constant(np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4]))
    smooth, _ = obj.th_smoothness(ttq, ttw)

    # Obstacle objective.
    obstac, _ = obj.th_obstacle(q=ttq, u=ttu, cf=cf, xf=xf)

    # Full objective and gradient.
    ttf_obj = obstac + smooth
    ttfp_obj = th.grad(ttf_obj, wrt=ttq)
    f_obj = th.function(inputs=[ttq, ttmu, ttprec], outputs=ttf_obj, mode=th.compile.FAST_RUN)
    fp_obj = th.function(inputs=[ttq, ttmu, ttprec], outputs=ttfp_obj, mode=th.compile.FAST_RUN)

    # TODO HACK to get the kinematics function loaded.
    yield None

    for k in range(4, 5):
        for s in range(5, 6):

            print(f"Running: k={k}, s={s}")

            # Our initial, straight-line path.
            q0straight = h.get_6dof_straight_path()

            # Re-generate obstacle fields until we have a valid starting position.
            while True:
                mu, cov = h.get_spheres(k, s)
                path_clear = h.path_clear_wrap(mu, cov, f_xf)
                if path_clear(q0straight[:2]) and path_clear(q0straight[-2:]):
                    break
            prec = np.linalg.inv(cov)

            solver = h.solve_chomp(q0straight,
                                   f_obj=lambda q_: f_obj(q_, mu, prec),
                                   fp_obj=lambda q_: fp_obj(q_, mu, prec),
                                   path_clear=path_clear)
            qn, objv, n = next(solver)
            while True:
                try: qn, objv, n = next(solver)
                except StopIteration as e:
                    success = e.value
                    break
                print(f"{n}: {objv}")

            yield success, qn, mu, cov
            print("----------")


def vis_main():
    import sys
    import vispy.scene
    import vispy.app
    from vis_src.ellipse import Ellipse
    import time

    gen = exp1()
    # TODO HACK to get the kinematics function loaded.
    assert next(gen) is None

    ellipses = []

    # Number of view ticks the user is allowed to look at things for.
    VT = 10
    global vt
    vt = 0

    def update(ev):
        global vt
        if vt < VT:
            vt = vt + 1
            return

        try:
            success, qn, mu, cov = next(gen)
        except StopIteration:
            return

        # We just completed a trial, so let the viewer look at it for VT ticks.
        vt = 0

        # Remove previous obstacles.
        for e in ellipses:
            e.parent = None

        # Add current obstacles.
        for m, c in zip(mu, cov):
            e = Ellipse(mu=m, cov=c, std=2., parent=view.scene, edge_color=(0, 0, 0, 1))
            ellipses.append(e)

        # Update the trajectory.
        cs_scat.set_data(np.reshape(f_xf(qn), newshape=(-1, h.D)), edge_color=scolor, face_color=scolor)
        cs_scat.update()
        cs_traj.set_data(qn[:, :3])
        cs_traj.update()

        print("Viewing...")

    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()

    # A dummy path for vis init purposes.
    vq0 = h.get_6dof_straight_path()

    # Visualization name info:
    # ct_* is chomp straight.
    # cr_* is chomp rrt
    # rr_* is rrt

    # A trajectory plot typedef.
    PlotTrajectory = vispy.scene.visuals.create_visual_node(vispy.visuals.LinePlotVisual)
    cs_traj = PlotTrajectory(vq0[:, :3], width=.1, color='green',
                             edge_color='w', symbol='o', face_color=(0.2, 0.2, 1, 0.8),
                             parent=view.scene)

    # The scatter of the robot's body.
    cs_scat = vispy.scene.visuals.Markers()
    scolor = (0, 1, 0, 1)
    cs_scat.set_data(np.reshape(f_xf(vq0), newshape=(-1, h.D)),
                     edge_color=scolor, face_color=scolor)
    view.add(cs_scat)

    view.camera = 'turntable'

    timer = vispy.app.Timer(connect=update, interval=1.0)
    timer.start()

    if sys.flags.interactive != 1:
        vispy.app.run()


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "--vis":
        vis_main()
    else:
        pass  # TODO
