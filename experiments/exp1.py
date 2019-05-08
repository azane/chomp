import experiments.helpers as h
import theano.tensor as tt
import theano as th
import src.kinematics as kn
import src.obstacle as obs
import src.objective as obj
import src.rrt as rrt
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
    # xf = kn.th_6dof_translation_only
    global f_xf
    f_xf = th.function(inputs=[ttq], outputs=xf(ttq, ttu), mode=th.compile.FAST_RUN)

    # Obstacle function.
    ttmu = tt.dmatrix('mu')
    ttprec = tt.dtensor3('prec')
    # cf = obs.th_gm_closest_obstacle_cost_wrap(ttmu, ttprec)
    cf = obs.th_gm_obstacle_cost_wrap(ttmu, ttprec)
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

    for k in range(8, 9):
        for s in range(10, 11):

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

            # Naming note:
            # ct_* is chomp straight.
            # cr_* is chomp rrt
            # rr_* is rrt

            # <CHOMP Straight Line Init>

            print("CHOMP-Straight Line Init")
            cs_solver = h.solve_chomp(q0straight,
                                      f_obj=lambda q_: f_obj(q_, mu, prec),
                                      fp_obj=lambda q_: fp_obj(q_, mu, prec),
                                      path_clear=path_clear)
            cs_qn, cs_objv, n = next(cs_solver)
            while True:
                try: cs_qn, cs_objv, n = next(cs_solver)
                except StopIteration as e:
                    cs_success = e.value
                    break
                print(f"{n}: {cs_objv}")

            cs_results = (cs_success, cs_qn)

            # </CHOMP Straight Line Init>

            # <RRT>
            print("RRT")
            rrt_bounds = [(-h.BBOX, h.BBOX)] * 3 + [(-.0001 * np.pi, .0001 * np.pi)] * 3
            rrt_planner = rrt.RRT_GM6DOF(mu=mu, cov=cov, u=ttu,
                                         start=q0straight[0], goal=q0straight[-1],
                                         bounds=rrt_bounds)
            rr_n = 0
            rr_qn = None  # type: np.ndarray
            while not rrt_planner.done and not rr_n > 300:
                rr_qn = rrt_planner.plan()
                rr_n += 1

            rr_results = (rrt_planner.done, rr_qn)

            # </RRT>

            # <CHOMP RRT Init>
            print("CHOMP-RRT Init")
            print(rr_qn[:, :3])
            cr_qn0 = h.chomp_path_from_rrt(rr_qn)
            print(cr_qn0[:, :3])
            cr_solver = h.solve_chomp(cr_qn0,
                                      f_obj=lambda q_: f_obj(q_, mu, prec),
                                      fp_obj=lambda q_: fp_obj(q_, mu, prec),
                                      path_clear=path_clear, miniter=50)
            cr_qn, cr_objv, n = next(cr_solver)
            while True:
                try:
                    cr_qn, cr_objv, n = next(cr_solver)
                except StopIteration as e:
                    cr_success = e.value
                    break
                print(f"{n}: {cr_objv}")

            cr_results = (cr_success, cr_qn)

            # <CHOMP RRT Init>

            yield cs_results, rr_results, cr_results, mu, cov
            print("----------")

    print("Done")


def vis_main():
    import sys
    import vispy.scene
    import vispy.app
    import vispy.visuals
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
            cs_results, rr_results, cr_results, mu, cov = next(gen)
            cs_success, cs_qn = cs_results
            rr_success, rr_qn = rr_results
            cr_success, cr_qn = cr_results
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
        cs_scat.set_data(np.reshape(f_xf(cs_qn), newshape=(-1, h.D)), edge_color=cs_color, face_color=cs_color)
        cs_scat.update()
        cs_traj.set_data(cs_qn[:, :3])
        cs_traj.update()

        rr_scat.set_data(np.reshape(f_xf(rr_qn), newshape=(-1, h.D)), edge_color=rr_color, face_color=rr_color)
        rr_scat.update()
        rr_traj.set_data(rr_qn[:, :3])
        rr_traj.update()

        cr_scat.set_data(np.reshape(f_xf(cr_qn), newshape=(-1, h.D)), edge_color=cr_color, face_color=cr_color)
        cr_scat.update()
        cr_traj.set_data(cr_qn[:, :3])
        cr_traj.update()

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
    rr_traj = PlotTrajectory(vq0[:, :3], width=.1, color='green',
                             edge_color='w', symbol='o', face_color=(0.2, 0.2, 1, 0.8),
                             parent=view.scene)
    cr_traj = PlotTrajectory(vq0[:, :3], width=.1, color='green',
                             edge_color='w', symbol='o', face_color=(0.2, 0.2, 1, 0.8),
                             parent=view.scene)

    # The scatter of the robot's body.
    cs_scat = vispy.scene.visuals.Markers()
    cs_color = (0, 1, 0, 1)
    cs_scat.set_data(np.reshape(f_xf(vq0), newshape=(-1, h.D)),
                     edge_color=cs_color, face_color=cs_color)
    view.add(cs_scat)

    rr_scat = vispy.scene.visuals.Markers()
    rr_color = (0, 0, 1, 1)
    rr_scat.set_data(np.reshape(f_xf(vq0), newshape=(-1, h.D)),
                     edge_color=rr_color, face_color=rr_color)
    view.add(rr_scat)

    cr_scat = vispy.scene.visuals.Markers()
    cr_color = (1, 0, 0, 1)
    cr_scat.set_data(np.reshape(f_xf(vq0), newshape=(-1, h.D)),
                     edge_color=cr_color, face_color=cr_color)
    view.add(cr_scat)

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
