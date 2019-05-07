import experiments.helpers as h
import theano.tensor as tt
import theano as th
import src.kinematics as kn
import src.obstacle as obs
import src.objective as obj
import numpy as np
from typing import *


def exp1():

    # Retrieve a non-random robot body.
    ttu = h.get_L_3d_const_robot()

    # The path variable.
    ttq = tt.dmatrix('q')

    # Kinematic function.
    xf = kn.th_6dof_rigid
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

    for k in range(3, 5):
        for s in range(10, 20):

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


if __name__ == "__main__":
    for success_, qn_, mu_, cov_ in exp1():
        print(success_)