from . import kinematics as kn
from . import objective as obj
from . import obstacle as obs
import theano as th
import theano.tensor as tt
import numpy as np


def scenario1():
    """
    Build a gradient for the scenario where:
    1. Our obstacles are just a bunch of ellipsoids.
    2. Our robot is a random point cloud.
    3. Our robot moves with translation only.
    """

    D = 3

    # Robot.
    q = tt.dmatrix('q')
    u = tt.constant(np.random.normal(loc=0., scale=1., size=(10, D)), name='u')

    # Obstacles.
    K = 10
    mus = np.random.normal(loc=0., scale=10., size=(K, D))
    dd = np.random.normal(loc=0., scale=3., size=(K, D, 5))
    cov = []
    for x in dd:
        cov.append(np.cov(x))
    cov = np.array(cov)
    precs = np.linalg.inv(cov)
    ttmu = tt.constant(mus)
    ttprec = tt.constant(precs)

    # Smoothness objective.
    smooth, _ = obj.th_smoothness(q)

    # Obstacle objective.
    obstac, _ = obj.th_obstacle(q=q, u=u,
                                cf=obs.th_gm_obstacle_cost_wrap(ttmu, ttprec),
                                xf=kn.th_translation_only)

    y_obj = obstac + smooth
    yp_obj = th.grad(y_obj, wrt=q)

    return th.function(inputs=[q], outputs=y_obj),\
           th.function(inputs=[q], outputs=yp_obj),
