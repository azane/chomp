import experiments.helpers as h
import theano.tensor as tt
import theano as th
import src.kinematics as kn
import src.obstacle as obs
import src.objective as obj
import numpy as np

D = h.D

# Retrieve a non-random robot body constant.
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


# Clear path check. Wrapper so we only compute Ainv once per set of obstacles.
def path_clear_wrap(mu, cov):

    Ainv = np.linalg.inv(np.linalg.cholesky(cov))

    def wrap(q):
        # Clear if the path of closest approach between all adjacent qpath members
        #  is outside of the obstacle closest to that path.
        xx = f_xf(q)
        x1 = xx[1:]
        x2 = xx[:-1]
        d = obs.np_el_nearestd(x1=x1.reshape(-1, D), x2=x2.reshape(-1, D), mu=mu, Ainv=Ainv)
        # Clear if no point is within 2.1 stdevs of an ellipse.
        col = d < 2.1
        return not np.any(col)

    return wrap







