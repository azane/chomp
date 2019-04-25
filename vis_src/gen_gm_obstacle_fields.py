import vis_src.vis_6dof_gm_helpers as h
import os
import numpy as np

dpath = os.path.join(os.path.dirname(__file__), '..', 'data')

if __name__ == "__main__":
    if not os.path.isdir(dpath):
        os.mkdir(dpath)
    for i in range(10):
        mu, cov = h.get_gm_obstacle_field()
        np.save(os.path.join(dpath, f"mu{i}"), mu)
        np.save(os.path.join(dpath, f"cov{i}"), cov)
