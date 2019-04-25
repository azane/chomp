import sys
import numpy as np
import src.kinematics as kn
from vispy import app, scene


class Vehicle(object):
    def __init__(self, vehicle, scene_):
        super().__init__()
        assert vehicle.shape == (4,2)
        self.vehicle = vehicle
        self.scene = scene_

        self._pa = scene.visuals.Polygon(pos=self._arrow(self.vehicle), parent=self.scene, color='gray')
        self._pv = scene.visuals.Polygon(pos=self.vehicle, parent=self.scene, color='white')

    def _arrow(self, v):
        m = np.mean(v, axis=0)
        return np.array([
            v[0],
            m,
            v[-1]
        ])

    def draw(self, f):
        v = f(self.vehicle)
        self._pa.pos = self._arrow(v)
        self._pv.pos = v


vehicle = np.array([
    [0., 0.5],
    [2., 0.5],
    [2., -0.5],
    [0., -0.5]
])*20. + 100.

canvas = scene.SceneCanvas(size=(500, 500), keys='interactive')

v = Vehicle(vehicle, canvas.scene)


# TODO TODO
# I think all we have to do is run this for each corner of the vehicle...
# The vehicle's starting angle needs to be computed from its four corners...
# And then we run this for each of those four points. This gives us their end position, and in fact
#  every position along the arc when seen as a function of the arc length.

def update(event):
    kn.np_ackermann(xyt=xyt, a=1., rinv=.1)


timer = app.Timer('auto', connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()