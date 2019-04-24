import sys
import numpy as np

from vispy import app, scene


class Vehicle(object):
    def __init__(self, vehicle, scene_):
        super().__init__()
        assert vehicle.shape == (4,2)
        self.vehicle = vehicle
        self.arrow = self.vehicle[[1, 2, 2]]
        self.arrow[1, 1] += 0.5
        self.scene = scene_

        self._pa = scene.visuals.Polygon(pos=self.arrow, parent=self.scene, color='grey')
        self._pv = scene.visuals.Polygon(pos=self.vehicle, parent=self.scene, color='white')

    def draw(self, f):
        self._pa.set_data(f(self.arrow))
        self._pa.update()
        self._pv.set_data(f(self.vehicle))
        self._pv.update()

vehicle = np.array([
    [0., 0.5],
    [2., 0.5],
    [2., -0.5],
    [0., -0.5]
])

canvas = scene.SceneCanvas(size=(500, 500), keys='interactive')

v = Vehicle(vehicle, canvas.scene)


def update(event):
    v.draw(lambda x: x * 50 + np.random.randint(100, 300))


timer = app.Timer('auto', connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()