import sys
import numpy as np

from vispy import app, scene

vehicle = np.array([
    [0., 0.5],
    [2., 0.5],
    [2., -0.5],
    [0., -0.5]
])
arrow = np.array([
    [1.5, 0.5],
    [2., 0.],
    [1.5, -0.5]
])

canvas = scene.SceneCanvas(size=(500, 500), keys='interactive')

scene.visuals.Polygon(pos=arrow*10+200, parent=canvas.scene, color='grey')
scene.visuals.Polygon(pos=vehicle*10+200, parent=canvas.scene, color='white')


def update(event):
    pass


timer = app.Timer('auto', connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()