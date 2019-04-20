import numpy as np
import sys
from vispy import app, visuals, scene
import src.objective as obj

# Adapted from https://github.com/vispy/vispy/blob/master/examples/basics/visuals/line_plot3d.py by Andy Zane

# build visuals
Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)

# build canvas
canvas = scene.SceneCanvas(keys='interactive', title='plot3d', show=True)

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 6


def get_data():
    N = 60
    x = np.sin(np.linspace(-2, 2, N) * np.pi)
    y = np.cos(np.linspace(-2, 2, N) * np.pi)
    z = np.linspace(-2, 2, N)
    pos = np.c_[x, y, z]

    pos += np.random.rand(*pos.shape)*.5

    return pos


data1 = get_data()
p1 = Plot3D(data1, width=2.0, color='green',
            edge_color='w', symbol='o', face_color=(0.2, 0.2, 1, 0.8),
            parent=view.scene)
data2 = get_data()
p2 = Plot3D(data2, width=2.0, color='red',
            edge_color='w', symbol='o', face_color=(0.2, 0.2, 1, 0.8),
            parent=view.scene)


# CHOMP covariant descent.
_, fp, _ = obj.ffp_smoothness()
K = obj.slow_fdiff_1(len(data1)-2)
Ainv = np.linalg.inv(K.T.dot(K))
def update(ev):
    qq1 = fp(data1)
    data1[1:-1] -= 0.01*Ainv.dot(qq1[1:-1])
    p1.set_data(data1)
    p1.update()

    qq2 = fp(data2)
    data2[1:-1] -= 0.01 * qq2[1:-1]
    p2.set_data(data2)
    p2.update()


timer = app.Timer(connect=update, interval=0.04)
timer.start()

if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()
