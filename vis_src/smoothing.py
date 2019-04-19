import numpy as np
import sys
from vispy import app, visuals, scene
import src.objective as obj

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


data = get_data()
p = Plot3D(data, width=2.0, color='red',
           edge_color='w', symbol='o', face_color=(0.2, 0.2, 1, 0.8),
           parent=view.scene)


# CHOMP covariant descent.
_, fp, _ = obj.ffp_smoothness()
K = obj.slow_fdiff_1(len(data)-2)
Ainv = np.linalg.inv(K.T.dot(K))
def update(ev):
    qq = fp(data)
    data[1:-1] -= 0.01*Ainv.dot(qq[1:-1])
    # data[1:-1] -= 0.01*(qq[1:-1])
    p.set_data(data)
    p.update()


timer = app.Timer(connect=update, interval=0.04)
timer.start()

if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()
