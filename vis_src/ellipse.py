# Adapted from https://github.com/vispy/vispy/blob/master/vispy/visuals/sphere.py by Andy Zane

from vispy.geometry import create_sphere
from vispy.visuals.mesh import MeshVisual
from vispy.visuals.visual import CompoundVisual
from vispy.visuals.transforms import MatrixTransform
import numpy as np
from vispy.scene.visuals import create_visual_node


class EllipseVisual(CompoundVisual):
    """Visual that displays an ellipse

    Parameters
    ----------
    mu : the mean of the ellipse
    cov : the covariance of the ellipse
    std : the standard deviation at which to plot the surface.
    cols : int
        Number of cols that make up the sphere mesh
        (for method='latitude' and 'cube').
    rows : int
        Number of rows that make up the sphere mesh
        (for method='latitude' and 'cube').
    depth : int
        Number of depth segments that make up the sphere mesh
        (for method='cube').
    subdivisions : int
        Number of subdivisions to perform (for method='ico').
    method : str
        Method for generating sphere. Accepts 'latitude' for
        latitude-longitude, 'ico' for icosahedron, and 'cube'
        for cube based tessellation.
    vertex_colors : ndarray
        Same as for `MeshVisual` class.
        See `create_sphere` for vertex ordering.
    face_colors : ndarray
        Same as for `MeshVisual` class.
        See `create_sphere` for vertex ordering.
    color : Color
        The `Color` to use when drawing the sphere faces.
    edge_color : tuple or Color
        The `Color` to use when drawing the sphere edges. If `None`, then no
        sphere edges are drawn.
    """
    def __init__(self, mu, cov, std=2., cols=15, rows=15, depth=15, subdivisions=3,
                 method='latitude', vertex_colors=None, face_colors=None,
                 color=(0.5, 0.5, 1, 1), edge_color=None, **kwargs):

        mesh = create_sphere(rows, cols, depth, radius=std,
                             subdivisions=subdivisions, method=method)

        # Take the spherical mesh and project through the covariance, like you do
        #  when sampling a multivariate gaussian.
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
        A = np.linalg.cholesky(cov)
        verts = mesh.get_vertices()
        verts2 = np.matmul(A[None, :, :], verts[:, :, None])
        verts3 = verts2 + mu[None, :, None]
        mesh.set_vertices(np.squeeze(verts3))

        self._mesh = MeshVisual(vertices=mesh.get_vertices(),
                                faces=mesh.get_faces(),
                                vertex_colors=vertex_colors,
                                face_colors=face_colors, color=color)
        if edge_color:
            self._border = MeshVisual(vertices=mesh.get_vertices(),
                                      faces=mesh.get_edges(),
                                      color=edge_color, mode='lines')
        else:
            self._border = MeshVisual()

        CompoundVisual.__init__(self, [self._mesh, self._border], **kwargs)
        self.mesh.set_gl_state(polygon_offset_fill=True,
                               polygon_offset=(1, 1), depth_test=True)

    @property
    def mesh(self):
        """The vispy.visuals.MeshVisual that used to fil in.
        """
        return self._mesh

    @property
    def border(self):
        """The vispy.visuals.MeshVisual that used to draw the border.
        """
        return self._border

Ellipse = create_visual_node(EllipseVisual)
