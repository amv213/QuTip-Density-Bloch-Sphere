import numpy as np
from qutip import Bloch
from scipy.interpolate import interp2d
from matplotlib import cm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):

    """
    https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


class BlochDensity(Bloch):

    """
    Adapted from
    https://stackoverflow.com/questions/28320247/density-plot-on-a-sphere-python
    """

    def __init__(self):
        super().__init__()
        self.arrow_color = 'k'
        self.arrow_width = 3

    def plot_back(self):

        radius_scaling = 1.

        # back half of sphere
        u = np.linspace(0, np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = radius_scaling*np.outer(np.cos(u), np.sin(v))
        y = radius_scaling*np.outer(np.sin(u), np.sin(v))
        z = radius_scaling*np.outer(np.ones(np.size(u)), np.cos(v))

        # contour grid
        us = np.linspace(0, np.pi, 200)
        vs = np.linspace(0, np.pi, 200)
        xs = radius_scaling*np.outer(np.cos(us), np.sin(vs))
        ys = radius_scaling*np.outer(np.sin(us), np.sin(vs))
        zs = radius_scaling*np.outer(np.ones(np.size(us)), np.cos(vs))
        colours = np.empty(xs.shape, dtype=object)
        for i in range(len(xs)):
            for j in range(len(ys)):
                theta = np.arctan2(ys[i,j], xs[i,j])
                phi = np.arccos(zs[i,j])

                colours[i,j] = self.density(theta, phi)

        self.axes.plot_surface(xs, ys, zs, rstride=1, cstride=1,
                           facecolors=colours,
                           linewidth=0, antialiased=True, shade=False)
        # wireframe
        self.axes.plot_wireframe(x, y, z, rstride=5, cstride=5,
                             color=self.frame_color,
                             alpha=self.frame_alpha)
        # equator
        self.axes.plot(radius_scaling * np.cos(u), radius_scaling * np.sin(u), zs=0, zdir='z',
                   lw=self.frame_width, color=self.frame_color)
        self.axes.plot(radius_scaling * np.cos(u), radius_scaling * np.sin(u), zs=0, zdir='x',
                   lw=self.frame_width, color=self.frame_color)
        #self.axes.plot(radius_scaling * np.cos(u), radius_scaling * np.sin(u),
         #              zs=0, zdir='y', lw=self.frame_width,
          #             color=self.frame_color)

    def plot_arrow_back(self):

        """
        Plot an arrow on the surface of the Bloch Sphere
        """

        #u = np.linspace(-np.pi/2, 0, 50)
        #px = np.cos(u)
        #py = np.sin(u)
        # to equator
        #self.axes.plot(px[:-9], py[:-9], zs=0, zdir='x', lw=self.arrow_width, color=self.arrow_color)
        #a = Arrow3D([0, 0], [px[-9], px[-6]], [py[-9], py[-6]], mutation_scale=20, lw=self.arrow_width, arrowstyle="-|>", color=self.arrow_color)
        # to -0.3
        #self.axes.plot(px[:-18], py[:-18], zs=0, zdir='x', lw=self.arrow_width, color=self.arrow_color)
        #a = Arrow3D([0, 0], [px[-18], px[-15]], [py[-18], py[-15]], mutation_scale=20, lw=self.arrow_width, arrowstyle="-|>", color=self.arrow_color)

        # Rephasing
        #u = np.linspace(0, np.pi/2, 50)  # hor
        #px = np.cos(u)
        #py = np.sin(u)
        #self.axes.plot(px[39:45], py[39:45], zs=0, zdir='z', lw=self.arrow_width, color=self.arrow_color)
        #a = Arrow3D([px[43], px[46]], [py[43], py[46]], [0, 0], mutation_scale=20, lw=self.arrow_width, arrowstyle="-|>", color=self.arrow_color)

        # Dephasing
        #u = np.linspace(np.pi / 2, np.pi, 50)  # hor
        #px = np.cos(u)
        #py = np.sin(u)
        #self.axes.plot(px[:6], py[:6], zs=0, zdir='z', lw=self.arrow_width, color=self.arrow_color)
        #a = Arrow3D([px[4], px[7]], [py[4], py[7]], [0, 0], mutation_scale=20, lw=self.arrow_width, arrowstyle="-|>", color=self.arrow_color)

        # Full through top, with offset
        u = np.linspace(0.3, np.pi/2, 50)
        px = np.cos(u)
        py = np.sin(u)
        self.axes.plot(px, py, zs=0, zdir='x', lw=self.arrow_width, color=self.arrow_color)

        #self.axes.add_artist(a)

    def plot_front(self):

        radius_scaling = 1.

        # front half of sphere
        u = np.linspace(-np.pi, 0, 50)
        v = np.linspace(0, np.pi, 50)
        x = radius_scaling*np.outer(np.cos(u), np.sin(v))
        y = radius_scaling*np.outer(np.sin(u), np.sin(v))
        z = radius_scaling*np.outer(np.ones(np.size(u)), np.cos(v))

        # contour grid
        us = np.linspace(-np.pi, 0, 200)
        vs = np.linspace(0, np.pi, 200)
        xs = radius_scaling*np.outer(np.cos(us), np.sin(vs))
        ys = radius_scaling*np.outer(np.sin(us), np.sin(vs))
        zs = radius_scaling*np.outer(np.ones(np.size(us)), np.cos(vs))
        colours = np.empty(xs.shape, dtype=object)
        for i in range(len(xs)):
            for j in range(len(ys)):
                theta = np.arctan2(ys[i,j], xs[i,j])
                phi = np.arccos(zs[i,j])

                colours[i, j] = self.density(theta, phi)

        self.axes.plot_surface(xs, ys, zs, rstride=1, cstride=1,
                           facecolors=colours,
                           linewidth=0, antialiased=True, shade=False)


        # wireframe
        self.axes.plot_wireframe(x, y, z, rstride=5, cstride=5,
                             color=self.frame_color,
                             alpha=self.frame_alpha)
        # equator
        self.axes.plot(radius_scaling * np.cos(u), radius_scaling * np.sin(u),
                   zs=0, zdir='z', lw=self.frame_width,
                   color=self.frame_color)

        self.axes.plot(radius_scaling * np.cos(u), radius_scaling * np.sin(u),
                   zs=0, zdir='x', lw=self.frame_width,
                   color=self.frame_color)

        #self.axes.plot(radius_scaling * np.cos(u), radius_scaling * np.sin(u),
         #              zs=0, zdir='y', lw=self.frame_width,
          #             color=self.frame_color)

    def plot_arrow_front(self):

        """
        Plot an arrow on the surface of the Bloch Sphere
        """

        #u = np.linspace(-np.pi / 2, -np.pi, 50) # from equator
        #u = np.linspace(-np.pi/2, -np.pi-0.3, 50) # from +0.3
        #px = np.cos(u)
        #py = np.sin(u)

        # Full
        #self.axes.plot(px, py, zs=0, zdir='x', lw=self.arrow_width, color=self.arrow_color, zorder=100)

        # To Sx, Sy = 0 plane, through bottom
        #self.axes.plot(px[:-9], py[:-9], zs=0, zdir='x', lw=self.arrow_width, color=self.arrow_color, zorder=100)
        #a = Arrow3D([0, 0], [px[-9], px[-6]], [py[-9], py[-6]], mutation_scale=20, lw=self.arrow_width, arrowstyle="-|>", color=self.arrow_color, zorder=101)

        # To Sx, Sy = 0 plane, through top
        #u = np.linspace(-3*np.pi/2, -np.pi + 0.3, 50)  # to low 0.3
        #px = np.cos(u)
        #py = np.sin(u)
        #self.axes.plot(px[:-5], py[:-5], zs=0, zdir='x', lw=self.arrow_width, color=self.arrow_color, zorder=100)
        #a = Arrow3D([0, 0], [px[-6], px[-3]], [py[-6], py[-3]], mutation_scale=20, lw=self.arrow_width, arrowstyle="-|>", color=self.arrow_color, zorder=101)

        # Dephasing
        #u = np.linspace(-np.pi/2, 0, 50) # hor
        #px = np.cos(u)
        #py = np.sin(u)
        #self.axes.plot(px[:-46], py[:-46], zs=0, zdir='z', lw=self.arrow_width, color=self.arrow_color, zorder=100)
        #a = Arrow3D([px[-47], px[-44]], [py[-47], py[-44]], [0, 0], mutation_scale=20, lw=self.arrow_width, arrowstyle="-|>", color=self.arrow_color, zorder=101)

        # Rephasing
        u = np.linspace(0, -np.pi/2, 50)  # hor
        px = np.cos(u)
        py = np.sin(u)
        self.axes.plot(px[39:45], py[39:45], zs=0, zdir='z', lw=self.arrow_width, color=self.arrow_color, zorder=101)
        a = Arrow3D([px[43], px[46]], [py[43], py[46]], [0, 0], mutation_scale=20, lw=self.arrow_width, arrowstyle="-|>", color=self.arrow_color, zorder=101)

        # pi/2 quarter turn (TODO)
        #px = 0.5*np.ones(50)
        #py = np.linspace(-0.6, 0.6, 50)
        #pz = np.sqrt(1 - 0.5**2 - py**2)
        #self.axes.plot(px, py, pz, zdir='z', lw=self.arrow_width, color=self.arrow_color, zorder=101)

        self.axes.add_artist(a)

    def plot_axes_labels(self):
        # axes labels
        opts = {'fontsize': self.font_size,
                'color': self.font_color,
                'horizontalalignment': 'center',
                'verticalalignment': 'center',
                'zorder': 300}

        self.axes.text(0, -self.xlpos[0], 0, self.xlabel[0], **opts)
        self.axes.text(0, -self.xlpos[1], 0, self.xlabel[1], **opts)

        self.axes.text(self.ylpos[0], 0, 0, self.ylabel[0], **opts)
        self.axes.text(self.ylpos[1], 0, 0, self.ylabel[1], **opts)

        self.axes.text(0, 0, self.zlpos[0], self.zlabel[0], **opts)
        self.axes.text(0, 0, self.zlpos[1], self.zlabel[1], **opts)

        for a in (self.axes.w_xaxis.get_ticklines() +
                  self.axes.w_xaxis.get_ticklabels()):
            a.set_visible(False)
        for a in (self.axes.w_yaxis.get_ticklines() +
                  self.axes.w_yaxis.get_ticklabels()):
            a.set_visible(False)
        for a in (self.axes.w_zaxis.get_ticklines() +
                  self.axes.w_zaxis.get_ticklabels()):
            a.set_visible(False)

    def render(self, fig=None, axes=None):
        """
        Render the Bloch sphere and its data sets in on given figure and axes.
        """
        if self._rendered:
            self.axes.clear()

        self._rendered = True

        # Figure instance for Bloch sphere plot
        if not fig:
            self.fig = plt.figure(figsize=self.figsize)

        if not axes:
            self.axes = Axes3D(self.fig, azim=self.view[0], elev=self.view[1])

        if self.background:
            self.axes.clear()
            self.axes.set_xlim3d(-1.3, 1.3)
            self.axes.set_ylim3d(-1.3, 1.3)
            self.axes.set_zlim3d(-1.3, 1.3)
        else:
            self.plot_axes()
            self.axes.set_axis_off()
            self.axes.set_xlim3d(-0.7, 0.7)
            self.axes.set_ylim3d(-0.7, 0.7)
            self.axes.set_zlim3d(-0.7, 0.7)

        self.axes.grid(False)
        self.plot_back()
        self.plot_points()
        self.plot_vectors()
        #self.plot_arrow_back()
        self.plot_front()
        #self.plot_arrow_front()
        self.plot_axes_labels()
        self.plot_annotations()


def myDistr(theta, phi):

    a = 0.2
    ct = -np.pi / 2
    b = 0
    c = 32
    cp = np.pi / 2

    offset = (theta - ct)

    f1 = np.exp(-(a*(offset)**2 + 2*b*(theta - ct)*(phi - cp)*np.sqrt(a*c) + c*(phi - cp)**2))
    f2 = np.exp(-(a*(offset - 2*np.pi)**2 + 2*b*(theta - ct)*(phi - cp)*np.sqrt(a*c) + c*(phi - cp)**2))
    f3 = np.exp(-(a*(offset + 2*np.pi)**2 + 2*b*(theta - ct)*(phi - cp)*np.sqrt(a*c) + c*(phi - cp)**2))

    return f1 + f2 + f3


if __name__ == "__main__":

    b = BlochDensity()
    b.sphere_alpha = 0.5

    thetas, phis = np.linspace(-np.pi, np.pi, 1000), np.linspace(0, np.pi, 1000)
    density = myDistr(thetas[None, :], phis[:, None])

    # scale density to a maximum of 1
    density /= density.max()
    interpolated_density = interp2d(thetas, phis, density)

    cmap = cm.get_cmap('Blues')

    def f(theta, phi):
        rgba = cmap(interpolated_density(theta, phi)[0])
        # replace solid transparency with fading out transparency (with small bkg opaqueness)
        rgbaa = rgba[:3] + (.9*interpolated_density(theta, phi)[0] + .1,)
        return rgbaa

    b.density = f

    b.set_label_convention("sx sy sz")

    b.show()