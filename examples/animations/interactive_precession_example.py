import sys
from pathlib import Path

import numpy as np
from ase.io import read
from matplotlib.cm import viridis
from vispy import app, scene
from vispy.color import ColorArray
from vispy.visuals.transforms import STTransform

import ked
from ked.microscope import *
from ked.reciprocal_lattice import *
from ked.simulation import calculate_excitation_error

wavelength = electron_wavelength(200) / 1e-10

hkl = generate_hkl_points(n=25)
cif = read(Path(ked.__file__).parent.joinpath("Fe alpha.cif"))
reciprocal = reciprocal_vectors(*cif.get_cell())
g = calculate_g_vectors(hkl, reciprocal)

# apply any rotation to g vectors
# euler_angles = [122.33840256, 40.82669342, 279.85280663]
# euler_angles = [123.10517286, 41.05359795, 279.52062714]
# g = rotate_vector(euler_angles, g)

S_MAX = 0.05
s = calculate_excitation_error(g, wavelength)
s_mask = np.abs(s) < S_MAX

# add (empty) markers to vispy canvas
markers = scene.visuals.Markers()
markers.set_data(
    pos=g[~s_mask],
    edge_width=0.0,
    face_color=(1, 1, 1, 0.1),
    scaling=False,
)

markers_active = scene.visuals.Markers()
markers_active.set_data(
    pos=g[s_mask],
    edge_color="k",
    face_color="k",
    size=10,
    scaling=False,
)

canvas = scene.SceneCanvas(
    keys="interactive", bgcolor="white", size=(800, 600), show=True
)

view = canvas.central_widget.add_view()
view.camera = "arcball"

# view.add(markers)
view.add(markers_active)

sphere = scene.visuals.Sphere(1 / wavelength, parent=view.scene, color=(1, 0, 0, 0.5))
center = (0, 0, 1 / wavelength)
sphere.transform = STTransform(translate=center)

view.camera.set_range(x=[-10, 10])

# precession angle
psi = np.deg2rad(4.5)

axis = scene.visuals.XYZAxis(parent=view.scene)
axis.set_data(width=2)

# line to Ewald sphere center
arrow = scene.visuals.Arrow(pos=np.array(((0, 0, 0), center)), width=5, arrow_size=10)
view.add(arrow)


def update(event):
    """ """
    global markers_active, sphere, g, wavelength, S_MAX, psi, arrow

    new_center = calculate_ewald_sphere_center(wavelength, psi, event.elapsed)
    arrow.set_data(pos=np.array(((0, 0, 0), new_center)))
    sphere.transform = STTransform(translate=new_center)

    new = calculate_excitation_error(g, wavelength, psi, event.elapsed)
    mask = np.abs(new) <= S_MAX
    c = (S_MAX - np.abs(new[mask])) / S_MAX
    markers_active.set_data(pos=g[mask], face_color=ColorArray(viridis(c)))


timer = app.Timer()
timer.connect(update)
timer.start()

if __name__ == "__main__" and sys.flags.interactive == 0:
    canvas.show()
    if sys.flags.interactive == 0:
        canvas.app.run()
