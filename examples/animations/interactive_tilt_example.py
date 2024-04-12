from pathlib import Path
import sys

from ase.io import read
import ked123
from ked123.microscope import *
from ked123.reciprocal_lattice import *
from ked123.simulation import calculate_excitation_error
from matplotlib.cm import gray, viridis
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from vispy import app, geometry, scene
from vispy.color import ColorArray
from vispy.geometry import create_cylinder
from vispy.visuals.mesh import MeshData, MeshVisual
from vispy.visuals.transforms import STTransform

wavelength = electron_wavelength(200) / 1e-10

hkl = generate_hkl_points(None, None, np.arange(-5, 5 + 1), n=25)
cif = read(Path(ked123.__file__).parent.joinpath("Fe alpha.cif"))
reciprocal = reciprocal_vectors(*cif.get_cell())
g = calculate_g_vectors(hkl, reciprocal)

# apply any rotation to g vectors
# euler_angles = [122.33840256, 40.82669342, 279.85280663]
# euler_angles = [123.10517286, 41.05359795, 279.52062714]
# g = rotate_vector(euler_angles, g)

S_MAX = 0.01
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

sphere = scene.visuals.Sphere(1 / wavelength, parent=view.scene, color=(0, 0, 0, 0.5))
center = (0, 0, 1 / wavelength)
sphere.transform = STTransform(translate=center)

view.camera.set_range(x=[-10, 10])

# create cylinder
mesh = trimesh.load("/Users/paddyharrison/Documents/GitHub/3D-SPED/KED/ellipsoid.stl")
vert_scaling = np.array((1, 0.5, 0.5)) * 0.2
verts = mesh.vertices * vert_scaling
offset = np.ptp(verts, axis=0) / 2.0
verts = verts - offset

faces = mesh.faces
mdata = geometry.MeshData(verts, mesh.faces)
mesh = scene.visuals.Mesh(
    meshdata=mdata,
    shading="flat",
)
view.add(mesh)

# shading = ShadingFilter()
# mesh.attach(shading)

# precession angle
psi = np.deg2rad(0.0)

axis = scene.visuals.XYZAxis(parent=view.scene)
axis_data = axis.pos * 15
axis.set_data(width=2)

# line to Ewald sphere center
arrow = scene.visuals.Arrow(pos=np.array(((0, 0, 0), center)), width=5, arrow_size=10)
view.add(arrow)

period = 3 * np.pi
omega = 0
sf = 0.015
rotvec = np.array((1, 1, 0))
rotvec = rotvec / np.linalg.norm(rotvec) * sf
offset = rotvec * period / 2.0


def update(event):
    """ """
    global markers_active, sphere, g, wavelength, S_MAX, psi, arrow

    new_center = calculate_ewald_sphere_center(wavelength, psi, omega)
    arrow.set_data(pos=np.array(((0, 0, 0), new_center)))
    sphere.transform = STTransform(translate=new_center)

    rot = Rotation.from_rotvec(rotvec * (event.elapsed % period) - offset)
    new_g = rot.apply(g)
    new = calculate_excitation_error(new_g, wavelength, psi, omega)
    mask = np.abs(new) <= S_MAX
    c = (S_MAX - np.abs(new[mask])) / S_MAX
    markers_active.set_data(pos=new_g[mask], face_color=ColorArray(gray(c)))

    # update cylinder
    mesh.set_data(vertices=rot.apply(verts) + (0, 0, 5), faces=faces)

    # update XYZ axis
    axis.set_data(rot.apply(axis_data) + (0, 0, 5), width=3)


timer = app.Timer()
timer.connect(update)
timer.start()

if __name__ == "__main__" and sys.flags.interactive == 0:
    canvas.show()
    if sys.flags.interactive == 0:
        canvas.app.run()
