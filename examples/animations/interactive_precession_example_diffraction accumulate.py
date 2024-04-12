from pathlib import Path
import sys
import time

from ase.io import read
import ked123
from ked123.generator import CrystalDiffractionGenerator
from ked123.microscope import *
from ked123.reciprocal_lattice import *
from ked123.simulation import kernel_blurred_disk
from ked123.template import calculate_excitation_error
from matplotlib.cm import viridis
import numpy as np
from orix.quaternion.rotation import Rotation
from orix.vector.vector3d import Vector3d
from vispy import app, scene
from vispy.color import ColorArray
from vispy.geometry import create_cone
from vispy.scene.visuals import Image, Mesh
from vispy.visuals.transforms import STTransform

V = 200  # kV
wavelength = electron_wavelength(V)

hkl = generate_hkl_points(n=25)
cif = read(Path(ked123.__file__).parent.joinpath("data", "testing", "Fe alpha.cif"))

atoms = cif * (10, 10, 3)
apos = atoms.get_positions()
apos -= apos.mean(axis=0)

generator = CrystalDiffractionGenerator(cif, V)

# add (empty) markers to vispy canvas
markers = scene.visuals.Markers()
markers.set_data(
    pos=apos,
    edge_width=0.0,
    face_color="r",
)
markers.scaling = False


canvas = scene.SceneCanvas(
    keys="interactive", bgcolor="white", size=(400, 800), show=True
)

view = canvas.central_widget.add_view()
view.camera = "turntable"
view.camera.elevation = 45
view.camera.set_range(x=[-30, 30])
view.camera.azimuth = 0


view.add(markers)

# sphere = scene.visuals.Sphere(1 / wavelength, parent=view.scene, color=(1, 0, 0, 0.5))
# center = (0, 0, 1 / wavelength)
# sphere.transform = STTransform(translate=center)


beam_height = 50
cone = create_cone(100, 3, beam_height)
cone_verts = cone.get_vertices()
cone_verts[:, 2] = beam_height - cone_verts[:, 2]
cone_mesh = Mesh(cone_verts, cone.get_faces())
view.add(cone_mesh)

image_shape = np.array((256, 256))
image_scale = (0.2, 0.2)
image = Image(np.zeros(image_shape), clim=(0, 0.1))
view.add(image)
image_translation = -(image_shape * image_scale) / 2
image.transform = STTransform(
    scale=image_scale + (1,), translate=tuple(image_translation) + (-50,)
)

# axis = scene.visuals.XYZAxis(parent=view.scene)
# axis.set_data(width=2)

# line to Ewald sphere center
# arrow = scene.visuals.Arrow(pos=np.array(((0, 0, 0), center)), width=5, arrow_size=10)
# view.add(arrow)

precession_semiangle = np.deg2rad(5)
PIXEL_SIZE = 0.03
PSF = kernel_blurred_disk(5, 0.5)
S_MAX = 0.005

diff = np.zeros(image_shape, dtype=np.float32)

prev_time = 0
count = 0


def update(event):
    """ """
    global wavelength, S_MAX, image, diff, prev_time, count

    time = event.elapsed % (4 * np.pi)

    # break in animation
    if time > 2 * np.pi:
        count = 0
        return

    nx, ny, _ = calculate_ewald_sphere_center(
        wavelength, precession_semiangle, event.elapsed
    )
    # get perpendicular axis
    px, py = ny, -nx
    rot = Rotation.from_axes_angles((px, py, 0), precession_semiangle)
    cone_verts_rot = ~rot * Vector3d(cone_verts)
    cone_mesh.set_data(vertices=cone_verts_rot.data, faces=cone.get_faces())
    # get vector perpendicular to this (projected onto xy)

    temp = generator.generate_templates(rot, S_MAX, flip=False)
    pattern = temp.generate_diffraction_pattern(
        image_shape,
        PIXEL_SIZE,
        psf=PSF,
        direct_beam=True,
    )

    if event.elapsed % (2 * np.pi) < 0.1:
        diff[:] = pattern.image
    else:
        diff += pattern.image
        # diff /= event.elapsed + 1

    count += 1
    image.set_data(diff / count)

    # arrow.set_data(pos=np.array(((0, 0, 0), new_center)))
    # sphere.transform = STTransform(translate=new_center)
    # time.sleep(0.2)


timer = app.Timer()
timer.connect(update)
timer.start()

if __name__ == "__main__" and sys.flags.interactive == 0:
    canvas.show()
    if sys.flags.interactive == 0:
        canvas.app.run()
