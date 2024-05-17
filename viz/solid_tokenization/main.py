import numpy as np
from manim import *
from itertools import product


VOLUME_SIZE = 16
TUBE_LENGTH = 8
FILL_OPACITY = 0.4
CAMERA_ROTATION_RATE = 3 * DEGREES  # n degrees per second


class main(ThreeDScene):
    def construct(self):
        big_cube = Cube(side_length=VOLUME_SIZE, fill_opacity=FILL_OPACITY, stroke_width=2, stroke_color=WHITE)

        # set up axes with labels
        x_axis = VGroup(
            NumberLine(x_range=[0, VOLUME_SIZE, 1], color=RED).shift(DOWN * VOLUME_SIZE/2, IN * VOLUME_SIZE/2),
            NumberLine(x_range=[0, VOLUME_SIZE, 1], color=RED).rotate(90 * DEGREES, axis=RIGHT).shift(DOWN * VOLUME_SIZE/2, IN * VOLUME_SIZE/2),
        )
        x_label = Integer(VOLUME_SIZE, color=RED).next_to(x_axis, DOWN).rotate(-90 * DEGREES, axis=OUT).rotate(-90 * DEGREES, axis=UP)
        z_axis = VGroup(
            NumberLine(x_range=[0, VOLUME_SIZE, 1], color=GREEN).rotate(90 * DEGREES, axis=UP).shift(LEFT * VOLUME_SIZE/2, DOWN * VOLUME_SIZE/2),
            NumberLine(x_range=[0, VOLUME_SIZE, 1], color=GREEN).rotate(90 * DEGREES, axis=UP).rotate(90 * DEGREES, axis=OUT).shift(LEFT * VOLUME_SIZE/2, DOWN * VOLUME_SIZE/2),
        )
        z_label = Integer(VOLUME_SIZE, color=GREEN).next_to(z_axis, DOWN).rotate(-90 * DEGREES, axis=OUT).rotate(-90 * DEGREES, axis=UP)
        y_axis = VGroup(
            NumberLine(x_range=[0, VOLUME_SIZE, 1], color=BLUE).rotate(90 * DEGREES, axis=OUT).shift(LEFT * VOLUME_SIZE/2, IN * VOLUME_SIZE/2),
            NumberLine(x_range=[0, VOLUME_SIZE, 1], color=BLUE).rotate(90 * DEGREES, axis=OUT).rotate(90 * DEGREES, axis=UP).shift(LEFT * VOLUME_SIZE/2, IN * VOLUME_SIZE/2),
        )
        y_label = Integer(VOLUME_SIZE, color=BLUE).next_to(y_axis, LEFT).rotate(-90 * DEGREES, axis=OUT).rotate(-90 * DEGREES, axis=UP)
        side_axes = VGroup(
            x_axis,
            x_label,
            y_axis,
            y_label,
            z_axis,
            z_label,
        )

        # 3d rotation over big cube reprenting the volume
        self.begin_ambient_camera_rotation(rate=CAMERA_ROTATION_RATE)
        self.set_camera_orientation(phi=75 * DEGREES, theta=180 * DEGREES, focal_distance=VOLUME_SIZE * 3)

        self.play(Write(big_cube), Write(side_axes), run_time=2)

        self.wait(5)  # 7 seconds

        # set up voxels in the volume
        cubes = [
            Cube(side_length=1, fill_opacity=FILL_OPACITY/VOLUME_SIZE, stroke_width=1, stroke_color=WHITE)
            for _ in range(VOLUME_SIZE**3)
        ]
        XYZ_TO_IDX = {}
        for i, (z, y, x) in enumerate(product(range(VOLUME_SIZE), repeat=3)):
            cubes[i].shift((x - VOLUME_SIZE/2 + 0.5)* RIGHT, (y - VOLUME_SIZE/2 + 0.5) * UP, (z - VOLUME_SIZE/2 + 0.5) * OUT)
            XYZ_TO_IDX[(x, y, z)] = i

        # emphasize the voxels in the volume
        self.play(FadeOut(big_cube), FadeIn(VGroup(*cubes)), run_time=2)
        self.play(FadeOut(side_axes), run_time=2)

        self.wait(3)  # 14 seconds

        # set up faded/emphasized voxels
        CHOSEN_CUBE = (0, 0, -1)
        CHOSEN_CUBE = (CHOSEN_CUBE[0] % VOLUME_SIZE, CHOSEN_CUBE[1] % VOLUME_SIZE, CHOSEN_CUBE[2] % VOLUME_SIZE)
        CHOSEN_CUBE_IDX = XYZ_TO_IDX[CHOSEN_CUBE]
        non_chosen_cubes = VGroup(*[cube for i, cube in enumerate(cubes) if i != CHOSEN_CUBE_IDX])
        non_chosen_cubes.generate_target()
        non_chosen_cubes.target.set_fill(opacity=FILL_OPACITY/VOLUME_SIZE/6)
        non_chosen_cubes.target.set_stroke(opacity=0.1)
        chosen_cube = cubes[CHOSEN_CUBE_IDX]

        # set up camera movement
        phi, theta, focal_distance, gamma, zoom = self.camera.get_value_trackers()
        # set up targets so we can return to this view later
        phi_stop_value = phi.get_value()
        theta_stop_value = theta.get_value()
        focal_distance_stop_value = focal_distance.get_value()

        # focus on a particular voxel
        self.stop_ambient_camera_rotation()
        self.play(
            phi.animate.set_value(55 * DEGREES),
            theta.animate.set_value(225 * DEGREES),
            focal_distance.animate.set_value(2 * VOLUME_SIZE),
            MoveToTarget(non_chosen_cubes),
            chosen_cube.animate.set_fill(opacity=0.4),
            run_time=1,
        )

        self.wait(3)  # 18 seconds

        self.play(
            chosen_cube.animate.set_fill(opacity=1.0),
            run_time=0.6
        )

        self.wait(0.6)

        self.play(
            chosen_cube.animate.set_fill(opacity=0.0),
            run_time=0.6
        )

        self.wait(1.2)  # 21 seconds

        # set up target for cubes to return to original view
        cubes_group = VGroup(*cubes)
        cubes_group.generate_target()
        cubes_group.target.set_fill(opacity=FILL_OPACITY/VOLUME_SIZE)
        cubes_group.target.set_stroke(opacity=1)

        # return camera to original view
        self.play(
            MoveToTarget(cubes_group),
            phi.animate.set_value(phi_stop_value),
            theta.animate.set_value(theta_stop_value),
            focal_distance.animate.set_value(focal_distance_stop_value),
            run_time=1,
        )
        self.begin_ambient_camera_rotation(rate=CAMERA_ROTATION_RATE)

        self.wait(3)  # 25 seconds

        # load example schematic to display with voxels
        example_schematic = np.load("example_schematic.npy")

        # set up display of example schematic
        for z, y, x in product(range(VOLUME_SIZE), repeat=3):
            idx = XYZ_TO_IDX[(x, y, z)]
            if example_schematic[z, y, x] == 1:
                cubes_group.target[idx].set_fill(opacity=1.0)
            else:
                cubes_group.target[idx].set_fill(opacity=0.0)

        self.play(
            MoveToTarget(cubes_group),
            run_time=1,
        )

        self.wait(3)  # 29 seconds

        # set up voxels to show flattening
        for i in range(len(cubes)):
            cubes_group.target[i].move_to(ORIGIN + RIGHT * i)

        self.play(
            MoveToTarget(cubes_group),
            run_time=4,
        )
