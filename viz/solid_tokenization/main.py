import numpy as np
from manim import *
from itertools import product


VOLUME_SIZE = 8
TUBE_LENGTH = 4
FILL_OPACITY = 0.4
VOXELS_FILL_OPACITY = 0.1
VOXELS_STROKE_OPACITY = 1
CAMERA_ROTATION_RATE = 3 * DEGREES  # n degrees per second

TUBE_TOKENS = np.arange(TUBE_LENGTH ** 2).reshape(TUBE_LENGTH ** 2, 1).repeat(TUBE_LENGTH, axis=1)
for i in range(TUBE_LENGTH):
    TUBE_TOKENS[:, (-1-i)] = TUBE_TOKENS[:, (-1-i)] // (2 ** i) % 2

TUBE_TO_TOKENS = {
    tuple(tube.tolist()): token
    for token, tube in enumerate(TUBE_TOKENS)
}


class SolidTokenization(ThreeDScene):
    def construct(self):
        big_cube = Cube(side_length=VOLUME_SIZE, fill_opacity=FILL_OPACITY, stroke_width=2, stroke_color=WHITE)

        # set up axes with labels
        x_axis = VGroup(
            NumberLine(x_range=[0, VOLUME_SIZE, 1], color=RED).shift(DOWN * VOLUME_SIZE/2, IN * VOLUME_SIZE/2),
            NumberLine(x_range=[0, VOLUME_SIZE, 1], color=RED).rotate(90 * DEGREES, axis=RIGHT).shift(DOWN * VOLUME_SIZE/2, IN * VOLUME_SIZE/2),
        )
        x_label = Integer(VOLUME_SIZE, color=RED, font_size=12*VOLUME_SIZE).next_to(x_axis, DOWN * 2).rotate(-90 * DEGREES, axis=OUT).rotate(-90 * DEGREES, axis=UP)
        z_axis = VGroup(
            NumberLine(x_range=[0, VOLUME_SIZE, 1], color=GREEN).rotate(90 * DEGREES, axis=UP).shift(LEFT * VOLUME_SIZE/2, DOWN * VOLUME_SIZE/2),
            NumberLine(x_range=[0, VOLUME_SIZE, 1], color=GREEN).rotate(90 * DEGREES, axis=UP).rotate(90 * DEGREES, axis=OUT).shift(LEFT * VOLUME_SIZE/2, DOWN * VOLUME_SIZE/2),
        )
        z_label = Integer(VOLUME_SIZE, color=GREEN, font_size=12*VOLUME_SIZE).next_to(z_axis, DOWN * 2).rotate(-90 * DEGREES, axis=OUT).rotate(-90 * DEGREES, axis=UP)
        y_axis = VGroup(
            NumberLine(x_range=[0, VOLUME_SIZE, 1], color=BLUE).rotate(90 * DEGREES, axis=OUT).shift(LEFT * VOLUME_SIZE/2, IN * VOLUME_SIZE/2),
            NumberLine(x_range=[0, VOLUME_SIZE, 1], color=BLUE).rotate(90 * DEGREES, axis=OUT).rotate(90 * DEGREES, axis=UP).shift(LEFT * VOLUME_SIZE/2, IN * VOLUME_SIZE/2),
        )
        y_label = Integer(VOLUME_SIZE, color=BLUE, font_size=12*VOLUME_SIZE).next_to(y_axis, IN * 2).rotate(-90 * DEGREES, axis=OUT).rotate(-90 * DEGREES, axis=UP)
        side_axes = VGroup(
            x_axis,
            x_label,
            y_axis,
            y_label,
            z_axis,
            z_label,
        )
        self.add(big_cube, side_axes)

        # 3d rotation over big cube reprenting the volume
        self.begin_ambient_camera_rotation(rate=CAMERA_ROTATION_RATE)
        self.set_camera_orientation(phi=75 * DEGREES, theta=180 * DEGREES, focal_distance=VOLUME_SIZE * 3, zoom=3/VOLUME_SIZE)
        self.play(Write(big_cube), Write(side_axes), run_time=2)

        self.wait(5)  # 7 seconds

        # set up voxels in the volume
        cubes = [
            Cube(side_length=1, fill_opacity=0, stroke_width=1, stroke_color=WHITE).set_stroke(opacity=VOXELS_STROKE_OPACITY/VOLUME_SIZE)
            for _ in range(VOLUME_SIZE**3)
        ]
        XYZ_TO_IDX = {}
        for i, (z, y, x) in enumerate(product(range(VOLUME_SIZE), repeat=3)):
            cubes[i].shift((x - VOLUME_SIZE/2 + 0.5)* RIGHT, (y - VOLUME_SIZE/2 + 0.5) * UP, (z - VOLUME_SIZE/2 + 0.5) * OUT)
            XYZ_TO_IDX[(x, y, z)] = i

        self.add(*cubes)

        # set up big cube to fade out and emphasize the voxels in the volume
        big_cube.generate_target()
        big_cube.target.set_fill(opacity=FILL_OPACITY)

        # emphasize the voxels in the volume
        self.play(
            MoveToTarget(big_cube),
            FadeIn(VGroup(*cubes)),
            run_time=2
        )
        self.play(FadeOut(side_axes), run_time=2)

        self.wait(3)  # 14 seconds

        # set up faded/emphasized voxels
        CHOSEN_CUBE = (0, 0, -1)
        CHOSEN_CUBE = (CHOSEN_CUBE[0] % VOLUME_SIZE, CHOSEN_CUBE[1] % VOLUME_SIZE, CHOSEN_CUBE[2] % VOLUME_SIZE)
        CHOSEN_CUBE_IDX = XYZ_TO_IDX[CHOSEN_CUBE]
        non_chosen_cubes = VGroup(*[cube for i, cube in enumerate(cubes) if i != CHOSEN_CUBE_IDX])
        chosen_cube = cubes[CHOSEN_CUBE_IDX]
        chosen_cube.generate_target()
        chosen_cube.target.set_fill(opacity=FILL_OPACITY)
        chosen_cube.target.set_stroke(opacity=1.0)
        big_cube.target.set_stroke(opacity=0.0)
        big_cube.target.set_fill(opacity=0.0)

        # set up camera movement
        phi, theta, focal_distance, gamma, zoom = self.camera.get_value_trackers()
        # set up targets so we can return to this view later
        phi_stop_value = phi.get_value()
        theta_stop_value = theta.get_value()
        focal_distance_stop_value = focal_distance.get_value()
        zoom_stop_value = zoom.get_value()

        # focus on a particular voxel
        self.stop_ambient_camera_rotation()
        self.play(
            phi.animate.set_value(55 * DEGREES),
            theta.animate.set_value(225 * DEGREES),
            zoom.animate.set_value(5/VOLUME_SIZE),
            non_chosen_cubes.animate.set_stroke(opacity=VOXELS_STROKE_OPACITY/VOLUME_SIZE/4),
            MoveToTarget(chosen_cube),
            MoveToTarget(big_cube),
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

        # set up cubes to return to original view
        cubes_group = VGroup(*cubes)
        big_cube.target.set_stroke(opacity=1.0)
        big_cube.target.set_fill(opacity=FILL_OPACITY)

        # return camera to original view
        self.play(
            cubes_group.animate.set_stroke(opacity=VOXELS_STROKE_OPACITY/VOLUME_SIZE),
            MoveToTarget(big_cube),
            phi.animate.set_value(phi_stop_value),
            theta.animate.set_value(theta_stop_value),
            focal_distance.animate.set_value(focal_distance_stop_value),
            zoom.animate.set_value(zoom_stop_value),
            run_time=1,
        )
        self.begin_ambient_camera_rotation(rate=CAMERA_ROTATION_RATE)

        self.wait(3)  # 25 seconds

        # load example schematic to display with voxels
        example_schematic = np.load("example_schematic.npy")

        # set up display of example schematic
        voxel_values = []
        for z, y, x in product(range(VOLUME_SIZE), repeat=3):
            idx = XYZ_TO_IDX[(x, y, z)]
            if example_schematic[x, z, y] == 1:
                cubes_group.target[idx].set_fill(opacity=1.0)
                voxel_values.append(1)
            else:
                cubes_group.target[idx].set_fill(opacity=0.0)
                voxel_values.append(0)
            cubes_group.target.set_stroke(opacity=VOXELS_STROKE_OPACITY/VOLUME_SIZE/4)

        self.play(
            MoveToTarget(cubes_group),
            FadeOut(big_cube),
            run_time=1,
        )

        self.wait(3)  # 29 seconds

        # raise voxels in preparation for flattening
        self.play(
            cubes_group.animate.shift(OUT * (VOLUME_SIZE/2 + 1)),
            run_time=1,
        )

        for i, cube in enumerate(cubes):
            cube.generate_target()
            cube.target.move_to(ORIGIN + RIGHT * (i - VOLUME_SIZE/2 + 0.5))
            cube.target.set_stroke(opacity=1.0)

        # move voxels to show flattening
        self.play(LaggedStart(
            *[
                MoveToTarget(cube)
                for cube in cubes
            ],
            lag_ratio=1/(VOLUME_SIZE**3),
            run_time=8.5,
        ))

        self.stop_ambient_camera_rotation()  # 38.5 seconds

        # move camera to show flattened voxels
        self.play(
            theta.animate.set_value(270 * DEGREES),
            run_time=1,
        )

        self.wait(3)  # 42.5 seconds

        # split up flattened voxels into groups based on tube length
        tubes = []
        tubes_values = []
        for i in range((VOLUME_SIZE ** 3) // TUBE_LENGTH):
            tube = VGroup(*cubes[i*TUBE_LENGTH:(i+1)*TUBE_LENGTH])
            tube.generate_target()
            tube.target.shift(RIGHT * i)
            tubes.append(tube)
            tubes_values.append(voxel_values[i*TUBE_LENGTH:(i+1)*TUBE_LENGTH])

        # move tubes to show splitting
        self.play(
            *[
                MoveToTarget(tube)
                for tube in tubes
            ],
            run_time=2,
        )

        tube_labels = []
        for tube, tube_values in zip(tubes, tubes_values):
            token_idx = TUBE_TO_TOKENS[tuple(tube_values)]
            print(token_idx)
            label = Integer(token_idx, color=WHITE, font_size=96).next_to(tube, OUT * 3).rotate(90 * DEGREES, axis=RIGHT)
            tube_labels.append(label)
        tube_labels = VGroup(*tube_labels)

        # show token labels
        self.play(
            Write(tube_labels),
            run_time=1,
        )

        # move the tubes left multiple times
        for i in range(4):
            self.play(
                cubes_group.animate.shift(LEFT * (TUBE_LENGTH + 1)),
                tube_labels.animate.shift(LEFT * (TUBE_LENGTH + 1)),
                run_time=1,
            )

        self.wait(3)  # 52.5 seconds

        # replace the tubes with words as an example of tokenization
        example_sentence = "The quick brown fox jumps over the lazy dog"
        example_tokens = example_sentence.split()
        token_labels = []
        for i, token in enumerate(example_tokens):
            token_labels.append(Text(token, color=WHITE, font="Consolas", font_size=48).rotate(90 * DEGREES, axis=RIGHT).next_to(tubes[i], ORIGIN))

        self.play(LaggedStart(
            *[
                Transform(tube, label)
                for tube, label in zip(tubes, token_labels)
            ],
            lag_ratio=0.05,
            run_time=3,
        ))

        self.wait(3)  # 58.5 seconds

# TODO: another scene just quickly showing the token mappings from tubes to token indices
