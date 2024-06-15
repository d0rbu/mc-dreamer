from manim import *
import numpy as np

class MinecraftVolume(ThreeDScene):
    def construct(self):
        # Define the 3D voxel space dimensions and the block types
        dim = 5
        voxel_size = 1.0  # Make cubes touch each other by setting the voxel size to 1.0

        # Create a 5x5x5 matrix with values representing air, dirt, grass, and stone
        minecraft_matrix = np.zeros((dim, dim, dim), dtype=int)
        for x in range(dim):
            for y in range(dim):
                for z in range(dim):
                    if z == 0:
                        minecraft_matrix[x, y, z] = 3  # Stone
                    elif z == 1 or z == 2:
                        minecraft_matrix[x, y, z] = 1  # Dirt
                    elif z == 3:
                        minecraft_matrix[x, y, z] = 2  # Grass
                    else:
                        minecraft_matrix[x, y, z] = 0  # Air

        # Define colors for some block types for visualization (air=0, dirt=1, grass=2, stone=3)
        block_colors = {
            0: WHITE,       # Air
            1: LIGHT_BROWN, # Dirt
            2: GREEN,       # Grass
            3: GRAY         # Stone
        }

        # Create a 3D grid of voxels
        voxels = VGroup()
        for x in range(dim):
            for y in range(dim):
                for z in range(dim):
                    block_type = minecraft_matrix[x, y, z]
                    color = block_colors[block_type]
                    cube = Cube(side_length=voxel_size)
                    cube.move_to(np.array([x, y, z]) * voxel_size)
                    cube.set_fill(color=color, opacity=0.1 if block_type == 0 else 0.5)
                    cube.set_stroke(color=BLACK, width=0.1)  # Add grid lines between cubes
                    voxels.add(cube)

        # Position the voxels in the scene
        voxels.move_to(ORIGIN)
        voxels.scale(0.5)

        # Add title text
        title = Text("5x5x5 Minecraft-like Volume").scale(0.7).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        # Display initial voxels
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.play(FadeIn(voxels))

        # Rotate the volume slowly for better visualization
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(10)
        self.stop_ambient_camera_rotation()

        # Convert the 3D matrix to a string format with ellipses
        matrix_string = (
            "[[[ 3,  3,  3, ...,  0,  0,  0],\n"
            "  [ 3,  3,  3, ...,  0,  0,  0],\n"
            "  [ 3,  3,  3, ...,  0,  0,  0],\n"
            "  ...,\n"
            "  [ 3,  3,  3, ...,  0,  0,  0],\n"
            "  [ 3,  3,  3, ...,  0,  0,  0],\n"
            "  [ 3,  3,  3, ...,  0,  0,  0]],\n\n"
            " [[ 1,  1,  1, ...,  0,  0,  0],\n"
            "  [ 1,  1,  1, ...,  0,  0,  0],\n"
            "  [ 1,  1,  1, ...,  0,  0,  0],\n"
            "  ...,\n"
            "  [ 1,  1,  1, ...,  0,  0,  0],\n"
            "  [ 1,  1,  1, ...,  0,  0,  0],\n"
            "  [ 1,  1,  1, ...,  0,  0,  0]],\n\n"
            " [[ 2,  2,  2, ...,  0,  0,  0],\n"
            "  [ 2,  2,  2, ...,  0,  0,  0],\n"
            "  [ 2,  2,  2, ...,  0,  0,  0],\n"
            "  ...,\n"
            "  [ 2,  2,  2, ...,  0,  0,  0],\n"
            "  [ 2,  2,  2, ...,  0,  0,  0],\n"
            "  [ 2,  2,  2, ...,  0,  0,  0]]]"
        )
        matrix_text = Text(matrix_string, font_size=24).scale(0.5)

        # Ensure the text faces the camera directly
        matrix_text.rotate(PI / 2, axis=UP)
        matrix_text.rotate(PI / 2, axis=RIGHT)

        # Transition the voxels to the matrix text
        self.play(Transform(voxels, matrix_text), run_time=2)
        self.wait(10)

if __name__ == "__main__":
    config.media_width = "80%"
    scene = MinecraftVolume()
    scene.render()
