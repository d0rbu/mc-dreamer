from manim import *
import numpy as np

class MinecraftVolume(ThreeDScene):
    def construct(self):
        # Define the 3D voxel space dimensions and the block types
        dim = 5
        voxel_size = 1.0  # Make cubes touch each other by setting the voxel size to 1.0

        # Create a 5x5x5 matrix with values ranging from 0 to 255
        minecraft_matrix = np.random.randint(0, 256, (dim, dim, dim))

        # Define colors for some block types for visualization (air=0, dirt=1, grass=2, stone=3)
        block_colors = {
            0: WHITE,     # Air
            1: LIGHT_BROWN,     # Dirt
            2: GREEN,     # Grass
            3: GRAY       # Stone
        }

        # Create a 3D grid of voxels
        voxels = VGroup()
        for x in range(dim):
            for y in range(dim):
                for z in range(dim):
                    block_type = minecraft_matrix[x, y, z]
                    color = block_colors.get(block_type % 4, BLUE)  # Use modulo for simplicity
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

        # Ending
        self.play(FadeOut(voxels), FadeOut(title))
        self.wait(1)

if __name__ == "__main__":
    config.media_width = "80%"
    scene = MinecraftVolume()
    scene.render()
