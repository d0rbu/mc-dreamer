from manim import *
import numpy as np

class VoxelDiffusion(ThreeDScene):
    def construct(self):
        # Define the 3D voxel space dimensions
        dim = 5
        voxel_size = 0.5

        # Create the initial voxel colors (randomly assigned)
        voxel_colors = np.random.rand(dim, dim, dim, 3)

        # Create a 3D grid of voxels
        voxels = VGroup()
        for x in range(dim):
            for y in range(dim):
                for z in range(dim):
                    # Create a cube for each voxel
                    cube = Cube(side_length=voxel_size)
                    cube.move_to(np.array([x, y, z]) * voxel_size * 1.2)
                    color = rgb_to_color(voxel_colors[x, y, z])
                    if(x > 0 and y >0 and z >0):
                        cube.set_fill(color=color, opacity=0.15)
                    else:
                        cube.set_fill(color=color, opacity=0.9)
                    cube.set_stroke(color=color, width=0.1)
                    voxels.add(cube)

        # Position the voxels in the scene
        voxels.move_to(ORIGIN)
        voxels.scale(0.5)

        # Add title text
        title = Text("3D Voxel Diffusion Process").scale(0.7)
        self.add_fixed_in_frame_mobjects(title)
        title.to_edge(UP)

        # Display initial voxels
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.play(Write(title))
        self.play(FadeIn(voxels))

        # Define the final structure with gray bottom layer, white walls, and black corner pillars
        target_colors = np.zeros((dim, dim, dim, 3))
        for x in range(dim):
            for y in range(dim):
                for z in range(dim):
                    if z == 0:
                        target_colors[x, y, z] = [0, .6, 0]  # Green bottom layer
                    elif (x == 0 or x == dim-1) and (y == 0 or y == dim-1):
                        target_colors[x, y, z] = [0, 0, 0]  # Black corner pillars
                    else:
                        target_colors[x, y, z] = [0.4, 0.2, 0]  # Brown walls

        num_steps = 10
        self.begin_ambient_camera_rotation(rate=0.05)
        for step in range(num_steps):
            voxel_colors = np.random.rand(dim, dim, dim, 3)
            alpha = step / num_steps
            new_colors = (1 - alpha) * voxel_colors + alpha * target_colors
            more_rand_color = (1 - alpha/2) * voxel_colors + alpha*2 * target_colors
            new_voxels = VGroup()
            for x in range(dim):
                for y in range(dim):
                    for z in range(dim):
                        new_cube = Cube(side_length=voxel_size)
                        new_cube.move_to(np.array([x, y, z]) * voxel_size * 1.2)
                        new_color = rgb_to_color(new_colors[x, y, z])
                        rand_c = rgb_to_color(more_rand_color[x, y, z])
                        if(x > 0 and y >0 and z >0):
                            new_cube.set_fill(color=rand_c, opacity=0.15)
                        else:
                            new_cube.set_fill(color=new_color, opacity=0.9)
                        new_cube.set_stroke(color=new_color, width=0.1)
                        new_voxels.add(new_cube)
            new_voxels.move_to(ORIGIN)
            new_voxels.scale(0.5)
            self.play(Transform(voxels, new_voxels), run_time=0.5)
            self.wait(0.5)

        
        
        
        # Highlight the air blocks within a 4x4x4 cube with a specific color
        # air_blocks = VGroup(new_voxels)
        # for x in range(1, dim-1):
        #     for y in range(1, dim-1):
        #         for z in range(1, dim-1):
        #             air_cube = Cube(side_length=voxel_size)
        #             air_cube.move_to(np.array([x, y, z]) * voxel_size * 1.2)
        #             air_cube.set_fill(color=GRAY, opacity=0.2)
        #             air_cube.set_stroke(color=GRAY, width=0.1)
        #             air_blocks.add(air_cube)
        # air_blocks.move_to(ORIGIN)
        # air_blocks.scale(0.5)

        # Explanation for air blocks
        air_text = Text("Air blocks are not optimized for color").scale(0.7)
        self.add_fixed_in_frame_mobjects(air_text)
        air_text.to_edge(DOWN)
        
        self.play(Write(air_text)) # FadeIn(air_blocks), 
        self.wait(2)

        # Lower the opacity of a 4x4x4 volume in the top corner
        # top_corner_blocks = VGroup()
        # for x in range(dim-4, dim):
        #     for y in range(dim-4, dim):
        #         for z in range(dim-4, dim):
        #             top_cube = Cube(side_length=voxel_size)
        #             top_cube.move_to(np.array([x, y, z]) * voxel_size * 1.2)
        #             top_cube.set_fill(color=WHITE, opacity=0.1)
        #             top_cube.set_stroke(color=WHITE, width=0.1)
        #             top_corner_blocks.add(top_cube)
        # top_corner_blocks.move_to(ORIGIN)
        # top_corner_blocks.scale(0.5)
        
        

        # Fade out everything
        self.play(FadeOut(voxels),  FadeOut(air_text), FadeOut(title)) # FadeOut(top_corner_blocks),  FadeOut(air_blocks),
        self.stop_ambient_camera_rotation()
        
        
        self.wait(2)

if __name__ == "__main__":
    config.media_width = "80%"
    scene = VoxelDiffusion()
    scene.render()
