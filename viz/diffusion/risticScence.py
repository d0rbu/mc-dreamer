from manim import *
import numpy as np

class InterestingHeuristic(Scene):
    def construct(self):
        # Initial matrix representation as a string
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

        # Display the initial matrix
        self.play(Write(matrix_text))
        self.wait(2)

        # Add text explaining the weights
        explanation_text = Text(
            "Assigning weights to each block ID:\n"
            "[0, 0.1, 0.1, 0.2 , ...] for [Air, Dirt/Stone, Grass, ...]",
            font_size=24
        ).to_edge(DOWN)

        self.play(Write(explanation_text))
        self.wait(2)

        # Define the matrix with interesting values
        minecraft_matrix = np.zeros((5, 5, 5), dtype=int)
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if z == 0:
                        minecraft_matrix[x, y, z] = 3  # Stone
                    elif z == 1 or z == 2:
                        minecraft_matrix[x, y, z] = 1  # Dirt
                    elif z == 3:
                        minecraft_matrix[x, y, z] = 2  # Grass
                    else:
                        minecraft_matrix[x, y, z] = 0  # Air

        # Transform values to their interesting scores
        interest_values = np.zeros_like(minecraft_matrix, dtype=float)
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if minecraft_matrix[x, y, z] == 0:  # Air
                        interest_values[x, y, z] = 0.0
                    elif minecraft_matrix[x, y, z] in [1, 3]:  # Dirt or Stone
                        interest_values[x, y, z] = 0.1
                    elif minecraft_matrix[x, y, z] == 2:  # Grass
                        interest_values[x, y, z] = 0.2

        # Convert interest values to a string format
        interest_string = (
            "[[[ 0.1,  0.1,  0.1, ...,  0.0,  0.0,  0.0],\n"
            "  [ 0.1,  0.1,  0.1, ...,  0.0,  0.0,  0.0],\n"
            "  [ 0.1,  0.1,  0.1, ...,  0.0,  0.0,  0.0],\n"
            "  ...,\n"
            "  [ 0.1,  0.1,  0.1, ...,  0.0,  0.0,  0.0],\n"
            "  [ 0.1,  0.1,  0.1, ...,  0.0,  0.0,  0.0],\n"
            "  [ 0.1,  0.1,  0.1, ...,  0.0,  0.0,  0.0]],\n\n"
            " [[ 0.1,  0.1,  0.1, ...,  0.0,  0.0,  0.0],\n"
            "  [ 0.1,  0.1,  0.1, ...,  0.0,  0.0,  0.0],\n"
            "  [ 0.1,  0.1,  0.1, ...,  0.0,  0.0,  0.0],\n"
            "  ...,\n"
            "  [ 0.1,  0.1,  0.1, ...,  0.0,  0.0,  0.0],\n"
            "  [ 0.1,  0.1,  0.1, ...,  0.0,  0.0,  0.0],\n"
            "  [ 0.1,  0.1,  0.1, ...,  0.0,  0.0,  0.0]],\n\n"
            " [[ 0.2,  0.2,  0.2, ...,  0.0,  0.0,  0.0],\n"
            "  [ 0.2,  0.2,  0.2, ...,  0.0,  0.0,  0.0],\n"
            "  [ 0.2,  0.2,  0.2, ...,  0.0,  0.0,  0.0],\n"
            "  ...,\n"
            "  [ 0.2,  0.2,  0.2, ...,  0.0,  0.0,  0.0],\n"
            "  [ 0.2,  0.2,  0.2, ...,  0.0,  0.0,  0.0],\n"
            "  [ 0.2,  0.2,  0.2, ...,  0.0,  0.0,  0.0]]]"
        )
        interest_text = Text(interest_string, font_size=24).scale(0.5)

        # Transform the original matrix text to interest values text
        self.play(Transform(matrix_text, interest_text), run_time=2)
        self.wait(2)

        # Draw lines converging to the right edge of the transformed matrix
        lines = VGroup()
        for i in range(5):
            line = Line(
                start=interest_text.get_right() + RIGHT * 0.2 * 1.1 + .9 * UP * (i - 2),
                end=interest_text.get_right() + RIGHT * 2,
                stroke_width=0.75
            )
            lines.add(line)
        self.play(*[Create(line) for line in lines])

        # Calculate the average interest score
        avg_interest = np.mean(interest_values)
        avg_interest_text = Text(f"Interesting score: {avg_interest:.2f}", font_size=24).next_to(lines, RIGHT)

        # Animate the calculation of the average interest
        self.play(Write(avg_interest_text))
        self.wait(2)

        # Ending
        self.play(FadeOut(interest_text), FadeOut(avg_interest_text), FadeOut(explanation_text), *[FadeOut(line) for line in lines])

        # Add the part for interesting_solid_ratio()
        self.interesting_solid_ratio_heuristic(minecraft_matrix, avg_interest)

    def interesting_solid_ratio_heuristic(self, minecraft_matrix, avg_interest):
        # Explanation text for interesting_solid_ratio
        explanation_text = Text(
            "Calculating interesting_solid_ratio:\n",
            # font_size= 24 
            font_size= 32
        ).to_edge(UP) 
        step_text = Text("Step 1: Count the number of non-air blocks", font_size=24).to_edge(DOWN)
        self.play(Write(explanation_text), Write(step_text))
        self.wait(2)

        # Highlight and remove air blocks in the matrix
        non_air_matrix = np.copy(minecraft_matrix)
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if minecraft_matrix[x, y, z] == 0:  # Air
                        non_air_matrix[x, y, z] = -1

        non_air_string = str(non_air_matrix).replace('-1', ' ')
        non_air_text = Text(non_air_string, font_size=24).scale(0.5)

        self.play(Transform(self.mobjects[0], non_air_text), run_time=2)
        self.wait(2)

        # Calculate the number of non-air blocks
        num_non_air_blocks = np.sum(non_air_matrix != -1)
        num_non_air_text = Text(f"Number of non-air blocks: {num_non_air_blocks}", font_size=24).to_edge(RIGHT)

        self.play(Write(num_non_air_text))
        self.wait(2)

        # Explanation for Step 2
        explanation_text_step2 = Text(
            "interesting_solid_ratio = Average Interest / Number of non-air blocks",
            font_size=24
        ).to_edge(DOWN)

        self.play(Transform(step_text, explanation_text_step2))
        self.wait(2)

        # Calculate interesting_solid_ratio
        if num_non_air_blocks > 0:
            interesting_solid_ratio = avg_interest / num_non_air_blocks
        else:
            interesting_solid_ratio = 0.0
        interesting_solid_ratio_text = Text(f"interesting_solid_ratio: {interesting_solid_ratio:.4f}", font_size=24).to_edge(RIGHT).shift(DOWN)

        self.play(Write(interesting_solid_ratio_text))
        self.wait(2)

        # Ending
        self.play(FadeOut(num_non_air_text), FadeOut(interesting_solid_ratio_text), FadeOut(explanation_text_step2), FadeOut(step_text), FadeOut(explanation_text))

        # Add the part for fewer_blocks()
        self.fewer_blocks_heuristic(minecraft_matrix)

    def fewer_blocks_heuristic(self, minecraft_matrix):
        # Explanation text for fewer_blocks
        explanation_text = Text(
            "Calculating fewer_blocks heuristic:\n",
            font_size=32
        ).to_edge(UP)
        step_text = Text("Step 1: Calculate unique_block_ratio", font_size=24).to_edge(DOWN)
        self.play(Write(explanation_text), Write(step_text))
        self.wait(2)

        # Step 1: Calculate unique_block_ratio
        unique_blocks = np.unique(minecraft_matrix)
        unique_block_ratio = len(unique_blocks) / 256
        unique_block_ratio_text = Text(f"unique_block_ratio: {unique_block_ratio:.4f}", font_size=24).to_edge(RIGHT)

        self.play(Write(unique_block_ratio_text))
        self.wait(2)

        # Explanation for Step 2
        explanation_text_step2 = Text(
            "Step 2: Apply fewer_blocks heuristic\n"
            "Visualized with a graph:",
            font_size=24
        ).to_edge(DOWN)

        self.play(Transform(step_text, explanation_text_step2))
        self.wait(2)

        # Remove the matrix before drawing the graph
        self.play(FadeOut(self.mobjects[0]))

        # Define the graph vertices for the trapezoid
        good_ratio = 14 / 256
        bad_ratio = 50 / 256
        vertices = [
            [0, 0, 0],  # bottom-left
            [good_ratio, 1, 0],  # top-left
            [bad_ratio, 1, 0],  # top-right
            [1, 0, 0]  # bottom-right
        ]

        # Draw the trapezoid graph
        graph = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1, 0.2],
            x_length=7,
            y_length=4.5,
            axis_config={"include_numbers": True}
        )

        trapezoid = Polygon(*[graph.c2p(*vertex) for vertex in vertices], color=WHITE, fill_opacity=0.0)
        self.play(Create(graph), Create(trapezoid))

        # Add the fewer_blocks score point on the graph
        if unique_block_ratio <= good_ratio:
            fewer_blocks_score = unique_block_ratio / good_ratio
            point = Dot(graph.c2p(unique_block_ratio, fewer_blocks_score), color=RED)
        elif unique_block_ratio >= bad_ratio:
            fewer_blocks_score = 1 - (unique_block_ratio - bad_ratio)
            point = Dot(graph.c2p(unique_block_ratio, fewer_blocks_score), color=RED)
        else:
            fewer_blocks_score = 1.0
            point = Dot(graph.c2p(unique_block_ratio, fewer_blocks_score), color=RED)

        point_label = Text(f"({unique_block_ratio:.2f}, {fewer_blocks_score:.2f})", font_size=18).next_to(point, UP)
        self.play(Create(point), Write(point_label))
        Wait(2)
        self.play(FadeOut(trapezoid), FadeOut(point), FadeOut(point_label), FadeOut(graph), FadeOut(unique_block_ratio_text), FadeOut(step_text), FadeOut(explanation_text_step2))

if __name__ == "__main__":
    config.media_width = "80%"
    scene = InterestingHeuristic()
    scene.render()
