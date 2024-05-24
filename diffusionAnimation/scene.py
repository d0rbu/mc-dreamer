'''from manim import *
import numpy as np
import cv2

class ForwardDiffusionProcess(Scene):
    def construct(self):
        # Load the image
        img_path = "cat-0.png"  # Replace with the path to your image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

        # Convert image to Manim Mobject
        manim_image = ImageMobject(image)
        manim_image.scale(3.5)  # Scale the image to fit the scene

        # Add the initial image to the scene
        self.add(manim_image)
        self.wait(1)

        # Define the forward diffusion process
        def add_noise(image, sigma):
            noise = np.random.normal(0, sigma, image.shape)
            noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
            return noisy_image

        # Create animation frames
        num_frames = 20
        
        # First phase: Noise from 0 to 50
        sigma_values = np.linspace(0, 50, num_frames)
        prev_noise_image = image
        for sigma in sigma_values:
            noisy_image = add_noise(prev_noise_image, sigma)
            prev_noise_image = noisy_image
            manim_noisy_image = ImageMobject(noisy_image)
            manim_noisy_image.scale(3.5)
            self.add(manim_noisy_image)
            self.remove(manim_image)
            manim_image = manim_noisy_image
            self.wait(0.3)
        
        # Second phase: Noise from 50 to 100
        sigma_values = np.linspace(50, 100, num_frames)
        for sigma in sigma_values:
            noisy_image = add_noise(prev_noise_image, sigma)
            prev_noise_image = noisy_image
            manim_noisy_image = ImageMobject(noisy_image)
            manim_noisy_image.scale(3.5)
            self.add(manim_noisy_image)
            self.remove(manim_image)
            manim_image = manim_noisy_image
            self.wait(0.3)

        self.wait(2)

if __name__ == "__main__":
    config.media_width = "80%"
    scene = ForwardDiffusionProcess()
    scene.render()
'''

'''from manim import *
import numpy as np
import cv2

class ForwardDiffusionProcessDisplay(Scene):
    def construct(self):
        # Load the image
        img_path = "cat-0.png"  # Replace with the path to your image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

        # Convert image to Manim Mobject
        cat_image = ImageMobject(image)
        cat_image.scale(2)  # Scale the image to fit the scene
        cat_image.to_edge(LEFT)

        # Add the initial image to the scene
        self.add(cat_image)
        self.wait(1)

        # Define the forward diffusion process
        def add_noise(image, sigma):
            noise = np.random.normal(0, sigma, image.shape)
            noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
            return noisy_image

        # Create the first noised image
        sigma_first = 30
        first_noised_image = add_noise(image, sigma_first)
        manim_first_noised_image = ImageMobject(first_noised_image)
        manim_first_noised_image.scale(2)
        manim_first_noised_image.next_to(cat_image, RIGHT, buff=1.5)

        # Create the fully noised image
        # First phase: Noise from 0 to 50
        sigma_values = np.linspace(0, 50, 20)
        prev_noise_image = image
        for sigma in sigma_values:
            noisy_image = add_noise(prev_noise_image, sigma)
            prev_noise_image = noisy_image
            manim_noisy_image = ImageMobject(noisy_image)
        
        # Second phase: Noise from 50 to 100
        sigma_values = np.linspace(50, 100, 20)
        
        for sigma in sigma_values:
            noisy_image = add_noise(prev_noise_image, sigma)
            prev_noise_image = noisy_image
            manim_noisy_image = ImageMobject(noisy_image)

        # Draw arrows and ellipses
        arrow1 = Arrow(cat_image.get_right(), manim_first_noised_image.get_left(), buff=0.2)
        ellipses = VGroup(Dot(), Dot(), Dot()).arrange(RIGHT, buff=0.5).next_to(manim_first_noised_image, RIGHT, buff=1.5)
        arrow2 = Arrow(manim_first_noised_image.get_right(), ellipses.get_left(), buff=0.2)
        arrow3 = Arrow(ellipses.get_right(), manim_noisy_image.get_left(), buff=0.2)

        # Add text at the bottom
        text = Text("Forward Diffusion Process").to_edge(DOWN)

        # Add elements to the scene with animations
        self.play(Write(text))
        self.wait(2)
        self.play(GrowArrow(arrow1), FadeIn(manim_first_noised_image))
        self.wait(0.5)
        self.play(GrowArrow(arrow2), FadeIn(ellipses))
        self.wait(0.5)
        self.play(GrowArrow(arrow3), FadeIn(manim_noisy_image))
        self.wait(0.5)
        

if __name__ == "__main__":
    config.media_width = "80%"
    scene = ForwardDiffusionProcessDisplay()
    scene.render()'''


'''from manim import *
import numpy as np
import cv2

class ForwardDiffusionProcessDisplay(Scene):
    def construct(self):
        # Load the image
        img_path = "cat-0.png"  # Replace with the path to your image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

        # Define the forward diffusion process
        def add_noise(image, sigma):
            noise = np.random.normal(0, sigma, image.shape)
            noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
            return noisy_image

        # Create the fully noised image
        # First phase: Noise from 0 to 50
        sigma_values = np.linspace(0, 50, 20)
        prev_noise_image = image
        for sigma in sigma_values:
            noisy_image = add_noise(prev_noise_image, sigma)
            prev_noise_image = noisy_image
            manim_noisy_image = ImageMobject(noisy_image)
        
        # Second phase: Noise from 50 to 100
        sigma_values = np.linspace(50, 100, 30)
        
        for sigma in sigma_values:
            noisy_image = add_noise(prev_noise_image, sigma)
            prev_noise_image = noisy_image
            manim_noisy_image = ImageMobject(noisy_image)

        # Create the noised images
        num_images = 6
        sigma_values = np.linspace(0, 100, num_images)
        images = [ImageMobject(add_noise(image, sigma)).scale(1.5) for sigma in sigma_values]
        images[5] = manim_noisy_image.scale(1.5)
        
        # Position the images in a row
        images_group = Group(*images).arrange(RIGHT, buff=1.0)
        images_group.scale_to_fit_width(config.frame_width - 1)

        # Draw a blue arrow below the images
        arrow = Arrow(
            start=images_group.get_left() + DOWN * 1.5, 
            end=images_group.get_right() + DOWN * 1.5, 
            color=BLUE
        )

        # Add text at the bottom
        text = Text("Forward Diffusion Process").to_edge(DOWN)

        # Add elements to the scene with animations
        self.play(FadeIn(images_group))
        
        self.play(GrowArrow(arrow))
        self.play(Write(text))
        self.wait(2)

if __name__ == "__main__":
    config.media_width = "80%"
    scene = ForwardDiffusionProcessDisplay()
    scene.render()


'''
from manim import *
import numpy as np
import cv2

class ForwardAndBackwardDiffusionProcessDisplay(Scene):
    def construct(self):
        # Load the image
        img_path = "cat-0.png"  # Replace with the path to your image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

        # Define the forward diffusion process
        def add_noise(image, sigma):
            noise = np.random.normal(0, sigma, image.shape)
            noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
            return noisy_image

        # Create the noised images for forward diffusion
        num_images = 6
        sigma_values = np.linspace(0, 100, num_images)
        forward_images = [ImageMobject(add_noise(image, sigma)).scale(1.5) for sigma in sigma_values]

        # Position the images in a row
        forward_images_group = Group(*forward_images).arrange(RIGHT, buff=1.0)
        forward_images_group.scale_to_fit_width(config.frame_width - 1)

        # Draw a blue arrow below the images
        forward_arrow = Arrow(
            start=forward_images_group.get_left() + DOWN * 1.5, 
            end=forward_images_group.get_right() + DOWN * 1.5, 
            color=BLUE
        )

        # Add text at the bottom
        forward_text = Text("Forward Diffusion Process").to_edge(DOWN)

        # Add noise formula
        noise_formula = MathTex(
            r"q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)"
        ).to_edge(UP)
        
        beta_progression = MathTex(
            r"\beta_1 < \beta_2 < \cdots < \beta_T",
            color=GREEN
        ).to_edge(UP).shift(2*DOWN)

        # Display forward diffusion process
        self.play(Write(forward_text), Write(noise_formula))
        self.play(FadeIn(forward_images_group), GrowArrow(forward_arrow), Write(beta_progression))
        self.wait(2)

        # Prepare for backward diffusion process
        self.play(FadeOut(forward_images_group), FadeOut(forward_arrow), FadeOut(forward_text), FadeOut(noise_formula), FadeOut(beta_progression))

        # Create the noised images for backward diffusion
        backward_images = list(reversed(forward_images))

        # Position the images in a row
        backward_images_group = Group(*backward_images).arrange(RIGHT, buff=1.0)
        backward_images_group.scale_to_fit_width(config.frame_width - 1)

        # Draw a green arrow below the images for backward process
        backward_arrow = Arrow(
            start=backward_images_group.get_left() + DOWN * 1.5, 
            end=backward_images_group.get_right() + DOWN * 1.5, 
            color=GREEN
        )

        # Add text at the bottom for backward process
        backward_text = Text("Backward Diffusion Process").to_edge(DOWN)

        # Add mathematical explanation
        minimize_formula = MathTex(
            r"L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta \left( \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t \right) \right\|^2 \right]"
        ).to_edge(UP)

        # Text explaining epsilon and predicted noise
        epsilon_text = MathTex(r"\text{where } \epsilon \text{ is noise}").scale(0.7).next_to(minimize_formula, DOWN)
        predicted_noise_text = MathTex(r"\epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \text{ is predicted noise by the model}").scale(0.7).next_to(epsilon_text, DOWN)

        # Display backward diffusion process with math explanation
        self.play(FadeIn(backward_images_group), GrowArrow(backward_arrow), Write(backward_text))
        self.wait(1)
        self.play(Write(minimize_formula))
        self.wait(1)
        self.play(Write(epsilon_text))
        self.play(Write(predicted_noise_text))
        self.wait(2)

if __name__ == "__main__":
    config.media_width = "80%"
    scene = ForwardAndBackwardDiffusionProcessDisplay()
    scene.render()


