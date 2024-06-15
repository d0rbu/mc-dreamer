from manim import *

class LaTeXExample(Scene):
    def construct(self):
        text = MathTex(r"E = mc^2")
        self.play(Write(text))
        self.wait(2)

if __name__ == "__main__":
    config.media_width = "80%"
    scene = LaTeXExample()
    scene.render()
