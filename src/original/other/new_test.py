from manim import *

class SquareAnimation(Scene):
    def construct(self):
        square = Square(fill_opacity=1, fill_color=BLUE)
        self.play(Create(square))
        self.wait()
        self.play(square.animate.rotate(PI/4))
        self.wait()
