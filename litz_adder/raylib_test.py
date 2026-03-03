"""
A quick test of Raylib in Python (via the pyray package;
use `pip install raylib` to install).
"""

import pyray as rl

rl.init_window(800, 600, "Hello Raylib")
while not rl.window_should_close():
	rl.begin_drawing()
	rl.clear_background(rl.RAYWHITE)
	rl.draw_text("Hello!", 190, 200, 20, rl.LIGHTGRAY)
	rl.end_drawing()
rl.close_window()