""" Configuration options for displaying camera feedback.
"""
import os
import numpy as np
import cv2

image_shape = (800, 800, 3)
image_fps = 30

bb_fps_const = 0.1
bb_colors = [(255, 255,255), (0, 0, 255)] # unripe, ripe
line_type = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA

reward_animation_steps = 20
reward_animation_offset = np.square(10*np.sin(-np.pi/3.0))

error_loc = (7, image_shape[1] - 7)
error_radius = 5

arrow_center = np.array([image_shape[0]//2, image_shape[1] - 75])
arrow_base_len = 30.0
arrow_base_width = 15.0
arrow_tip_len = 15.0
arrow_tip_width = 25.0
arrow_base_offset = 10.0
