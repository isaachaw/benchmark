import cv2

from image_utils import *
from generate import generate
from main import get_random_bg_img, get_random_fg_img
from plot import draw_iou

import matplotlib.pyplot as plt


if __name__ == "__main__":
    options = GenerateOptions()
    options.glare_img_path = './glare_images'
    options.random_distortion = False
        
    while True:
        fg_img = get_random_fg_img()
        bg_img = get_random_bg_img()

        new_img, points = generate(bg_img, fg_img, options)
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

        img_h, img_w = new_img.shape[:2]

        prediction_points = points + [int(img_w * 0.1), int(img_h * 0.1)]

        polygon = sort_points_clockwise(points)
        prediction_polygon = sort_points_clockwise(prediction_points)

        i_area, i_points = calculate_intersection(polygon, prediction_polygon)
        u_area, u_points = calculate_union(polygon, prediction_polygon)

        draw_iou(new_img, polygon, prediction_polygon, i_points)
    