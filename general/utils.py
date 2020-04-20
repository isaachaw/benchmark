import os
from typing import List

import cv2
import numpy as np


def get_all_file_paths(path_base: str, extension: str=None) -> List[str]:
    file_paths = []
    for dir_path, dir_names, file_names in os.walk(path_base):
        for file_name in file_names:
            if not extension or file_name.lower().endswith(extension):
                file_paths.append(os.path.join(dir_path, file_name))
    return file_paths


def read_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    return img


def rescale(img: np.ndarray, scale=1.) -> np.ndarray:
    if scale != 1.:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    return img
