from collections import namedtuple
import os
from typing import List, Tuple
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Base_Path = "C:/workspace/consolsys/InnoLab/benchmark"
Base_Path = "D:/workspace/benchmark"
Region = namedtuple("Region", ("top", "left", "bottom", "right"))
Dimensions = namedtuple("Shape", ("height", "width"))

class OptionsA(object):
    def __init__(self):
        self.random_brightness = True
        self.random_glare = True
        self.random_transform = True
        self.random_distortion = True


def change_brightness(img: np.ndarray, value=0) -> np.ndarray:
    if value != 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value > 0:
            v = np.where(v < 255 - value, v + value, 255)
        else:
            v = np.where(v > -value, v + value, 0)
        v = v.astype(np.uint8)
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def blur(img: np.ndarray, kernel_size=5) -> np.ndarray:
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)
    return img


def rescale(img: np.ndarray, scale=1.) -> np.ndarray:
    if scale != 1.:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return img


def get_fg_crop_region(fg_dims: Dimensions, bg_dims: Dimensions, fg_location: Tuple[int, int]) -> Region:
    fg_img_h, fg_img_w = fg_dims.height, fg_dims.width
    bg_img_h, bg_img_w = bg_dims.height, bg_dims.width
    fg_x, fg_y = fg_location

    fg_crop_top = max(-fg_y, 0)
    fg_crop_left = max(-fg_x, 0)
    fg_crop_bottom = min(-fg_y + bg_img_h, fg_img_h)
    fg_crop_right = min(-fg_x + bg_img_w, fg_img_w)

    fg_crop_region = Region(fg_crop_top, fg_crop_left, fg_crop_bottom, fg_crop_right)
    return fg_crop_region


def get_bg_crop_region(fg_dims: np.ndarray, bg_dims: np.ndarray, fg_location: Tuple[int, int]) -> Region:
    fg_img_h, fg_img_w = fg_dims.height, fg_dims.width
    bg_img_h, bg_img_w = bg_dims.height, bg_dims.width
    fg_x, fg_y = fg_location

    bg_crop_top = max(fg_y, 0)
    bg_crop_left = max(fg_x, 0)
    bg_crop_bottom = min(fg_y + fg_img_h, bg_img_h)
    bg_crop_right = min(fg_x + fg_img_w , bg_img_w)

    bg_crop_region = Region(bg_crop_top, bg_crop_left, bg_crop_bottom, bg_crop_right)
    return bg_crop_region


def crop(img: np.ndarray, region: Region) -> np.ndarray:
    return img[region.top:region.bottom, region.left:region.right, :]


def overlay(fg_img: np.ndarray, bg_img: np.ndarray, fg_weight: np.ndarray) -> np.ndarray:
    overlayed = fg_weight * fg_img + (1. - fg_weight) * bg_img
    overlayed = overlayed.astype(np.uint8)
    return overlayed


def patch_overlay(fg_img: np.ndarray, bg_img: np.ndarray, fg_weight: np.ndarray, location: Tuple[int, int]) -> np.ndarray:
    fg_dims = get_dims(fg_img)
    bg_dims = get_dims(bg_img)
    fg_crop_region = get_fg_crop_region(fg_dims, bg_dims, location)
    bg_crop_region = get_bg_crop_region(fg_dims, bg_dims, location)
    
    cropped_fg_img = crop(fg_img, fg_crop_region)
    cropped_fg_weight = crop(fg_weight, fg_crop_region)
    cropped_bg_img = crop(bg_img, bg_crop_region)
    patch = overlay(cropped_fg_img, cropped_bg_img, cropped_fg_weight)
    patch = patch.astype(np.uint8)

    img = bg_img.copy()
    img[bg_crop_region.top:bg_crop_region.bottom, bg_crop_region.left:bg_crop_region.right, :] = patch

    return img


class RandomTransform(object):
    def __init__(self, img_dims: Dimensions, transform_pct=.15):
        img_h, img_w = img_dims.height, img_dims.width
        points = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])
        transformed_points = np.float32([
            [random.uniform(0, img_w * transform_pct), random.uniform(0, img_h * transform_pct)],
            [random.uniform(img_w * (1 - transform_pct), img_w), random.uniform(0, img_h * transform_pct)],
            [random.uniform(img_w * (1 - transform_pct), img_w), random.uniform(img_h * (1 - transform_pct), img_h)],
            [random.uniform(0, img_w * transform_pct), random.uniform(img_h * (1 - transform_pct), img_h)],
        ])
        self.img_h = img_h
        self.img_w = img_w
        self.matrix = cv2.getPerspectiveTransform(points, transformed_points)


    def apply(self, img: np.ndarray, border_value: List[int]) -> np.ndarray:
        transformed = cv2.warpPerspective(img, self.matrix, (self.img_w, self.img_h), dst=np.zeros_like(img), borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
        transformed = np.reshape(transformed, img.shape)
        return transformed


    def apply_points(self, points: np.ndarray) -> np.ndarray:
        new_points = np.asarray([points], dtype=np.float32)
        new_points = cv2.perspectiveTransform(new_points, self.matrix)
        return new_points[0]


class RandomGlare(object):
    def __init__(self):
        image_paths = [
            os.path.join(Base_Path, "glare_images/kisspng-line-symmetry-angle-point-pattern-5a680f613edda1.6763382615167691212575.png"),
            os.path.join(Base_Path, "glare_images/star-light-png-transparent.png"),
            os.path.join(Base_Path, "glare_images/tv-glare-png-2-transparent.png"),
        ]
        self._glare_images = [self._read_image(img_path) for img_path in image_paths]


    def _read_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_channels = img.shape[2]
        assert img_channels == 4
        return img


    def _get_random_glare_location(self, glare_img, bg_img):
        glare_img_h, glare_img_w = glare_img.shape[:2]
        bg_img_h, bg_img_w = bg_img.shape[:2]

        glare_x = random.randint(0, bg_img_w) - glare_img_w // 2
        glare_y = random.randint(0, bg_img_h) - glare_img_h // 2

        return glare_x, glare_y


    def apply(self, img: np.ndarray) -> np.ndarray:
        glare_img = self._glare_images[random.randint(0, len(self._glare_images) - 1)]
        glare_img = rescale(glare_img, 1. + random.random())
        glare_dims = get_dims(glare_img)
        img_dims = get_dims(img)

        weight = glare_img[:, :, [3]] / 255.
        glare_img = glare_img[:, :, :3]

        glare_location = self._get_random_glare_location(glare_img, img)

        glare_crop_region = get_fg_crop_region(glare_dims, img_dims, glare_location)
        image_crop_region = get_bg_crop_region(glare_dims, img_dims, glare_location)

        cropped_weight = crop(weight, glare_crop_region)

        patch = overlay(crop(glare_img, glare_crop_region), crop(img, image_crop_region), cropped_weight)
        patch = patch.astype(np.uint8)

        img = img.copy()
        img[image_crop_region.top:image_crop_region.bottom, image_crop_region.left:image_crop_region.right, :] = patch

        return img


def get_random_brightness(range=100) -> int:
    value = random.randrange(-range, range + 1)
    return value


def get_random_scale(scale_min=0.5, scale_max=1.5) -> float:
    if scale_min <= 0:
        raise ValueError("scale_min must be bigger than 0")
    if scale_max <= 0:
        raise ValueError("scale_max must be bigger than 0")
    if scale_max <= scale_min:
        raise ValueError("scale_max must be bigger than scale_min")
    scale = random.uniform(scale_min, scale_max)
    return scale


def get_random_placement(fg_dims: Dimensions, bg_dims: Dimensions) -> Tuple[int, int]:
    if fg_dims.height > bg_dims.height or fg_dims.width > bg_dims.width:
        raise Exception("background is smaller than foreground")
    x_range = bg_dims.width - fg_dims.width
    y_range = bg_dims.height - fg_dims.height
    x = random.randint(0, x_range)
    y = random.randint(0, y_range)
    return x, y


class RandomPlacement(object):
    def __init__(self, bg_shape: Tuple[int, int], fg_shape: Tuple[int, int]):
        bg_h, bg_w = bg_shape
        fg_h, fh_w = fg_shape
        if fg_h > bg_h or fg_w > bg_w:
            raise Exception("background is smaller than foreground")

        w_range = bg_w - fg_w
        h_range = bg_h - fg_h
        self.x_translation = random.randint(0, w_range)
        self.y_translation = random.randint(0, h_range)


    def apply(self,   image: np.ndarray) -> np.ndarray:
        pass


    def apply_points(self, points: np.ndarray) -> np.ndarray:
        pass


class RandomDistortion(object):
    def __init__(self, img_shape: Tuple[int, int]):
        img_h, img_w = img_shape
        
        k1 = 1.0e-4
        k2 = .0
        p1 = .0
        p2 = .0
        dist_coeff = np.asarray([k1, k2, p1, p2], dtype=np.float32)

        # camera information
        cam = np.eye(3, dtype=np.float32)
        cam[0, 2] = img_w / 2.0
        cam[1, 2] = img_h / 2.0
        cam[0, 0] = 10.
        cam[1, 1] = 10.

        mapx, mapy = cv2.initUndistortRectifyMap(cam, dist_coeff, None, None, (img_w, img_h), 5)

        self.k = cam
        self.d = dist_coeff
        self.mapx = mapx
        self.mapy = mapy


    def apply(self, img: np.ndarray, border_value: List[int]) -> np.ndarray:
        distorted = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR, dst=np.zeros_like(img), borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
        distorted = np.reshape(distorted, img.shape)
        return distorted


    def apply_points(self, points: np.ndarray) -> np.ndarray:
        new_points = np.expand_dims(points, 0).astype(np.float32)
        new_points = cv2.undistortPoints(new_points, self.k, self.d, np.eye(3), self.k)
        return np.squeeze(new_points)


def get_points(img_h: int, img_w: int) -> np.ndarray:
    return np.asarray([
        (0, 0),
        (int(img_w * .25), 0),
        (int(img_w * .5), 0),
        (int(img_w * .75), 0),
        (img_w, 0),
        (img_w, int(img_h * .25)),
        (img_w, int(img_h * .5)),
        (img_w, int(img_h * .75)),
        (img_w, img_h),
        (int(img_w * .75), img_h),
        (int(img_w * .5), img_h),
        (int(img_w * .25), img_h),
        (0, img_h),
        (0, int(img_h * .75)),
        (0, int(img_h * .5)),
        (0, int(img_h * .25)),
    ])


def transform_image(image: np.ndarray, options: OptionsA) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # scale = get_random_scale()
    # image = rescale(image, scale)

    img_h, img_w = image.shape[:2]
    img_dims = get_dims(image)
    
    # cv2.imshow("rescaled", image)
    
    weight_mask = np.ones([img_h, img_w, 1], dtype=np.float32)
    points = get_points(img_h, img_w)
    
    # background_image = image

    if options.random_brightness:
        brightness = get_random_brightness(30)
        image = change_brightness(image, brightness)
        # cv2.imshow("brightness", image)

    if options.random_glare:
        glare = RandomGlare()
        image = glare.apply(image)
        # cv2.imshow("glare", image)

    if options.random_transform:
        transform = RandomTransform(img_dims)
        image = transform.apply(image, (255, 255, 255))
        weight_mask = transform.apply(weight_mask, (0,))
        points = transform.apply_points(points)
        # cv2.imshow("transformed", image)

    if options.random_distortion:
        distortion = RandomDistortion(img_dims)
        image = distortion.apply(image, (255, 255, 255))
        weight_mask = distortion.apply(weight_mask, (0,))
        points = distortion.apply_points(points)
        # cv2.imshow("distorted", image)

    # image = overlay(image, background_image, weight_mask)
    # image = blur(image, kernel_size=3)
    
    # cv2.imshow("overlayed", image)

    return image, weight_mask, points


def get_dims(img: np.ndarray) -> Dimensions:
    img_h, img_w = img.shape[:2]
    return Dimensions(img_h, img_w)


def generate(bg_img: np.ndarray, fg_img: np.ndarray, options: OptionsA) -> np.ndarray:
    fg_dims = get_dims(fg_img)
    bg_dims = get_dims(bg_img)

    # rescale foreground to fit
    h_scale = 1.
    if fg_dims.height > bg_dims.height:
        h_scale = 0.8 * bg_dims.height / fg_dims.height
    w_scale = 1.
    if fg_dims.width > bg_dims.width:
        w_scale = 0.8 * bg_dims.width / fg_dims.width
    scale = min(h_scale, w_scale)
    scale = get_random_scale(scale_min=(scale / 2.), scale_max=scale)
    
    fg_img = rescale(fg_img, scale)
    fg_img, fg_weight_masks, fg_points = transform_image(fg_img, options)

    random_location = get_random_placement(fg_dims, bg_dims)
    img = patch_overlay(fg_img, bg_img, fg_weight_masks, random_location)
    fg_points += np.asarray(random_location)

    return img, fg_points


if __name__ == "__main__":
    while True:
        img = cv2.imread(os.path.join(Base_Path, "Mykad/cropped/Android - Huawei/4.jpg"))

        fg_img = rescale(img)

        new_img, points = generate(img, fg_img, OptionsA())

        for point in points:
            cv2.circle(new_img, tuple(point), 3, (0, 0, 255), -1)
        
        cv2.imshow("New", new_img)
        
        k = cv2.waitKey(0)
        if k == 27:
            break

    cv2.destroyAllWindows()
