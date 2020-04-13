from collections import namedtuple
import os
from typing import List, Tuple
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np


Base_Path = "C:/workspace/consolsys/InnoLab/benchmark"
Region = namedtuple("Region", ("top", "left", "bottom", "right"))


def get_random_brightness(range=100) -> int:
    value = random.randrange(-range, range + 1)
    return value


def change_brightness(image: np.ndarray, value=0) -> np.ndarray:
    if value != 0:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value > 0:
            v = np.where(v < 255 - value, v + value, 255)
        else:
            v = np.where(v > -value, v + value, 0)
        v = v.astype(np.uint8)
        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image


def blur(image: np.ndarray, kernel_size=5) -> np.ndarray:
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)
    return image


def get_random_scale(scale_min=0.5, scale_max=1.5) -> float:
    if scale_min <= 0:
        raise ValueError("scale_min must be bigger than 0")
    if scale_max <= 0:
        raise ValueError("scale_max must be bigger than 0")
    if scale_max <= scale_min:
        raise ValueError("scale_max must be bigger than scale_min")
    scale = random.uniform(scale_min, scale_max)
    return scale


def rescale(image: np.ndarray, scale=1.) -> np.ndarray:
    if scale != 1.:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return image


class RandomTransform(object):
    def __init__(self, img_shape: Tuple[int, int], transform_pct=.15):
        img_h, img_w = img_shape
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


def get_fg_crop_region(fg_img: np.ndarray, bg_img: np.ndarray, fg_location: Tuple[int, int]) -> Region:
    fg_img_h, fg_img_w = fg_img.shape[:2]
    bg_img_h, bg_img_w = bg_img.shape[:2]
    fg_x, fg_y = fg_location

    fg_crop_top = max(-fg_y, 0)
    fg_crop_left = max(-fg_x, 0)
    fg_crop_bottom = min(-fg_y + bg_img_h, fg_img_h)
    fg_crop_right = min(-fg_x + bg_img_w, fg_img_w)

    fg_crop_region = Region(fg_crop_top, fg_crop_left, fg_crop_bottom, fg_crop_right)
    return fg_crop_region


def get_bg_crop_region(fg_img: np.ndarray, bg_img: np.ndarray, fg_location: Tuple[int, int]) -> Region:
    fg_img_h, fg_img_w = fg_img.shape[:2]
    bg_img_h, bg_img_w = bg_img.shape[:2]
    fg_x, fg_y = fg_location

    bg_crop_top = max(fg_y, 0)
    bg_crop_left = max(fg_x, 0)
    bg_crop_bottom = min(fg_y + fg_img_h, bg_img_h)
    bg_crop_right = min(fg_x + fg_img_w , bg_img_w)

    bg_crop_region = Region(bg_crop_top, bg_crop_left, bg_crop_bottom, bg_crop_right)
    return bg_crop_region


def crop(image: np.ndarray, region: Region) -> np.ndarray:
    return image[region.top:region.bottom, region.left:region.right, :]


class RandomGlare(object):

    def __init__(self):
        image_paths = [
            os.path.join(Base_Path, "glare_images/kisspng-line-symmetry-angle-point-pattern-5a680f613edda1.6763382615167691212575.png"),
            os.path.join(Base_Path, "glare_images/star-light-png-transparent.png"),
            os.path.join(Base_Path, "glare_images/tv-glare-png-2-transparent.png"),
        ]
        self._glare_images = [self._read_image(image_path) for image_path in image_paths]


    def _read_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image_channels = image.shape[2]
        assert image_channels == 4
        return image


    def _get_random_glare_location(self, glare_img, bg_img):
        glare_img_h, glare_img_w = glare_img.shape[:2]
        bg_img_h, bg_img_w = bg_img.shape[:2]

        glare_x = random.randint(0, bg_img_w) - glare_img_w // 2
        glare_y = random.randint(0, bg_img_h) - glare_img_h // 2

        return glare_x, glare_y


    def apply(self, image):
        glare_img = self._glare_images[random.randint(0, len(self._glare_images) - 1)]
        glare_img = rescale(glare_img, 1. + random.random())

        weight = glare_img[:, :, [3]] / 255.
        glare_img = glare_img[:, :, :3]

        glare_location = self._get_random_glare_location(glare_img, image)

        glare_crop_region = get_fg_crop_region(glare_img, image, glare_location)
        image_crop_region = get_bg_crop_region(glare_img, image, glare_location)

        cropped_weight = crop(weight, glare_crop_region)

        patch = overlay(crop(glare_img, glare_crop_region), crop(image, image_crop_region), cropped_weight)
        patch = patch.astype(np.uint8)

        image = image.copy()
        image[image_crop_region.top:image_crop_region.bottom, image_crop_region.left:image_crop_region.right, :] = patch

        return image


class RandomPlacement(object):
    def __init__(self, bg_shape: Tuple[int, int], fg_shape: Tuple[int, int]):
        bg_h, bg_w = bg_shape
        fg_h, fh_w = fg_shape
        if fg_h > bg_h or fg_w > bg_w:
            raise Exception("background is smaller than foreground")
        this.fg_h = fg_h
        this.fg_w = fg_w
        this.bg_h = bg_h
        this.bg_w = bg_w



    def apply(self, image: np.ndarray) -> np.ndarray:
        pass


    def apply_points(self, points: np.ndarray) -> np.ndarray:
        pass


class RandomDistortion(object):
    def __init__(self, img: np.ndarray):
        img_h, img_w = img.shape[:2]
        
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


def overlay(image1, image2, weight):
    overlayed = weight * image1 + (1. - weight) * image2
    overlayed = overlayed.astype(np.uint8)
    return overlayed


def get_region(img_h: int, img_w: int) -> np.ndarray:
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


def transform_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = get_random_scale()
    image = rescale(image, scale)

    img_h, img_w = image.shape[:2]
    
    # cv2.imshow("rescaled", image)
    
    weight_mask = np.ones([img_h, img_w, 1], dtype=np.float32)
    points = get_region(img_h, img_w)
    
    background_image = image

    brightness = get_random_brightness(30)
    image = change_brightness(image, brightness)
    
    # cv2.imshow("brightness", image)

    glare = RandomGlare()
    image = glare.apply(image)
    
    # cv2.imshow("glare", image)

    transform = RandomTransform((img_h, img_w))
    image = transform.apply(image, (255, 255, 255))
    weight_mask = transform.apply(weight_mask, (0,))
    points = transform.apply_points(points)
    
    # cv2.imshow("transformed", image)

    distortion = RandomDistortion(image)
    image = distortion.apply(image, (255, 255, 255))
    weight_mask = distortion.apply(weight_mask, (0,))
    points = distortion.apply_points(points)

    # cv2.imshow("distorted", image)

    image = overlay(image, background_image, weight_mask)
    image = blur(image, kernel_size=3)
    
    # cv2.imshow("overlayed", image)

    return image, weight_mask, points


if __name__ == "__main__":
    while True:
        image = cv2.imread(os.path.join(Base_Path, "Mykad/cropped/Android - Huawei/4.jpg"))
        img_h, img_w = image.shape[:2]
        image = cv2.resize(image, (img_w // 2, img_h // 2))
        cv2.imshow("original", image)

        transformed_image, weight_mask, points = transform_image(image)

        background_image = cv2.resize(image, weight_mask.shape[:2][::-1])

        image = overlay(transformed_image, background_image, weight_mask)
        
        # scale = get_random_scale()
        # image = rescale(image, scale)
        # cv2.imshow("rescaled", image)
        
        # background_image = image

        # brightness = get_random_brightness(30)
        # image = change_brightness(image, brightness)
        # cv2.imshow("brightness", image)

        # rg = RandomGlare()
        # image = rg.apply(image)
        # cv2.imshow("glare", image)
        
        # weight = np.ones([*image.shape[:2], 1], dtype=np.float32)

        # img_h, img_w = image.shape[:2]
        # points = get_region(img_h, img_w)
        
        # transform = RandomTransform((img_h, img_w))
        # image = transform.apply(image, (255, 255, 255))
        # weight = transform.apply(weight, (0,))
        # points = transform.apply_points(points)
        
        # cv2.imshow("transformed", image)

        # distortion = RandomDistortion(image)
        # image = distortion.apply(image, (255, 255, 255))
        # weight = distortion.apply(weight, (0,))
        # points = distortion.apply_points(points)

        # cv2.imshow("distorted", image)

        # image = overlay(image, background_image, weight)
        # image = blur(image, kernel_size=3)

        for point in points:
            cv2.circle(image, tuple(point), 3, (0, 0, 255), -1)
        
        cv2.imshow("overlayed", image)
        
        k = cv2.waitKey(0)
        if k == 27:
            break

    cv2.destroyAllWindows()
