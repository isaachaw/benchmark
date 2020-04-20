import os
import random
from typing import List, Tuple

import cv2
import numpy as np

from general import GenerateOptions, Dimensions, Region, Point2D, get_all_file_paths


IMG_FILE_EXTENSIONS = (".jpg", ".png", ".bmp")


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


def blur(img: np.ndarray, kernel_size=3) -> np.ndarray:
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 1)
    return img


def rescale(img: np.ndarray, scale=1.) -> np.ndarray:
    if scale != 1.:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return img


def get_fg_crop_region(fg_dims: Dimensions, bg_dims: Dimensions, fg_loc: Point2D) -> Region:
    fg_h, fg_w = fg_dims
    bg_h, bg_w = bg_dims
    fg_x, fg_y = fg_loc

    fg_crop_top = max(-fg_y, 0)
    fg_crop_left = max(-fg_x, 0)
    fg_crop_bottom = min(-fg_y + bg_h, fg_h)
    fg_crop_right = min(-fg_x + bg_w, fg_w)

    fg_crop_region = Region(fg_crop_top, fg_crop_left, fg_crop_bottom, fg_crop_right)
    return fg_crop_region


def get_bg_crop_region(fg_dims: np.ndarray, bg_dims: np.ndarray, fg_loc: Point2D) -> Region:
    fg_h, fg_w = fg_dims
    bg_h, bg_w = bg_dims
    fg_x, fg_y = fg_loc

    bg_crop_top = max(fg_y, 0)
    bg_crop_left = max(fg_x, 0)
    bg_crop_bottom = min(fg_y + fg_h, bg_h)
    bg_crop_right = min(fg_x + fg_w , bg_w)

    bg_crop_reg = Region(bg_crop_top, bg_crop_left, bg_crop_bottom, bg_crop_right)
    return bg_crop_reg


def crop(img: np.ndarray, reg: Region) -> np.ndarray:
    return img[reg.top:reg.bottom, reg.left:reg.right, :]


def overlay(fg_img: np.ndarray, bg_img: np.ndarray, fg_weight: np.ndarray) -> np.ndarray:
    overlayed = fg_weight * fg_img + (1. - fg_weight) * bg_img
    overlayed = overlayed.astype(np.uint8)
    return overlayed


def patch_overlay(fg_img: np.ndarray, bg_img: np.ndarray, fg_weight: np.ndarray, loc: Point2D) -> np.ndarray:
    fg_dims = get_dims(fg_img)
    bg_dims = get_dims(bg_img)
    fg_crop_reg = get_fg_crop_region(fg_dims, bg_dims, loc)
    bg_crop_reg = get_bg_crop_region(fg_dims, bg_dims, loc)
    
    cropped_fg_img = crop(fg_img, fg_crop_reg)
    cropped_fg_weight = crop(fg_weight, fg_crop_reg)
    cropped_bg_img = crop(bg_img, bg_crop_reg)
    patch = overlay(cropped_fg_img, cropped_bg_img, cropped_fg_weight)
    patch = patch.astype(np.uint8)

    img = bg_img.copy()
    img[bg_crop_reg.top:bg_crop_reg.bottom, bg_crop_reg.left:bg_crop_reg.right, :] = patch

    return img


def get_dims(img: np.ndarray) -> Dimensions:
    img_h, img_w = img.shape[:2]
    return Dimensions(img_h, img_w)


class RandomTransform(object):
    def __init__(self, img_dims: Dimensions, transform_pct=.15):
        img_h, img_w = img_dims
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
    def __init__(self, options: GenerateOptions):
        glare_img_path_base = options.glare_img_path
        if not glare_img_path_base:
            raise Exception("glare_img_path is not defined")
        img_paths = [file_path for file_path in get_all_file_paths(
            glare_img_path_base) if os.path.splitext(file_path)[1].lower() in IMG_FILE_EXTENSIONS]
        if not len(img_paths):
            raise Exception("glare_img_path contains no images")
        self._glare_images = [self._read_image(
            img_path) for img_path in img_paths]


    def _read_image(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_channels = img.shape[2]
        assert img_channels == 4
        return img


    def _get_random_glare_location(self, glare_img: np.ndarray, bg_img: np.ndarray) -> Point2D:
        glare_h, glare_w = glare_img.shape[:2]
        bg_h, bg_w = bg_img.shape[:2]

        glare_x = random.randint(0, bg_w) - glare_w // 2
        glare_y = random.randint(0, bg_h) - glare_h // 2

        return Point2D(glare_x, glare_y)


    def apply(self, img: np.ndarray) -> np.ndarray:
        glare_img = self._glare_images[random.randint(0, len(self._glare_images) - 1)]
        glare_img = rescale(glare_img, 1. + random.random())

        glare_weight = glare_img[:, :, [3]] / 255.
        glare_img = glare_img[:, :, :3]
        glare_loc = self._get_random_glare_location(glare_img, img)

        img = patch_overlay(glare_img, img, glare_weight, glare_loc)
        return img


def get_random_brightness(range=0) -> int:
    value = random.randint(-range, range)
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


def get_random_placement(fg_dims: Dimensions, bg_dims: Dimensions) -> Point2D:
    if fg_dims.height > bg_dims.height or fg_dims.width > bg_dims.width:
        raise Exception("background is smaller than foreground")
    x = random.randint(0, bg_dims.width - fg_dims.width)
    y = random.randint(0, bg_dims.height - fg_dims.height)
    return Point2D(x, y)


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


def get_points(img_dims: Dimensions, options: GenerateOptions) -> np.ndarray:
    img_h, img_w = img_dims

    if options.random_distortion:
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
    else:
        return np.asarray([
            (0, 0),
            (img_w, 0),
            (img_w, img_h),
            (0, img_h),
        ])


def create_mask(img_dims: Dimensions) -> np.ndarray:
    img_h, img_w = img_dims
    weight = np.ones([img_h, img_w, 1], dtype=np.float32)

    offset = int(img_h * .05)
    
    weight[0:offset, 0:offset, :] = 0.
    weight[0:offset, img_w - offset:img_w, :] = 0.
    weight[img_h - offset:img_h, img_w - offset:img_w, :] = 0.
    weight[img_h - offset:img_h, 0:offset, :] = 0.

    fill_size = tuple([offset] * 2)

    cv2.ellipse(weight, (offset, offset), fill_size, 180., 0., 90, 1., -1)
    cv2.ellipse(weight, (img_w - offset, offset), fill_size, 270., 0., 90, 1., -1)
    cv2.ellipse(weight, (img_w - offset, img_h - offset), fill_size, 0., 0., 90, 1., -1)
    cv2.ellipse(weight, (offset, img_h - offset), fill_size, 90., 0., 90, 1., -1)
    
    return weight


def expand_edge(img: np.ndarray, size=0) -> np.ndarray:
    if size == 0:
        return img.copy()
    img_h, img_w, img_c = img.shape[:3]
    zeros = np.zeros((img_h, size, img_c), dtype=img.dtype)
    img = np.concatenate((zeros, img, zeros), axis=1)
    zeros = np.zeros((size, img_w + 2 * size, img_c), dtype=img.dtype)
    img = np.concatenate((zeros, img, zeros), axis=0)
    return img


def create_blur_edge_mask(img_dims: Dimensions, edge_size=0) -> np.ndarray:
    img_h, img_w = img_dims
    weight = np.zeros([img_h, img_w, 1], dtype=np.float32)
    weight[0:edge_size, :, :] = 1.
    weight[-edge_size:, :, :] = 1.
    weight[:, 0:edge_size, :] = 1.
    weight[:, -edge_size:, :] = 1.
    return weight


def transform_image(img: np.ndarray, options: GenerateOptions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    img_dims = get_dims(img)

    # trim masks and points
    trim_size = int(img_dims.width * 0.01)
    trim_dims = Dimensions(img_dims.height - trim_size * 2, img_dims.width - trim_size * 2)
    weight_mask = expand_edge(create_mask(trim_dims), trim_size)
    blur_mask = create_blur_edge_mask(img_dims, edge_size=trim_size * 2)
    points = get_points(trim_dims, options) + (trim_size, trim_size)

    if options.random_brightness:
        brightness = get_random_brightness(30)
        img = change_brightness(img, brightness)
        # cv2.imshow("brightness", image)

    if options.random_glare:
        glare = RandomGlare(options)
        img = glare.apply(img)
        # cv2.imshow("glare", image)

    if options.random_transform:
        transform = RandomTransform(img_dims)
        img = transform.apply(img, (255, 255, 255))
        weight_mask = transform.apply(weight_mask, (0,))
        blur_mask = transform.apply(blur_mask, (0, 0))
        points = transform.apply_points(points)
        # cv2.imshow("transformed", image)

    if options.random_distortion:
        distortion = RandomDistortion(img_dims)
        img = distortion.apply(img, (255, 255, 255))
        weight_mask = distortion.apply(weight_mask, (0,))
        blur_mask = distortion.apply(blur_mask, (0,))
        points = distortion.apply_points(points)
        # cv2.imshow("distorted", image)

    return img, weight_mask, blur_mask, points


def generate(bg_img: np.ndarray, fg_img: np.ndarray, options: GenerateOptions) -> np.ndarray:
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
    fg_img, fg_weight_mask, fg_blur_mask, fg_points = transform_image(fg_img, options)
    fg_dims = get_dims(fg_img)

    loc = get_random_placement(fg_dims, bg_dims)
    img = patch_overlay(fg_img, bg_img, fg_weight_mask, loc)
    fg_points += np.asarray(loc)

    # blur the edge of foreground
    patch = img[loc.y:loc.y + fg_dims.height, loc.x:loc.x + fg_dims.width, :]
    blur_patch = blur(patch, 3)
    blur_patch = np.where(fg_blur_mask > 0, blur_patch, patch)
    img[loc.y:loc.y + fg_dims.height, loc.x:loc.x + fg_dims.width, :] = blur_patch

    return img, fg_points
