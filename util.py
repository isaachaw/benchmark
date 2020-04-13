from __future__ import division, print_function
from functools import reduce
import math
import operator
from typing import Tuple

from shapely.geometry import Polygon


Point2D = Tuple[float, float]
Point2Ds = Tuple[Point2D]

def calculate_intersection(polygon1: Polygon, polygon2: Polygon) -> Tuple[float, Point2Ds]:
    intersection = polygon1.intersection(polygon2)
    area = intersection.area
    points = tuple(intersection.boundary.coords)
    return (area, points)

def calculate_union(polygon1: Polygon, polygon2: Polygon) -> Tuple[float, Point2Ds]:
    union = polygon1.union(polygon2)
    area = union.area
    points = tuple(union.boundary.coords)
    return (area, points)

def calculate_iou(polygon1: Polygon, polygon2: Polygon):
    i_area, i_points = calculate_intersection(polygon1, polygon2)
    u_area, u_points = calculate_union(polygon1, polygon2)
    iou = i_area / u_area

def sort_points_clockwise(points: Point2Ds) -> Point2Ds:
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points), [len(points)] * 2))
    sorted_points = sorted(points, key=lambda point: math.atan2(*tuple(map(operator.sub, point, center))[::-1]))
    return sorted_points

if __name__ == "__main__":
    pts = [[2,3], [5,2],[4,1],[3.5,1],[1,2],[2,1],[3,1],[3,3],[4,3]]
    sort_points_clockwise(pts)
    