import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def draw_iou(img, gt_points, p_points, i_points):
    fig, ax = plt.subplots()
    ax.fill([p[0] for p in gt_points], [p[1] for p in gt_points], edgecolor="b", fill=False)
    ax.fill([p[0] for p in p_points], [p[1] for p in p_points], edgecolor="g", fill=False)
    ax.fill([p[0] for p in i_points], [p[1] for p in i_points], edgecolor="r", fill=False)
    ax.imshow(img)
    plt.show()
