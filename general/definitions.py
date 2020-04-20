from collections import namedtuple


Dimensions = namedtuple("Shape", ("height", "width"))
Point2D = namedtuple("Point2D", ("x", "y"))
Region = namedtuple("Region", ("top", "left", "bottom", "right"))
