# Python g-code library
import math

from pygcode import *
# Number library
import numpy as np
# Type hints
from typing import Self


class Path:
    pass


class LinearTransform:
    """
    An abstract linear transformation
    """

    def __init__(self, name: str):
        """
        Create a linear transformation
        :param name: Name of transformation
        """
        self.name = name

    def transform(self, path: Path) -> Path:
        """
        Transform a set of points
        :param path: Set of points to transform
        :return: Set of transformed points
        """
        p = Path([])
        q = self._transform_matrix()
        p.path = path.path @ q
        return p

    def _transform_matrix(self) -> np.ndarray:
        raise NotImplementedError()


class Point:
    """An arbitrary 4D vector"""

    def __init__(self, vec: np.ndarray | list[float]):
        """
        Create a point
        :param vec: Vector coordinate
        """
        # Convert to 4D
        if isinstance(vec, list):
            vec = np.array(vec)

        self.vec = vec if len(vec) == 4 else np.append(vec, [1]) if len(vec) == 3 else np.append(vec, [0, 1])

    def x(self) -> float:
        return self.vec[0]

    def y(self) -> float:
        return self.vec[1]

    def z(self) -> float:
        return self.vec[2]

    def xyz(self) -> tuple[float, float, float]:
        return self.x(), self.y(), self.z()

    def unit(self) -> Self:
        """
        Get the unit vector
        :return: Unit vector
        """
        return self / abs(self)

    def __abs__(self) -> float:
        return np.linalg.norm(self.vec)

    def __truediv__(self, other: float) -> Self:
        return Point(self.vec / other)

    def __mul__(self, other: float) -> Self:
        return Point(self.vec * other)

    def __add__(self, other: Self) -> Self:
        return Point(self.vec + other.vec)

    def __sub__(self, other: Self) -> Self:
        return Point(self.vec - other.vec)

    def mag(self) -> float:
        """
        Get the magnitude of the vector
        :return: vector magnitude
        """
        return abs(self)

    def __eq__(self, value: Self) -> bool:
        return (self.vec == value.vec).all()

    def move_to(self) -> GCodeLinearMove:
        """
        Return the g-code for a slow linear move
        :return: GCodeLinearMove
        """
        return GCodeLinearMove(X=self.x(), Y=self.y(), Z=self.z())

    def fast_to(self) -> GCodeRapidMove:
        """
        Return the g-code for a fast linear move
        :return: GCodeLinearMove
        """
        return GCodeRapidMove(X=self.x(), Y=self.y(), Z=self.z())


class Scale(LinearTransform):
    """A scaling transformation"""

    def __init__(self, point: Point):
        self.scale = point
        super().__init__(f"Scale {point.xyz()}")

    def _transform_matrix(self) -> np.ndarray:
        return np.array([[self.scale.x(), 0, 0, 0],
                         [0, self.scale.y(), 0, 0],
                         [0, 0, self.scale.z(), 0],
                         [0, 0, 0, 1]])


class Translate(LinearTransform):
    """A scaling transformation"""

    def __init__(self, point: Point):
        self.translate = point
        super().__init__(f"Translate {point.xyz()}")

    def _transform_matrix(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [self.translate.x(), self.translate.y(), self.translate.z(), 1]])


class RotateX(LinearTransform):
    """Counterclockwise rotation along the x-axis"""

    def __init__(self, angle: float):
        self.angle = math.radians(angle)
        super().__init__(f"Rotate X {angle} degrees")

    def _transform_matrix(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0],
                         [0, math.cos(self.angle), math.sin(self.angle), 0],
                         [0, -math.sin(self.angle), math.cos(self.angle), 0],
                         [0, 0, 0, 1]])


class RotateY(LinearTransform):
    """Counterclockwise rotation along the y-axis"""

    def __init__(self, angle: float):
        self.angle = math.radians(angle)
        super().__init__(f"Rotate Y {angle} degrees")

    def _transform_matrix(self) -> np.ndarray:
        return np.array([[math.cos(self.angle), 0, -math.sin(self.angle), 0],
                         [0, 1, 0, 0],
                         [math.sin(self.angle), 0, math.cos(self.angle), 0],
                         [0, 0, 0, 1]])


class RotateZ(LinearTransform):
    """Counterclockwise rotation along the y-axis"""

    def __init__(self, angle: float):
        self.angle = math.radians(angle)
        super().__init__(f"Rotate Z {angle} degrees")

    def _transform_matrix(self) -> np.ndarray:
        return np.array([[math.cos(self.angle), -math.sin(self.angle), 0, 0],
                         [math.sin(self.angle), math.cos(self.angle), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])


class RotateAxis(LinearTransform):
    """Counterclockwise rotation along a specific axis"""

    def __init__(self, axis: Point, angle: float):
        """
        Create a rotation transformation along an axis
        :param axis: Unit axis
        :param angle:
        """
        self.axis = axis
        self.angle = math.radians(angle)
        super().__init__(f"Rotate along {axis}, {angle} degrees")

    def _transform_matrix(self) -> np.ndarray:
        u = self.axis
        a = self.angle
        return np.array(
            [[u.x() ** 2 * (1 - math.cos(a)) + math.cos(a), u.x() * u.y() * (1 - math.cos(a)) - u.z() * math.sin(a),
              u.x() * u.z() * (1 - math.cos(a)) + u.y() * math.sin(a), 0],
             [u.x() * u.y() * (1 - math.cos(a)) + u.z() * math.sin(a), u.y() ** 2 * (1 - math.cos(a)) + math.cos(a),
              u.y() * u.z() * (1 - math.cos(a)) - u.x() * math.sin(a), 0],
             [u.x() * u.z() * (1 - math.cos(a)) - u.y() * math.sin(a),
              u.y() * u.z() * (1 - math.cos(a)) + u.x() * math.sin(a),
              u.z() ** 2 * (1 - math.cos(a)) + math.cos(a), 0],
             [0, 0, 0, 1]
             ])


class Path:
    """
    Vector path
    """

    def __init__(self, points: list[Point] | np.ndarray):
        self.path = np.array([point.vec for point in points]) if isinstance(points, list) else points

    def gcode(self) -> list[GCode]:
        return [Point(arr).move_to() for arr in self.path]

    def __eq__(self, value: Self) -> bool:
        return (self.path == value.path).all()


def serp(dims: tuple[int, int], stride: float, offset: Point, rotate: bool = False) -> Path:
    """
    Create a serpentine path with a given frequency and offset
    :param dims: Dimensions of the serpentine
    :param stride: Stride between lines
    :param offset: Position offset
    :return: Serpentine path
    """
    path = []  # 1x1 serpentine path

    # Scale to correct size
    width, height = dims

    # Create 1x1 path with given stride
    alternate = False  # Alternate side flag
    x = 0
    ds = stride / width
    while x < 1:
        if alternate:
            path.append(Point([x, 0]))
            path.append(Point([x, 1]))
        else:
            path.append(Point([x, 1]))
            path.append(Point([x, 0]))
        x += ds
        alternate = not alternate

    path = Path(path)  # Convert to path object for transformation

    # Rotate path and reflect along y-axis
    path = path if not rotate else Scale(Point([1, -1, 1])).transform(RotateZ(90).transform(path))

    return Translate(offset).transform(Scale(Point([width, height, 1])).transform(path))


# Preamble for the g-code script
preamble = [
    GCodeUseMillimeters(),
    GCodeAbsoluteDistanceMode(),
]

gcodes = preamble  + serp((36, 14), 2, Point([0, 0])).gcode() \
        + serp((35, 14), 2, Point([1, 0])).gcode() \
        + serp((36, 14), 5, Point([0, 0]), rotate=True).gcode() \
        + serp((36, 13), 5, Point([0, 1]),rotate=True).gcode()
# Print G-Codes
print('\n'.join(f'N{10 + i * 5} {str(g)}' for (i, g) in enumerate(gcodes)))
