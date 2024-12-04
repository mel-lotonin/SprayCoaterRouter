import unittest

from spraycoater import Point, Scale, Path


class ScaleTest(unittest.TestCase):

    def test_scale_square(self):
        origin = Path([Point([0, 0]), Point([0, 1]), Point([1, 1]), Point([1, 0])])
        scale = Scale(Point([2, 3]))
        scaled = Path([Point([0, 0]), Point([0, 3]), Point([2, 3]), Point([2, 0])])
        assert (scale.transform(origin.path) == scaled.path).all()

if __name__ == '__main__':
    unittest.main()
