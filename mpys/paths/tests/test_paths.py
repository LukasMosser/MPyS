import unittest
import mpys


class TestPaths(unittest.TestCase):
    def setUp(self):
        pass

    def test_random_path_2d(self):
        dimensions = (10, 11)
        path_2d = mpys.random_path(dimensions)

        values_x, values_y = [], []
        for x, y in path_2d:
            values_x.append(x)
            values_y.append(y)
        x_set = list(set(values_x))
        y_set = list(set(values_y))

        self.assertEquals(dimensions[0]*dimensions[1], len(path_2d))

        self.assertListEqual(range(dimensions[0]), x_set)
        self.assertListEqual(range(dimensions[1]), y_set)

    def test_random_path_3d(self):
        dimensions = (10, 11, 12)
        path_3d = mpys.random_path(dimensions)

        values_x, values_y, values_z = [], [], []
        for x, y, z in path_3d:
            values_x.append(x)
            values_y.append(y)
            values_z.append(z)

        x_set = list(set(values_x))
        y_set = list(set(values_y))
        z_set = list(set(values_z))

        self.assertEquals(dimensions[0] * dimensions[1] * dimensions[2], len(path_3d))

        self.assertListEqual(range(dimensions[0]), x_set)
        self.assertListEqual(range(dimensions[1]), y_set)
        self.assertListEqual(range(dimensions[2]), z_set)
