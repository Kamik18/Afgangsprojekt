import numpy as np
from fractions import Fraction
from scipy.odr import *
from scipy.odr.odrpack import ODR, Model, RealData


class featureDetection:
    def __init__(self):
        # vaqriables
        self.EPSILON = 1000  # The tolerance
        self.DELTA = 1000
        self.SNUM = 6
        self.PMIN = 1
        self.GMAX = 200
        self.SEED_SEGMENTS = []
        self.LINE_SEGMENTS = []
        self.LASERPOINTS = []
        self.LINE_PARAMS = None
        self.NP = len(self.LASERPOINTS) - 1
        self.LMIN = 10  # minimum length of a line segment
        self.LR = 0  # real length of a line segment
        self.PR = 0  # the number of laser points contained in the line segment

    # Euclidian distance from point 1 to point 2
    def dist_point2point(self, point1, point2):
        Px = (point1[0] - point2[0]) ** 2
        Py = (point1[1] - point2[1]) ** 2
        return np.sqrt(Px + Py)

    # Distance point to line written in the general form
    def dist_point2line(self, params, point):
        A, B, C = params
        distance = abs(A * point[0] + B * point[1] +
                       C) / np.sqrt(A**2 + B**2)
        return distance

    # Extract two points from a line equation under the slope intercepts form
    def line_2points(self, m, b):
        x = 5
        y = m * x + b
        x2 = 2000
        y2 = m * x2 + b
        return [(x, y), (x2, y2)]

    # General form to slope-intercept
    def lineForm_G2Si(self, A, B, C):
        m = -A / B
        b = -C / B
        return m, b

    # Slope-intercept to general form
    def lineForm_Si2G(self, m, b):
        A, B, C = -m, 1, -b
        if (A < 0):
            A, B, C = -A, -B, -C

        # Formel: (x - x1)/(x2 - x1) = (y - y1)/(y2 - y1)
        p0 = C  # 0 * m + b
        p1 = A + C  # 1 * m + b

        A = p1 - p0
        B = -1
        C = p0 * 1

        '''
        den_a = Fraction(A).limit_denominator(1000).as_integer_ratio()[1]
        den_c = Fraction(C).limit_denominator(1000).as_integer_ratio()[1]

        gcd = np.gcd(den_a, den_c)
        lcm = den_a * den_c / gcd

        A = A * lcm
        B = B * lcm
        C = C * lcm
        '''
        return A, B, C

    def line_intersect_general(self, params1, params2):
        a1, b1, c1 = params1
        a2, b2, c2 = params2

        denominator = (b1 * a2 - a1 * b2)
        if (denominator != 0):
            x = (c1 * b2 - b1 * c2) / denominator
            y = (a1 * c2 - a2 * c1) / denominator
            return x, y
        else:
            print("Div 0 Error")
            return np.inf, np.inf

    def points_2line(self, point1, point2):
        m, b = 0, 0
        if (point2[0] == point1[0]):
            pass
        else:
            m = (point2[1] - point1[1]) / (point2[0] - point1[0])
            b = point2[1] - m * point2[0]
        return m, b

    def projection_point2line(self, point, m, b):
        x, y = point
        m2 = -1 / m
        c2 = y - m2 * x
        intersection_x = - (b - c2) / (m - m2)
        intersection_y = m2 * intersection_x + c2
        return intersection_x, intersection_y

    def AD2pos(self, distance, angle, robot_position=(0, 0)):
        x = distance * np.cos(angle) + robot_position[0]
        y = -distance * np.sin(angle) + robot_position[1]
        return (int(x), int(y))

    def laser_points_set(self, data):
        self.LASERPOINTS = []
        if not data:
            pass
        else:
            for point in data:
                length = data[point]
                if length > 20:
                    coordinates = self.AD2pos(length, np.radians(point))
                    self.LASERPOINTS.append(coordinates)
        self.NP = len(self.LASERPOINTS) - 1

    # Define a function (quadratic in our case) to fit the data with.
    def linear_func(self, p, x):
        m, b = p
        return m * x + b

    def odr_fit(self, laser_points):
        x = np.array([i[0] for i in laser_points])
        y = np.array([i[1] for i in laser_points])

        # Create a model for fitting.
        linear_model = Model(self.linear_func)

        # Create a RealData object using our initiated data from above.
        data = RealData(x, y)

        # Set up ODR with the model and data.
        odr_model = ODR(data, linear_model, beta0=[0., 0.])

        # Run the regression.
        out = odr_model.run()
        m, b = out.beta
        return m, b

    def predictPoint(self, line_params, sensed_point, robotpos):
        m, b = self.points_2line(robotpos, sensed_point)
        params1 = self.lineForm_Si2G(m, b)
        predx, predy = self.line_intersect_general(params1, line_params)
        return predx, predy

    def seed_segment_detection(self, break_point_ind, robot_postion = (0, 0)):
        flag = True
        self.NP = max(0, self.NP)
        self.SEED_SEGMENTS = []
        for i in range(break_point_ind, (self.NP - self.PMIN)):
            predicted_points_to_draw = []
            j = i + self.SNUM
            m, c = self.odr_fit(self.LASERPOINTS[i:j])
            params = self.lineForm_Si2G(m, c)

            for k in range(i, j):
                predicted_point = self.predictPoint(
                    params, self.LASERPOINTS[k], robot_postion)
                if (np.isinf(predicted_point[0])):
                    if (k < (self.NP - 1)):
                        continue
                    else:
                        break
                predicted_points_to_draw.append(predicted_point)

                d1 = self.dist_point2point(
                    predicted_point, self.LASERPOINTS[k])
                if (d1 > self.DELTA):
                    flag = False
                    print("DELTA error")
                    break

                d2 = self.dist_point2line(params, predicted_point)
                if (d2 > self.EPSILON):
                    flag = False
                    print("EPSILON error")
                    break
            if flag:
                self.LINE_PARAMS = params
                return [self.LASERPOINTS[i:j], predicted_points_to_draw, (i, j)]
        return False

    def seed_segment_growing(self, indices, break_point):
        line_eq = self.LINE_PARAMS
        i, j = indices

        # Begining the final points in the line segments
        PB, PF = max(break_point, i - 1), min(j + 1, len(self.LASERPOINTS) - 1)

        while (self.dist_point2line(line_eq, self.LASERPOINTS[PF]) < self.EPSILON):
            if (PF > self.NP - 1):
                break
            else:
                m, b = self.odr_fit(self.LASERPOINTS[PB:PF])
                line_eq = self.lineForm_Si2G(m, b)

                POINT = self.LASERPOINTS[PF]

            PF = PF + 1
            NEXTPOINT = self.LASERPOINTS[PF]
            if (self.dist_point2point(POINT, NEXTPOINT) > self.GMAX):
                print("GMAX1 error")
                break

        PF = PF - 1

        while (self.dist_point2line(line_eq, self.LASERPOINTS[PB]) < self.EPSILON):
            if (PB < break_point):
                print("Break point-  error")
                break
            else:
                m, b = self.odr_fit(self.LASERPOINTS[PB:PF])
                line_eq = self.lineForm_Si2G(m, b)
                POINT = self.LASERPOINTS[PB]

            PB = PB - 1
            NEXTPOINT = self.LASERPOINTS[PB]
            if (self.dist_point2point(POINT, NEXTPOINT) > self.GMAX):
                print("GMAX2 error")
                break
        PB = PB + 1

        LR = self.dist_point2point(
            self.LASERPOINTS[PB], self.LASERPOINTS[PF])
        PR = len(self.LASERPOINTS[PB:PF])

        if ((LR >= self.LMIN) and (PR >= self.PMIN)):
            self.LINE_PARAMS = line_eq
            m, b = self.lineForm_G2Si(line_eq[0], line_eq[1], line_eq[2])
            self.two_points = self.line_2points(m, b)
            self.LINE_SEGMENTS.append(
                (self.LASERPOINTS[PB + 1], self.LASERPOINTS[PF - 1]))
            return [self.LASERPOINTS[PB:PF], self.two_points,
                    (self.LASERPOINTS[PB + 1], self.LASERPOINTS[PF - 1]), PF, line_eq, (m, b)]
        else:
            print("LMIN/PMIN error")
            return False
