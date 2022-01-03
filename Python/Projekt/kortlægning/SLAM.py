from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import cv2

# ICP parameters
EPS = 0.0001
MAX_ITER = 100

def icp_matching(corrected_points, feature_points):
    """
    Iterative Closest Point matching
    - input
    corrected_points: 2D points in the map
    new_points: 2D points in the new frame
    - output
    R: Rotation matrix
    T: Translation vector
    """
    # Convert the points
    temp_corrected_points = np.vstack(zip(*corrected_points))
    temp_feature_points = np.vstack(zip(*feature_points))


    H = None  # homogeneous transformation matrix

    dError = np.inf
    preError = np.inf
    count = 0

    while dError >= EPS:
        indexes, error = nearest_neighbor_association(
            temp_corrected_points, temp_feature_points)
        Rt, Tt = svd_motion_estimation(
            temp_corrected_points[:, indexes], temp_feature_points)
        # update current points
        temp_feature_points = (Rt @ temp_feature_points) + Tt[:, np.newaxis]

        dError = preError - error
        #print("Residual:", error)

        if dError < 0:  # prevent matrix H changing, exit loop
            break

        preError = error
        H = update_homogeneous_matrix(H, Rt, Tt)
        count += 1

        if (dError <= EPS) or (MAX_ITER <= count):
            break

    if count == 1:
        H = np.zeros((3, 3))
    
    R = np.array(H[0:-1, 0:-1])
    T = np.array(H[0:-1, -1])
    return R, T, error


def update_homogeneous_matrix(Hin, R, T):
    H = np.zeros((3, 3))
    H[0:2, 0:2] = R
    H[0:2, 2] = T
    H[2, 2] = 1.0

    if Hin is None:
        return H
    else:
        return Hin @ H

# Euclidian distance from point 1 to point 2
def dist_points(point1, point2):
    Px = (point1[0] - point2[0]) ** 2
    Py = (point1[1] - point2[1]) ** 2
    return np.sqrt(Px + Py)

def in_line(point1, point2, point3):
    dist = dist_points(point1, point3)
    if (0 == dist):
        dist = 0.01

    if ((point2[0] - point1[0]) != 0):
        a = (point2[1] - point1[1]) / (point2[0] - point1[0])
        b = point1[1] - a * point1[0]
        return abs((a * point3[0] + b) - point3[1]) / dist
    elif ((point3[0] - point1[0]) != 0):
        a = (point3[1] - point2[1]) / (point3[0] - point2[0])
        b = point2[1] - a * point2[0]
        return abs((a * point1[0] + b) - point1[1]) / dist
    else:
        return 0

def extract_feature_points(new_points):   
    # Find breakpoints
    feature_points = []
    dist = 25
    for item in range(0, len(new_points) - 2):
        # Validate the distance is closer than 25 mm
        if (((dist_points(new_points[item], new_points[item + 1]) < dist) and (dist_points(new_points[item + 1], new_points[item + 2]) < dist)) and
            ((in_line(new_points[item], new_points[item + 1], new_points[item + 2]) > 0.5) or (in_line(new_points[item], new_points[item + 1], new_points[item + 2]) < 0.01))):
            feature_points.append(new_points[item + 1])

    # Return the feature points
    return feature_points

def nearest_neighbor_association(map_points, new_points):
    # calc the sum of residual errors
    delta_points = []
    start_index = 1
    for i in range(0, len(new_points[0])):
        index = 0
        nearest = dist_points(
            (new_points[0][i], new_points[1][i]), (map_points[0][0], map_points[1][0]))

        for j in range(start_index, len(map_points[0]), 1):
            distance = dist_points(
                (new_points[0][i], new_points[1][i]), (map_points[0][j], map_points[1][j]))
            if (distance < nearest):
                nearest = distance
                index = j
            else:
                start_index = index
                break
        
        if (nearest == 0):
            nearest = 1

        weight = (1 / nearest)
        x = (map_points[0][index] - new_points[0][i]) * weight
        y = (map_points[1][index] - new_points[1][i]) * weight
        delta_points.append( (x, y) )

    error = sum(np.linalg.norm(delta_points, axis=0))

    # Calc index with nearest neighbor association
    d = np.linalg.norm(np.repeat(new_points, map_points.shape[1], axis=1)
                       - np.tile(map_points, (1, new_points.shape[1])), axis=0)
    indexes = np.argmin(d.reshape(new_points.shape[1], map_points.shape[1]), axis=1)

    return indexes, (error / len(delta_points))

def svd_motion_estimation(map_points, new_points):
    pm = np.mean(map_points, axis=1)
    cm = np.mean(new_points, axis=1)

    p_shift = map_points - pm[:, np.newaxis]
    c_shift = new_points - cm[:, np.newaxis]

    W = c_shift @ p_shift.T
    u, _, vh = np.linalg.svd(W)

    R = (u @ vh).T
    t = pm - (R @ cm)

    return R, t

def AD2pos(distance, angle, robot_position):
    x = distance * np.cos(angle - robot_position[2]) + robot_position[0]
    y = -distance * np.sin(angle - robot_position[2]) + robot_position[1]
    return (int(x), int(y))

def convert_data_set(data, robot_position=np.zeros(3)):
    points = []
    if not data:
        pass
    else:
        for point in data:
            length = data[point]
            # Validate the distance is greater than 20 cm
            #if 200 < length < 3000: # TODO: original
            if 200 < length < 2500:
                coordinates = AD2pos(length, np.radians(point), robot_position)
                points.append(coordinates)
    return points


def clear_path(map_points, map_poses):
    clearance_points = []
    discard_points = []
    threshold = 0.2

    END = len(map_points)
    for i in range(0, END):
        print("Iteration: ", (i + 1), " of ", END)
        nearest = np.inf

        for pos in map_poses:
            dist = dist_points(map_points[i], pos)
            if (dist < nearest):
                nearest = dist
            if (nearest < threshold):
                break
        if (nearest > threshold):
            clearance_points.append(map_points[i])
        else:
            discard_points.append(map_points[i])
    return clearance_points, discard_points

def clear_outliers(map_points):
    modified_points = []
    END = len(map_points)
    for i in range(0, END):
        print("Iteration: ", (i + 1), " of ", END)
        nearest = np.inf

        for j in range(0, END):
            if (i == j):
                continue
            dist = dist_points(map_points[i], map_points[j])
            if (dist < nearest):
                nearest = dist
            if (nearest < 10):
                break
        if (nearest < 10):
            modified_points.append(map_points[i])
    return modified_points

def main():
    laser_data = []

    robot_poses = []
    
    ROUND_FACTOR = 2

    START = 0
    END = len(laser_data)

    original_points = []
    for index in range(START, END):
        data_set = convert_data_set(laser_data[index], robot_poses[index])
        # Append the next points to the map
        for point in data_set:
            original_points.append((round(point[0] / 1000, ROUND_FACTOR), round(point[1] / 1000, ROUND_FACTOR)))
    original_points = np.unique(original_points, axis=0)

    map_points = []
    for point in convert_data_set(laser_data[START], robot_poses[START]):
            map_points.append((point[0], point[1]))
            #map_points.append((round(point[0] / 1000, ROUND_FACTOR), round(point[1] / 1000, ROUND_FACTOR)))
    map_poses = [np.zeros(2)]

    start_time = datetime.now()
    offset = np.zeros(3)
    thetas = [np.zeros(3)]
    errors = []
    corrected_points = map_points
    prev_points = map_points

    END = len(laser_data) - 200
    for index in range(START + 1, END):
        print("Iteration: ", (index + 1), " of ", END)

        robot_poses[index] = [robot_poses[index][0], robot_poses[index][1], robot_poses[index][2] * 0.97]
        
        # Convert the point set
        new_points = convert_data_set(laser_data[index], tuple(pos + offset for pos, offset in zip(robot_poses[index], offset)))

        # Find feature points
        feature_points = extract_feature_points(new_points)
        for feature in feature_points:
            new_points.remove(feature)
        
        # Calculate the icp matching
        R, T, error = icp_matching(corrected_points, new_points)
        #R, T, error = icp_matching(corrected_points, feature_points)

        # Calculate the new offset
        new_offset = (0, 0, 0)

        if (abs(R[1][0]) < 0.01):
            new_offset = (0, 0, R[1][0])
            
        offset = (offset[0] + new_offset[0], offset[1] + new_offset[1], offset[2] + new_offset[2])

        corrected_points = convert_data_set(laser_data[index], tuple(pos + offset for pos, offset in zip(robot_poses[index], offset)))

        if (np.array_equal((0, 0, 0), new_offset)):
            # Merge with previus point
            #threshold = 50 # TODO: original
            threshold = 100
            merge_list = []
            for point in corrected_points:
                nearest_point = min(prev_points, key = lambda p: (p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2)
                
                if (dist_points(point, nearest_point) < threshold):
                    merge_list.append(nearest_point)
                else:
                    merge_list.append(point)

            corrected_points = merge_list
            prev_points = corrected_points

            # Append the next points to the map
            for point in corrected_points:
                map_points.append((round(point[0] / 1000, ROUND_FACTOR), round(point[1] / 1000, ROUND_FACTOR)))
            map_poses.append(((robot_poses[index][0] + offset[0]) / 1000, (robot_poses[index][1] + offset[1]) / 1000))
        else:
            thetas.append(R[1][0])
            errors.append(error)
    
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    print("Total points before: ", len(map_points))
    map_points = np.unique(map_points, axis=0)
    print("Total points after: ", len(map_points))

    map_points, discard_map_points = clear_path(map_points, map_poses)
    
    map_points = clear_outliers(map_points)
    
    plt.cla()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    map_x, map_y = zip(*map_points)
    plt.plot(map_x, map_y, label="Kort", color='blue', marker='.', markersize=2, linestyle='None')
    #pos_x, pos_y = zip(*map_poses)
    #plt.plot(pos_x, pos_y, label="poses", color='green', marker='None', linewidth=1.5, linestyle='dashed')
    plt.axis("equal")
    plt.legend()
    plt.gca().set_position([0, 0, 1, 1])
    #plt.savefig("Corrected_optimized.pdf", bbox_inches='tight')
    plt.show()

    print('write file')
    with open("map_points.txt", 'w') as f:
        for i in map_points:
            f.write(str(i))

    exit(0)

if __name__ == '__main__':
    main()
