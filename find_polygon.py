import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import time
import math
import pickle
import occupancy_base
import A_star_example

def make_side(M, side_start,side_end, direction, side,  noise, type):
    for i in range(side_start, side_end):
        index = int(np.random.normal(0, noise, 1))
        if type == 'wall':
            object = 1
        else:
            object = 0
        if direction == 'vertical':
            M[i, side+index] = object
        else:
            M[side + index, i] = object
    return M


def find_rotation(M):
    max_sum = 0
    for rotation in range(90):
        M_rotated = rotate(M, angle=-rotation)
        M_rotated = M_rotated.round(decimals=0, out=None)
        max_sum_line = 0
        for j in range(M_rotated.shape[1]):
            sum_of_line = M_rotated[j].sum()
            #print(f'Sum of line {j} is {sum_of_line}')
            if sum_of_line > max_sum_line:
                max_sum_line = sum_of_line
                line = j
        if max_sum_line > max_sum:
            max_sum = max_sum_line
            best_rotation = rotation
            rotated_matrix = M_rotated
            best_line = line
    return best_rotation, rotated_matrix, best_line

def main(center_input):
    H = 150
    W = 150

    top_line = 20
    bottom_line = 60
    left_line = 10
    right_line = 80

    top_line_2 = 70
    bottom_line_2 = 125
    right_line_2 = 130




    rot_angle = 0
    opening_start = 50
    opening_end = 70


    search_width = 5
    search_window = 5
    search_thresh_low = 0.1
    search_thresh_high = 0.8
    opening_recognition_count = 3
    M = np.zeros((H, W))
    for i in range(2):
        M = make_side(M, top_line, bottom_line, 'vertical', left_line, 0, 'wall')
        M = make_side(M, top_line, bottom_line+40, 'vertical', right_line, 0, 'wall')
        M = make_side(M, left_line, right_line, 'horizontal', top_line, 0, 'wall')
        M = make_side(M, left_line, right_line-30, 'horizontal', bottom_line, 0, 'wall')

        M = make_side(M, left_line, right_line_2, 'horizontal', bottom_line_2, 0, 'wall')
        M = make_side(M, bottom_line, bottom_line_2, 'vertical', left_line, 0, 'wall')
        M = make_side(M, top_line_2, bottom_line_2, 'vertical', right_line_2, 0, 'wall')
        M = make_side(M, right_line, right_line_2, 'horizontal', top_line_2, 0, 'wall')
        M = make_side(M, left_line+30, right_line, 'horizontal', top_line_2+20, 0, 'wall')

    K = np.zeros((2*H, 2*W))


    P = []
    L = []
    with open('OFFICE.txt') as f:
        lines = f.readlines()

    center = np.array([0, 0])

    for line in lines:
        line_data = line.split(',')
        p_i = [float(line_data[0]),float(line_data[2])]
        P.append([p_i])
        angle = math.atan(p_i[0]/ p_i[1])
        dist = np.linalg.norm(p_i)
        L.append([[dist,angle]])
    P = np.array(P).T
    L = np.array(L).T




    # # Make an openeing
    # for i in range(10):
    #     M = make_side(M, opening_start, opening_end, 'horizontal', top_line, i % 2+1, 'opening')
    #

    #
    # M = make_side(M, opening_start, opening_end, 'horizontal', top_line-25, 9, 'wall')
    # M = make_side(M, opening_start, opening_end, 'horizontal', top_line-25, 3, 'wall')
    # M = make_side(M, opening_start, opening_end, 'horizontal', top_line-25, 7, 'wall')
    # M = make_side(M, opening_start, opening_end, 'horizontal', top_line-25, 6, 'wall')
    # M = make_side(M, opening_start, opening_end, 'horizontal', top_line-25, 5, 'wall')
    #

    # rotated = rotate(M, angle=rot_angle)
    # rotated = rotated.round(decimals=0, out=None)
    # points = np.nonzero(mat)
    # # plt.colorbar()
    # tic = time.perf_counter()
    # rot, mat, line = find_rotation(rotated)
    # line = 20
    # print(f"The rotation is {rot}, best line is: {line} calculated in {time.perf_counter()-tic} seconds")
    # #plt.imshow(mat)

    points = np.nonzero(M)
    num_not_zero = len(points[0])
    # K[center[0][1:10], center[1][1:10]] = 1

    points_x = np.array(points[0])
    points_y = np.array(points[1])
    point_mat = np.column_stack((points_x, points_y))
    center = center_input
    point_distances = []
    room_simulation = []
    max_dist = 0
    for i in range(num_not_zero):
        dist = np.linalg.norm(point_mat[i]-center)
        room_simulation.append([point_mat[i][0],0,point_mat[i][1], dist])
        if i == 0:
            with open('room.txt', 'w') as f:
                f.write(f'{point_mat[i][0]},{0},{point_mat[i][1]}, {dist}')
                f.write('\n')
        else:
            with open('room.txt', 'a') as f:
                f.write(f'{point_mat[i][0]},{0},{point_mat[i][1]}, {dist}')
                f.write('\n')
        point_distances.append(dist)
        if dist > max_dist:
            max_dist_x = point_mat[i, 0]
            max_dist_y = point_mat[i, 1]
            max_dist = dist
    pickle.dump(room_simulation, open("room.p", "wb"))

    # search_windows = []
    # for j in range(0, mat.shape[0],search_window ):
    #     search_windows.append(np.sum(np.nonzero(mat[line-search_width:line+search_width, j:j+search_window])))
    # window_counter = 0
    # line_index = np.array(np.nonzero(search_windows)[0])
    # line_values =[]
    # for k in range(len(line_index)):
    #     line_values.append(search_windows[line_index[k]])
    #
    #
    # line_average = np.average(search_windows)
    #
    #
    # opening_recognition_counter = 0
    # opening_first = -1
    # opening_last = -1
    # for i in range(len(search_windows)):
    #    # if search_windows[i] > search_thresh_low * line_average and search_windows[i] < search_thresh_high * line_average :
    #     if search_thresh_low * line_average < search_windows[i] < search_thresh_high * line_average:
    #         opening_recognition_counter += 1
    #         if opening_first < 0:
    #             opening_first = i
    #     else:
    #         opening_recognition_counter = 0
    #         opening_first = -1
    #     if opening_recognition_counter >= opening_recognition_count:
    #         opening_last = i
    #         print(f"Opening between {opening_first} and {opening_last}")
    #         # plt.plot([opening_first*search_window, opening_last*search_window] ,[line, line], linewidth=3)

    #print(search_windows)
    plt.imshow(M)
    plt.plot(center[1], center[0],'r*')
    pickle.dump(M, open("room_outline.p", "wb"))
    ccc= [center[1], center[0]]
    pickle.dump(ccc, open("room_center.p", "wb"))
    P = np.array(P)
    # plt.plot(P[0,0,:], P[1,0,:], 'r*')
    # plt.plot(P[1,0,2],P[0,0,2], 'g*')
    plt.axis('equal')
    # plt.plot(L[:,0], L[:,1], 'g*')
    plt.show()
    return M
