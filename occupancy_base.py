# This is an implementation of Occupancy Grid Mapping as Presented
# in Chapter 9 of "Probabilistic Robotics" By Sebastian Thrun et al.
# In particular, this is an implementation of Table 9.1 and 9.2

import csv
from scipy import ndimage
import scipy.io
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from tqdm import tqdm
import math
import time
import pickle

class Map():
    def __init__(self, xsize, ysize, grid_size):
        self.xsize = xsize+2 # Add extra cells for the borders
        self.ysize = ysize+2
        self.grid_size = grid_size # save this off for future use
        self.log_prob_map = np.zeros((self.xsize, self.ysize)) # set all to zero

        self.alpha = 1.0 # The assumed thickness of obstacles
        self.beta = 3.0*np.pi/180.0 # The assumed width of the laser beam
        self.z_max = 150.0 # The max reading from the laser

        # Pre-allocate the x and y positions of all grid positions into a 3D tensor
        # (pre-allocation = faster)
        self.grid_position_m = np.array([np.tile(np.arange(0, self.xsize*self.grid_size, self.grid_size)[:,None], (1, self.ysize)),
                                         np.tile(np.arange(0, self.ysize*self.grid_size, self.grid_size)[:,None].T, (self.xsize, 1))])

        # Log-Probabilities to add or remove from the map
        # self.l_occ = np.log(0.65/0.35)
        # self.l_free = np.log(0.35/0.65)

        self.l_occ = np.log(0.65 / 0.35)
        self.l_free = np.log(0.15 / 0.65)


    def update_map(self, pose, z):

        dx = self.grid_position_m.copy() # A tensor of coordinates of all cells
        dx[0, :, :] -= pose[0] # A matrix of all the x coordinates of the cell
        dx[1, :, :] -= pose[1] # A matrix of all the y coordinates of the cell
        theta_to_grid = np.arctan2(dx[1, :, :], dx[0, :, :]) - pose[2] # matrix of all bearings from robot to cell

        # Wrap to +pi / - pi
        theta_to_grid[theta_to_grid > np.pi] -= 2. * np.pi
        theta_to_grid[theta_to_grid < -np.pi] += 2. * np.pi

        dist_to_grid = scipy.linalg.norm(dx, axis=0) # matrix of L2 distance to all cells from robot

        # For each laser beam
        for z_i in z:
            r = z_i[0] # range measured
            b = z_i[1] # bearing measured
            # print(f'The range is {r} and the bearing is {b}')
            # Calculate which cells are measured free or occupied, so we know which cells to update
            # Doing it this way is like a billion times faster than looping through each cell (because vectorized numpy is the only way to numpy)
            free_mask = (np.abs(theta_to_grid - b) <= self.beta/2.0) & (dist_to_grid < (r - self.alpha/2.0))
            occ_mask = (np.abs(theta_to_grid - b) <= self.beta/2.0) & (np.abs(dist_to_grid - r) <= self.alpha/2.0)

            # Adjust the cells appropriately
            self.log_prob_map[occ_mask] += self.l_occ
            self.log_prob_map[free_mask] += self.l_free


def is_in_LOS(points,index, location, ax1):
    angle = math.atan2(index[0] - location[0], index[1] - location[1])
    i = 1
    path_point = np.array([int(i*math.sin(angle))+location[0], int(i*math.cos(angle))+location[1]])
    wall_counter = 0
    while np.linalg.norm(path_point-index)>=3:
        i += 1
        path_point = np.array([int(i*math.sin(angle))+location[0], int(i*math.cos(angle))+location[1]])
        if points[path_point[0], path_point[1]] == 1:
            wall_counter += 1
            break
    return wall_counter == 0


#if __name__ == '__main__':
def make_grid(old_map, point):
    P = []
    L = []
    L_3d = []
    height_threshold_low = -2
    height_threshold_high = 2
    # center_point = pickle.load(open("room_center.p", "rb"))
    # center_point = [center_point[1], center_point[0]]
    center_point = np.flip(point)
    with open('room.txt') as f:
    # center_point = [0, 0]
    # with open('cloud_full.csv') as f:
    # with open('OFFICE.txt') as f:
    # with open('Corridore.txt') as f:
        lines = f.readlines()


    center = []
    for line in lines:
        line_data = line.split(',')
        if height_threshold_low< float(line_data[1]) < height_threshold_high:
            # center_point = [0, 0]
            L_3d.append([float(line_data[0]), float(line_data[1]), float(line_data[2])])
            p_i = [float(line_data[0]), float(line_data[2])]
            P.append([p_i])
            angle = math.atan2(p_i[0]-center_point[0], p_i[1]-center_point[1])
            dist = np.linalg.norm([p_i[0]-center_point[0], p_i[1]-center_point[1]])
            # L.append([[10*dist, angle]])# distance increased 10 times
            L.append([[1 * dist, 0]])  # distance increased 10 times
            center.append([center_point[0], center_point[1], angle])
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    P = np.array(P).T
    L = np.array(L).T
    L_3d = np.array(L_3d).T
    center = np.array(center).T
    meas = L
    los_map = list(pickle.load(open("room_outline.p", "rb")))
    gen_map = pickle.load(open("room_outline.p", "rb"))
    los_map = np.array(los_map)
    len_map_x = (los_map.shape[0])
    len_map_y = (los_map.shape[1])
    grid_size = 1.0
    map = Map(int(len_map_x/grid_size), int(len_map_y/grid_size), grid_size)
    fig1, ax1 = plt.subplots(2, 1)
    los_point = center_point

    seen_map = np.zeros(los_map.shape)
    seen_map_list = []
    seen_center = []
    for i in range(los_map.shape[0]):
        for j in range(los_map.shape[1]):
            if los_map[i, j] == 1 and is_in_LOS(los_map, [i, j], los_point, ax1):
                #ax1[0].plot(j, i, 'g*')
                seen_map[i+int(np.random.normal(0, 1, 1)),j+int(np.random.normal(0, 1, 1))] = 1
                dist = np.linalg.norm([i - center_point[0], j - center_point[1]])
                angle = math.atan2(i - center_point[0], j - center_point[1])
                seen_center.append([center_point[1], center_point[0], angle])
                seen_map_list.append([[dist, 0]])


    meas = np.array(seen_map_list).T
    center = np.array(seen_center).T
    for i in tqdm(range(len(meas.T))):
        # map.update_map(state[:,i], meas[:,:,i].T) # update the map
        map.update_map(center[:,i], meas[:, :, i].T)  # update the map
        # # Real-Time Plotting
        # # (comment out these next lines to make it run super fast, matplotlib is painfully slow)
        # plt.clf()
        # # pose = state[:,i]
        # pose = center[:,i]
        # circle = plt.Circle((pose[1], pose[0]), radius=3.0, fc='y')
        # plt.gca().add_patch(circle)
        # arrow = pose[0:2] + np.array([3.5, 0]).dot(np.array([[np.cos(pose[2]), np.sin(pose[2])], [-np.sin(pose[2]), np.cos(pose[2])]]))
        # plt.plot([pose[1], arrow[1]], [pose[0], arrow[0]])
        # plt.imshow(1.0 - 1./(1.+np.exp(map.log_prob_map)), 'Greys')
        # plt.pause(0.005)





    # Final Plotting
    #ax1[0].plot(P[0,0,:], P[1,0,:], 'r+')
    ax1[0].plot(center_point[1], center_point[0], 'k*')
    ax1[0].imshow(seen_map)

    len_map_x = (los_map.shape[0])
    ax1[0].axis('equal')

    # plt.ioff()
    # plt.clf()
    low_tol = -1
    high_tol = 0.2
    total_map = np.zeros((map.log_prob_map.shape[0], map.log_prob_map.shape[1]))
    for i in range(map.log_prob_map.shape[0]):
        for j in range(map.log_prob_map.shape[1]):
            if map.log_prob_map[i, j] < low_tol or old_map.log_prob_map[i, j] < low_tol and np.sum(old_map.log_prob_map) != 0 :
                total_map[i, j] = -5
            if map.log_prob_map[i, j] > high_tol or old_map.log_prob_map[i, j] > high_tol and np.sum(old_map.log_prob_map) != 0 :
                total_map[i, j] = 1
    image = (1.0 - 1./(1.+np.exp(total_map)))
    # image = (map.log_prob_map)
    ax1[1].imshow(image.T, 'Greys')  # This is probability
    ax1[1].plot(center_point[1], center_point[0], 'b*')
    # ax1[1].imshow(map.log_prob_map, 'Greys') # log probabilities (looks really cool)
    map_for_nav = (1.0 - 1./(1.+np.exp(total_map)))
    # pickle.dump(map_for_nav, open("map.p", "wb"))
    # pickle.dump(los_point, open("los_point.p", "wb"))
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # # Make data.
    # X = L_3d[0,:]
    # Y = L_3d[2,:]
    # # X, Y = np.meshgrid(X, Y)
    # Z = L_3d[1,:]
    #
    # # Plot the surface.
    # surf = ax.scatter(X, Y, Z,
    #                        linewidth=0, antialiased=False)
    # ax.set_box_aspect([1, 1, 1])
    plt.show()
    map.log_prob_map = total_map
    return map_for_nav, map

