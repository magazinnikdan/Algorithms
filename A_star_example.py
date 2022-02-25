import pickle
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import time

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares
        # for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Adjacent squares
            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            # ############################################## Added 0.5 as accepted value! ##############################
            if maze[node_position[0]][node_position[1]] != 0 and maze[node_position[0]][node_position[1]] > 0.1:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)


# def is_top_border(map):
#     n = 3
#     top_borders = []
#     for i in range(n,len(map[0,:])-n):
#         for j in range(n,len(map[0,:])-n):
#             if map[i, j] == 0 and map[i, j+1] == 0 and map[i-1, j] > 0 and map[i-1, j+1] > 0 and map[i-1, j] < 0.6 and map[i-1, j+1] < 0.6 and map[i-2, j] ==0.5 and map[i-2, j+1] ==0.5:
#                 top_borders.append([i,j])
#     return np.array(top_borders)


def count_value(map, min_value,max_value ):
    counter = 0
    for i in range(len(map[0, :])):
        for j in range(len(map[1, :])):
            if map[i,j] >= min_value and map[i,j] <= max_value:
                counter += 1
    return counter


def find_borders(map, n):
    borders = []
    zero_thrash = n/2
    unknown_thrash = n / 3
    wall_thrash = 1
    for i in range(n,len(map[0,:])-n):
        for j in range(n,len(map[1,:])-n):
            test_area = map[i-n:i+n,j-n:j+n]
            zero_counter = (2 * n)**2 - np.count_nonzero(test_area)
            unknown_counter = count_value(test_area, 0.5,0.5)
            wall_counter = count_value(test_area, 0.6,1)
            if zero_counter > zero_thrash and unknown_counter > unknown_thrash and wall_counter < wall_thrash:
                borders.append([j, i])
    return np.array(borders)


def filter_unknown(unfiltered_map, n):
    filtered_map = unfiltered_map
    zero_thrash = (2*n)**2*0.75
    unknown_thrash = (2*n)**2*0.2
    wall_thrash = 1
    for i in range(n, len(unfiltered_map[0, :])-n):
        for j in range(n, len(unfiltered_map[1, :])-n):
            test_area = unfiltered_map[i-n:i+n, j-n:j+n]
            zero_counter = (2 * n)**2 - np.count_nonzero(test_area)
            unknown_counter = count_value(test_area, 0.5, 0.5)
            wall_counter = count_value(test_area, 0.6, 1)
            if zero_counter >= zero_thrash and unknown_counter >= unknown_thrash and wall_counter < wall_thrash:
                filtered_map[i-n:i+n,j-n:j+n] = 2
    filtered_map[filtered_map > 1] = 0
    return filtered_map



def best_border(map ,borders, start, flag):
    counter_min = np.inf
    min_distance = np.inf
    max_distance = 0
    good_border = []
    for border in borders:
        counter = np.sum(borders-border)
        distance = np.linalg.norm(start-border)
        if counter <= counter_min and map[border[0], border[1]] == 0 and flag == 'big':
            counter_min = counter
            good_border = border
        if distance <= min_distance and map[border[0], border[1]] == 0 and flag == 'close':
            min_distance = distance
            good_border = border
        if distance >= max_distance and map[border[1], border[0]] == 0 and flag == 'far':
            max_distance = distance
            good_border = border
    return np.array(good_border)


def thicken_walls(unthickened_map, n):
    new_map = unthickened_map
    for i in range(n, len(unthickened_map[0, :]) - n):
        for j in range(n, len(unthickened_map[1, :]) - n):
            if 0.5 < unthickened_map[i, j] <= 1:
                new_map[i - n:i + n, j - n:j + n] = 2
    new_map[new_map > 1] = 1
    return new_map


def main(mat, start):
    n = 6
    borders_flag = 'far'
    maze = [[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    maze = list(pickle.load(open("map.p", "rb")).round(1))
    maze = list(mat.round(1))
    # start = pickle.load(open("los_point.p", "rb"))
    start = list(np.flip(start))
    tic = time.perf_counter()
    maze_array = np.array(maze)
    maze_array = filter_unknown(maze_array, 3)
    maze_array = thicken_walls(maze_array, 4).T
    maze = list(maze_array)
    new_borders = find_borders(maze_array, n)
    good_border = best_border(maze_array, new_borders, start, borders_flag) # 'close' finds the closest navigation point, 'big' finds the bigest one
    k = 1
    border_size = n - k
    while len(good_border) < 1:
        border_size = n-k
        new_borders = find_borders(maze_array, border_size)
        good_border = best_border(maze_array, new_borders, start, borders_flag) # 'close' finds the closest navigation point, 'big' finds the bigest one
        if border_size == 1:
            break
        k += 1



    end = (good_border[1], good_border[0])
    fig, ax = plt.subplots(1)
    rotated_img = ndimage.rotate(maze_array, -0)
    ax.imshow(maze_array)
    ax.plot(new_borders[:, 0], new_borders[:, 1], 'c4')
    ax.plot(start[1], start[0], 'r*')
    ax.plot(end[1], end[0], 'b3')

    plt.show(block=False)
    #plt.pause(1)
    if border_size >= 3:
        path = astar(maze, start, end)

        ax.imshow(rotated_img)
        path_array = []
        for p in range(len(path)):
            path_array.append(list(path[p]))
        path_array = np.array(path_array)
        ax.plot(path_array[:,1], path_array[:,0], 1, linewidth = 3)
    else:
        ax.annotate('DONE!', (75, 75), fontsize=30, color="red")
    plt.show()
    return np.flip(end), border_size
# if __name__ == '__main__':
#     main()
#     plt.show()

