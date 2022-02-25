import pickle
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import time
import occupancy_base
import A_star_example
import find_polygon

start_point = [30, 50]
mat = find_polygon.main(start_point)
len_map_x = (mat.shape[0])
len_map_y = (mat.shape[1])
grid_size = 1.0
map_0 = occupancy_base.Map(int(len_map_x/grid_size), int(len_map_y/grid_size), grid_size)
start_point = [start_point[1], start_point[0]]
mat, occ_grid = occupancy_base.make_grid(map_0, start_point)
next_point, size = A_star_example.main(mat, start_point)
# nnn = [next_point[1], next_point[0]]
while size >= 4:
    #mat = find_polygon.main(nnn)
    mat, occ_grid = occupancy_base.make_grid(occ_grid, next_point)
    next_point, size = A_star_example.main(mat, next_point)
    print(size)
    # nnn = [next_point[1], next_point[0]]

#mat = find_polygon.main(nnn)
# mat, occ_grid = occupancy_base.make_grid(occ_grid, next_point)
# next_point, size = A_star_example.main(mat, next_point)
