import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
sys.path.append('../')

import dmp
import dmp_point_attractor

def read_data(file, writebox, spaces=False, rotate = False):

    f = open(file + '.txt', 'r')

    points = []
    for row in f:
        points.append(row.strip('\n').split(','))
    f.close()
    points = np.array(points, dtype='float')
    points = points[:, :2]

    # mirror along y-axis
    for i in range(points.shape[0]):
        points[i, 1] *= -1
        
    # center
    points[:, 0] -= np.min(points[:,0])
    points[:, 1] -= np.min(points[:,1])

    # normalize 
    maks = max(max(points[:, 0]), max(points[:, 1]))
    points[:, 0] /= maks
    points[:, 1] /= maks

    points[:,0] += writebox[0]
    points[:,1] += writebox[2]
    return points 

def draw_character(char='0'):
    if char.isdigit():
        y_des = read_data('/home/hanka/Desktop/DMP/' + char, [0, 1, 0, 1]).T
    else:
        y_des = read_data('/home/hanka/Desktop/DMP/characters/' + char + "_letter", [0, 1, 0, 1]).T

    character_dmp = dmp.DMP(dmps = 2, bfs = 5000)
    character_dmp.desired_trajectory(y_des=y_des)
    y, dy, ddy = character_dmp.run()
    
    dmp_file = {'x' : y[:, 0].tolist(), 'y' : y[:, 1].tolist()}
 
    with open('/home/hanka/xarm6-etf-lab/src/etf_modules/sim_bringup/dmps/dmp.yaml', 'w') as file:
        yaml.dump(dmp_file, file, default_flow_style=False)

    show_character(y_des, y, char)
    return character_dmp.weights


def show_character(y_des, y, char = 'traj'):
    plt.figure(1, figsize=(6, 6))
    plt.plot(y_des[0, :], y_des[1, :], "r", lw = 2)
    plt.plot(y[:, 0], y[:, 1], "g", lw=2)
    plt.title("DMP draw " + char)
    plt.legend(["desired path", "path"], loc = "lower right")
    plt.axis("equal")
    plt.xlim([-0.25, 1.25])
    plt.ylim([-0.25, 1]) 
    plt.show()

if __name__ == "__main__":
    char = sys.argv[1]
    print(char)
    draw_character(char)


