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

    # points[:, 0] /= max(points[:, 0])
    # points[:, 1] /= max(points[:, 1])
    points[:,0] += writebox[0]
    points[:,1] += writebox[2]
    return points 

def draw_character(char='0'):
    if char.isdigit():
        y_des = read_data('/home/hanka/Desktop/DMP/' + char, [0, 1, 0, 1]).T
    else:
        y_des = read_data('/home/hanka/Desktop/DMP/characters/' + char + "_letter", [0, 1, 0, 1]).T

    dmp = dmp_point_attractor.DMP_point_attractor(n_dmps = 2, n_bfs = 100)
    dmp.imitate_trajectory(y_des=y_des)
    y, dy, ddy = dmp.run()

    dmp_file = {'x' : y[:, 0].tolist(), 'y' : y[:, 1].tolist()}
 
    with open('/home/hanka/xarm6-etf-lab/src/etf_modules/sim_bringup/dmps/dmp.yaml', 'w') as file:
        yaml.dump(dmp_file, file, default_flow_style=False)

    show_character(y_des, y)
    return dmp.weights


def show_character(y_des, y):
    plt.figure(1, figsize=(6, 6))
    plt.plot(y_des[0, :], y_des[1, :], "r--", lw = 2)
    plt.plot(y[:, 0], y[:, 1], "g", lw=2)
    plt.title("DMP draw " + char)
    plt.legend(["desired path", "path"], loc = "lower right")
    plt.axis("equal")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2]) 
    plt.show()

if __name__ == "__main__":
    char = sys.argv[1]
    print(char)
    draw_character(char)

# print(open('dmp1.yaml').read())
# draw_character('o')
# draw_character('w')

# weights = draw_character('w')
# scale_x = 1.0
# scale_y = 2.5
# dmp = dmp_discrete.DMP_point_attractor(n_dmps = 2, n_bfs = 100, y0 = 0, goal = [scale_x, scale_y])
# dmp.weights = weights
# # dmp2 = dmp_discrete.DMP_point_attractor(dt = 0.001, n_dmps = 2, n_bfs = 100, run_time = 0.5, tau = 0.5, goal = [scale_x, scale_y])
# # dmp2.weights = weights

# # t = np.linspace(0, dmp.cs.run_time, dmp.timesteps)
# # t2 = np.linspace(0, dmp2.cs.run_time, dmp2.timesteps)
# y, dy, ddy = dmp.run()
# # y2, dy2, ddy2 = dmp2.run()

# plt.figure(2)
# # plt.subplot(211)
# # plt.plot(t, y[:, 0])
# # plt.xlim([0.0, 1.0])
# # plt.xlabel('time [s]')
# # plt.title('DMP 1 of letter w')
# # plt.grid()
# # plt.subplot(212)
# # plt.plot(t2, y2[:, 0])
# # plt.xlim([0.0, 0.5])
# # plt.title('DMP 1 of letter w sped up 2 times')
# # plt.grid()
# # plt.xlabel('time [s]')


# # plt.plot(y_des[0, :], y_des[1, :], "r--", lw = 2)
# plt.plot(-y[:, 0] + scale_x, -y[:, 1] + scale_y, "b", lw=2)
# # plt.plot(-y[:, 0], y[:, 1], "b", lw = 2)
# plt.title("DMP draw 6 x3")
# plt.legend(["desired path", "initial path", "scaled path"], loc = "lower right")
# plt.axis("equal")
# # plt.xlim([-0.5, 0.5 + scale_x])
# # plt.ylim([-0.5, 0.5 + scale_y])
# plt.show()  


