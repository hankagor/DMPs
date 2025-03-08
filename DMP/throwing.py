import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
sys.path.append('../')

import dmp
from draw_character import read_data, show_character


def parabola():
    y_des = read_data('/home/hanka/Desktop/DMP/parabola', [0, 1, 0, 1]).T
    x = y_des[0, -1] - y_des[0, 0]
    y = y_des[1, -1] - y_des[1, 0]
    y_des[0, :] /= x
    y_des[1, :] /= y
    
    y_des[0, :] *= 0.5
    y_des[1, :] *= 0.1
    
    y_des[0, :] += -0.5 * np.ones(np.shape(y_des[0, :]))
    y_des[1, :] += 0.4 * np.ones(np.shape(y_des[1, :]))
    plt.figure(2, figsize=(6, 6))
    plt.plot(y_des[0, :], y_des[1, :], "r", lw = 2)
    plt.show()
    traj_dmp = dmp.DMP(dmps = 2, bfs = 5000)
    traj_dmp.desired_trajectory(y_des=y_des)
    y, dy, ddy = traj_dmp.run()
    
    dmp_file = {'x' : y[:, 0].tolist(), 'y' : y[:, 1].tolist()}
 
    with open('/home/hanka/xarm6-etf-lab/src/etf_modules/sim_bringup/dmps/dmp.yaml', 'w') as file:
        yaml.dump(dmp_file, file, default_flow_style=False)

    show_character(y_des, y)
    return traj_dmp.weights
        


if __name__ == "__main__":
    parabola()
