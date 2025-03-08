from dmp import DMP
import numpy as np


class DMP_point_attractor(DMP):

    def __init__(self, **kwargs):
        super(DMP_point_attractor, self).__init__(**kwargs)

        self.gen_centers()
        self.h = np.ones(self.n_bfs) * self.n_bfs ** 1.5 / self.centers / self.cs.ax
        self.reset()

    def gen_centers(self):
        # evenly spaced centers in time
        centers = np.linspace(0, self.cs.run_time, self.n_bfs)

        # centers in canonical system
        self.centers = np.ones(len(centers))
        for i in range(len(centers)):
            # map time to canonical system exponentionally as first order system
            self.centers[i] = np.exp(-self.cs.ax*centers[i])



    def gen_forcing_term(self, x, dmp_idx):
        return x * (self.goal[dmp_idx] - self.y0[dmp_idx])
    
    def gen_psi(self, x):
        if isinstance(x, np.ndarray):
            x = x[:, None] # try with reshape as well

        return np.exp(-self.h*(x - self.centers) ** 2)
    
    def gen_weights(self, forcing_t_des):
        x = self.cs.run()
        psi = self.gen_psi(x)

        self.weights = np.zeros((self.n_dmps, self.n_bfs))

        for i in range(self.n_dmps):
            k = self.goal[i] - self.y0[i]
            for j in range(self.n_bfs):
                a = np.sum(x * psi[:, j] * forcing_t_des[:, i])
                b = np.sum(x ** 2 * psi[:, j])
                self.weights[i, j] = a / b / k

# test

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dmp = DMP_point_attractor(dt = 0.01, n_dmps=1, goal = 2, n_bfs=10, tau = 1.0, weights=np.ones((1, 10)), ax = 1.0)
    # x = dmp.cs.run()
    t = np.linspace(0, dmp.cs.run_time, dmp.cs.timesteps)
    # plt.figure(1, figsize=(6, 3))
    # plt.plot(t, dmp.gen_psi(x))
    # plt.title('Basis functions activations')
    # plt.xlabel('time [s]')
    # plt.ylabel('psi')

    y, dy, ddy = dmp.run()
    # plt.figure(1, figsize=(6, 3))
    # plt.plot(np.ones(len(y)) * dmp.goal, "r--", lw=2)
    # plt.plot(y, lw=2)
    # plt.title("DMP system - no forcing term")
    # plt.xlabel("time (ms)")
    # plt.ylabel("system trajectory")
    # plt.legend(["goal", "system state"], loc="lower right")
    # plt.tight_layout()

    plt.figure(2, figsize=(18, 4))
    n_bfs = [80000]
    run_time = 5.0
    path1 = np.sin(np.arange(0, run_time, 0.01) * 5)
    path2 = np.zeros(path1.shape)
    path2[50:] = 0.5
    path2 = np.sin(np.arange(0, run_time, 0.01) * 5) + 0.2 * np.cos(np.arange(0, run_time, 0.01) * 2*np.pi + np.pi/7) + 0.1 * np.sin(np.arange(0, run_time, 0.01) * 4*np.pi + np.pi/2)
    t = np.linspace(0, dmp.cs.run_time, path2.shape[0])
    
    for ii, bfs in enumerate(n_bfs):
        dmp = DMP_point_attractor(n_dmps=2, n_bfs=bfs, run_time = run_time)

        dmp.imitate_trajectory(y_des=np.array([path1, path2]))
        dmp.goal[0] = 5
        dmp.goal[1] = 5
        y, dy, ddy = dmp.run()


        # spatial 

        # plt.figure(1)
        # # plt.subplot(211)
        # a = plt.plot(path1 / path1[-1] * dmp.goal[0], "r--", lw=2)
        # b = plt.plot(path2 / path2[-1] * dmp.goal[1], "g--", lw=2)
        # plt.plot(y[:, 0], lw = 2)
        # # plt.subplot(212)
        # plt.plot(y[:, 1], lw = 2)


        # bfs

        plt.figure(2)
        plt.subplot(311)
        plt.plot(y[:, 1], lw=2)
        plt.subplot(312)
        plt.plot(dy[:, 1]/100, lw=2)
        plt.subplot(313)
        plt.plot(ddy[:, 1]/10000, lw=2)


    # spatial
    
    # plt.title("DMP")
    # plt.xlabel("time (ms)")
    # plt.ylabel("system trajectory")
    # plt.legend(["desired path" ,"desired scaled path", "DMP", "DMP after goal change"])


    # bfs

    # plt.subplot(311)
    # a = plt.plot(path1 / path1[-1] * dmp.goal[1], "r--", lw=2)
    # plt.title("DMP imitate path")
    # plt.xlabel("time (ms)")
    # plt.ylabel("system trajectory")
    # plt.legend([a[0]], ["desired path"], loc="lower right")
    plt.subplot(311)
    path_des = path2 / path2[-1] * dmp.goal[1]
    b = plt.plot(-path_des, "r--", lw=2)
    plt.title("system trajectory")
    plt.xlabel("time (ms)")
    plt.ylabel("y")
    plt.legend(["%i BFs" % i for i in n_bfs], loc="lower right")
    plt.grid()
    plt.tight_layout()

    plt.subplot(312)
    dt = np.diff(t)
    dpath_des = np.diff(path_des)
    b = plt.plot(-dpath_des, "r--", lw=2)
    plt.title("velocity")
    plt.xlabel("time (ms)")
    plt.grid()
    plt.tight_layout()
    plt.ylabel("dy")


    plt.subplot(313)
    ddpath_des = np.diff(dpath_des)
    b = plt.plot(-ddpath_des, "r--", lw=2)
    plt.title("accelaration")
    plt.xlabel("time (ms)")
    plt.ylabel("ddy")
    plt.grid()
    plt.tight_layout()
    plt.show()

    