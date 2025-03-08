from dmp import DMP
import numpy as np


class DMP_point_attractor(DMP):

    def __init__(self, **kwargs):
        super(DMP_point_attractor, self).__init__(**kwargs)

        self.find_centers()
        self.h = np.ones(self.bfs) * self.bfs ** 1.5 / self.centers / self.cs.ax
        self.reset()

    def find_centers(self):
        # evenly spaced centers in time
        centers = np.linspace(0, self.cs.run_time, self.bfs)

        # centers in canonical system
        self.centers = np.ones(len(centers))
        for i in range(len(centers)):
            # map time to canonical system exponentionally as first order system
            self.centers[i] = np.exp(-self.cs.ax*centers[i])



    def find_forcing_term(self, x, dmp_idx):
        return x * (self.goal[dmp_idx] - self.y0[dmp_idx])
    
    def find_psi(self, x):
        if isinstance(x, np.ndarray):
            x = x[:, None] # try with reshape as well

        return np.exp(-self.h*(x - self.centers) ** 2)
    
    def find_weights(self, forcing_t_des):
        x = self.cs.run()
        psi = self.find_psi(x)

        self.weights = np.zeros((self.dmps, self.bfs))

        for i in range(self.dmps):
            val = self.goal[i] - self.y0[i]
            for j in range(self.bfs):
                a = np.sum(x * psi[:, j] * forcing_t_des[:, i])
                b = np.sum(x ** 2 * psi[:, j])
                self.weights[i, j] = a / b / val

    