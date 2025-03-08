import numpy as np
from cs import CanonicalSystem

class DMP(object):

    def __init__(self, n_dmps, n_bfs, dt = 0.01, y0 = 0.0, goal = 1.0, tau = 1.0, weights = None, ay = None, by = None, **kwargs):
        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.dt = dt
        self.tau = tau
        self.cs = CanonicalSystem(dt = self.dt, tau = tau, **kwargs)
        self.timesteps = (int)(self.cs.timesteps)

        if isinstance(y0, (int, float)):
            y0 = np.ones(n_dmps) * y0
        self.y0 = y0
        if isinstance(goal, (int, float)):
            goal = np.ones(n_dmps) * goal
        self.goal = goal

        if weights is None:
            weights = np.zeros((self.n_dmps, self.n_bfs))
        self.weights = weights

        self.ay = np.ones(n_dmps) * 25.0 # Schaal 2012
        self.by = self.ay / 4.0 # Schaal 2012
        self.reset()

    def reset(self):
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset()

    def gen_psi(self):
        raise NotImplementedError()
    
    def gen_forcing_term(self, x, dmp_idx):
        raise NotImplementedError()
    
    def gen_weights(self, forcing_t_des):
        raise NotImplementedError()
    
    def step(self, error = 0.0):
        x = self.cs.step(error = error)
        psi = self.gen_psi(x)
        
        for i in range(self.n_dmps):
            f = self.gen_forcing_term(x, i) * (np.dot(psi, self.weights[i])) / np.sum(psi)
            self.ddy[i] = self.ay[i] * (self.by[i] * (self.goal[i] - self.y[i]) - self.dy[i]) + f
            self.dy[i] += self.ddy[i] * self.dt
            self.y[i] += self.dy[i] * self.dt

        return self.y, self.dy, self.ddy

    def run(self, **kwargs):
        self.reset()

        timesteps = self.timesteps

        y = np.zeros((timesteps, self.n_dmps))
        dy = np.zeros((timesteps, self.n_dmps))
        ddy = np.zeros((timesteps, self.n_dmps))

        for t in range(timesteps):
            y[t], dy[t], ddy[t] = self.step(**kwargs)

        return y, dy, ddy
          

    def imitate_trajectory(self, y_des):

        if y_des.ndim == 1:  
            y_des = y_des.reshape(1, len(y_des))

        self.y0 = y_des[:, 0].copy()
        self.goal = y_des[:, -1].copy()

        import scipy.interpolate

        path = np.zeros((self.n_dmps, self.timesteps))
        x = np.linspace(0, self.cs.run_time, y_des.shape[1])

        for i in range(self.n_dmps):
            path_gen = scipy.interpolate.interp1d(x, y_des[i])
            for t in range(self.timesteps):
                path[i, t] = path_gen(t*self.dt)

        y_des = path

        dy_des = np.gradient(y_des, axis = 1)/self.dt
        ddy_des = np.gradient(dy_des, axis = 1)/self.dt

        forcing_t_des = np.zeros((y_des.shape[1], self.n_dmps))
        
        for i in range(self.n_dmps):
            forcing_t_des[:, i] = ddy_des[i] - self.ay[i]*(self.by[i]*(self.goal[i] - y_des[i]) - dy_des[i])

        self.gen_weights(forcing_t_des)

        return self.run()
