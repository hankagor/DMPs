import numpy as np

class CanonicalSystem:
    def __init__(self, dt = 0.001, ax = 1.0, run_time = 1.0):
        self.dt = dt
        self.run_time = run_time
        self.timesteps = int(self.run_time / self.dt)
        self.ax = ax
        self.reset()

    def reset(self, initial_value = 1.0):
        # reset the canonical system
        self.curr_x = initial_value

    def step(self, tau = 1.0):
        # x = x' * dt = (-ax * x / tau) * dt
        self.curr_x += (-self.ax * self.curr_x ) * self.dt / tau
        return self.curr_x
        
    def run(self, tau = 1.0):
        # generate discrete canonical system
        
        self.x = np.zeros(int(self.timesteps * tau))
        self.reset()

        for i in range(self.timesteps):
            self.x[i] = self.curr_x
            self.step(tau)
    
        return self.x

    



