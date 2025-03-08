import numpy as np

class CanonicalSystem:
    def __init__(self, tau = 1.0, dt = 0.001, ax = 1.0, a_err = 1.0, run_time = 1.0):
        self.dt = dt
        self.run_time = run_time
        self.timesteps = int(self.run_time/self.dt/tau)
        self.ax = ax
        self.a_err = a_err
        self.reset()
        self.tau = tau

    def reset(self, initial_value = 1.0):
        self.curr_x = initial_value
    
    def run(self, error = None, **kwargs):
        self.x = np.zeros(self.timesteps)
        self.reset()
        for t in range(self.timesteps):
            self.x[t] = self.curr_x
            self.step(**kwargs)

        return self.x

    def step(self,  error = 0.0):
        # x = x' * dt = (-ax * x / tau) * dt
        self.curr_x += (-self.ax * self.curr_x )/(1.0 + self.a_err * error) * self.dt * self.tau
        return self.curr_x
        


import matplotlib.pyplot as plt
# cs1 = CanonicalSystem(tau = 0.2)
# cs2 = CanonicalSystem(tau = 0.5)
# cs1.run()
# cs2.run()
# plt.plot(cs1.x)
# plt.plot(cs2.x)
# plt.title('Canonical system')
# plt.xlabel('time [ms]')
# plt.ylabel('x')
# plt.legend(['tau = 0.2', 'tau = 0.5'])
# plt.tight_layout()
# plt.grid()
# plt.show()

# cs3 = CanonicalSystem(a_err = 3.0, run_time = 1.3)
# error = np.zeros(cs3.timesteps)

# for i in range(200, 400):
#     error[i] = 0.4*(i - 200)/100


# for i in range(400, 500):
#     error[i] = 0.8*(500 - i)/100
# # error[100:400] = 0.7*np.ones(300)

# x = np.zeros(cs3.timesteps)
# for i in range(cs3.timesteps):
#     x[i] = cs3.step(error=error[i])

# x2 = np.zeros(cs3.timesteps)
# cs3.reset()
# cs3.a_err = 1
# for i in range(cs3.timesteps):
#     x2[i] = cs3.step(error=error[i])
# x_no_err = cs3.run()
# plt.plot(error)
# plt.plot(x_no_err)
# plt.plot(x)
# plt.plot(x2)
# plt.grid()
# plt.tight_layout()
# plt.legend(['error', 'a_err = 0.0', 'a_err = 3.0', 'a_err = 1.0'])
# plt.xlabel('time [ms]')
# plt.ylabel('x')
# plt.title('Canonical system with error coupling')
# plt.show()



