from model.kinematic import KinematicModel
from simulator import Simulator

import numpy as np


kin = KinematicModel("config/eg4_xflr.yaml")

x0 = kin.defaultState
u0 = kin.defaultControl
params = np.zeros(3)

x_dot_test = kin.dyn_f(x0, u0, params)

print(x_dot_test)

# sim = Simulator(kin.dyn_f)

# t0, t1 = 0, 5                # start and end
# T = np.linspace(t0, t1, 10)  # the points of evaluation of solution

# sim.simulate(x0, u0, params,T)

