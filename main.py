from model.kinematic import KinematicModel

import numpy as np


kin = KinematicModel("config/eg4_xflr.yaml")

x0 = kin.defaultState
u0 = kin.defaultControl
params = np.zeros(3)

x_dot_test = kin.dyn_f(x0, u0, params)

