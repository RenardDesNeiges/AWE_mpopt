from scipy.integrate import solve_ivp
import numpy as np

class Simulator:
    
    def __init__(self, dyn_f):
        self.dyn_f = dyn_f
        
    def simulate(self, x0, u0, params, T):
        
        X = np.zeros((len(T), len(x0)))   # array for solution
        
        def f_dot(t,x):
            self.dyn_f(x,u0, params)
        
        sol1 = solve_ivp(f_dot, T, x0)