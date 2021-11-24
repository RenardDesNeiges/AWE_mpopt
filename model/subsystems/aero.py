import numpy as np
from utils.quaternions import Quaternion as quat
from config.env import Physics

class Aero():
    
    @staticmethod
    def force(v, w, control, q_bn, n_vW, kite) -> np.array:
        """Computes body frame aerodynamic forces given control and state params

        Args:
            v (np.array): velocity
            w (np.array): angular velocity
            control (np.array): control vector
            q_bn (np.array): body orientation quaternion
            n_vW (np.array): wind vector
            kite (np.array): kite model

        Returns:
            np.array: total aerodynamic force
        """
        
        # Interpret control values

        dE = control[0]
        dR = control[1]
        

        # Dynamics model
        # Aerodynamic (Wind) frame
        # Wind and apparent velocity in body frame
        b_vW = quat.transform(q_bn, n_vW)
        b_va = v - b_vW
        
        
        # Aerodynamic variables (Airspeed, angle of attack, side slip angle)
        Va = np.linalg.norm(b_va)
        alpha = np.arctan(b_va[2] / b_va[0])
        beta = np.arcsin(b_va[1] / Va)

        # Aerodynamic Forces in aerodynamic (wind) frame
        q_ba = quat.mult(quat.t2(alpha), quat.t3(-beta))


        V0 = Va

        dyn_press = 0.5 * Physics.rho * Va**2

        CL = kite.CL0 + kite.CLa * alpha + kite.CLq * kite.c / (2.0 * V0) \
                                                * w[1] + kite.CLde * dE
        CD = kite.CD0 + CL**2 / (np.pi * kite.e_oswald * kite.AR)

        # Forces in x, y, z directions: -Drag, Side force, -Lift
        LIFT = dyn_press * kite.S * CL
        DRAG = dyn_press * kite.S * CD
        SF = dyn_press * kite.S * (kite.CYb * beta \
            + kite.b / (2.0 * V0) * (kite.CYp * w[0] \
            + kite.CYr * w[2]) + kite.CYdr * dR)

        a_F_aero = np.array([-DRAG, SF, -LIFT])

        # Aerodynamic Forces and Moments in body frame
        b_F_aero = quat.transform(q_ba, a_F_aero)
        
        return b_F_aero, Va, alpha, beta
    
    
    def moment(v, w, control, q_bn, n_vW, kite) -> np.array:
        """Computes body frame aerodynamic moment on kite

        Args:
            v (np.array): velocity
            w (np.array): angular velocity
            control (np.array): control vector
            q_bn (np.array): body orientation quaternion
            n_vW (np.array): wind vector
            kite (np.array): kite model

        Returns:
            [np.array]: body frame aerodynamic moment
        """
        # Dynamics model
        # Aerodynamic (Wind) frame
        # Wind and apparent velocity in body frame
        
        b_vW = quat.transform(q_bn, n_vW)
        b_va = v - b_vW
        
        dE = control[0]
        dR = control[1]
        dA = control[2]

        # Aerodynamic variables (Airspeed, angle of attack, side slip angle)
        Va = np.linalg.norm(b_va)
        alpha = np.arctan(b_va[2] / b_va[0])
        beta = np.arcsin(b_va[1] / Va)

        # Aerodynamic Moments in aerodynamic (wind) frame

        V0 = Va

        dyn_press = 0.5 * Physics.rho * Va^2

        # Moments about x, y, z axes: L, M, N
        L = dyn_press * kite.S * kite.b * (kite.Cl0 + kite.Clb * beta + \
            kite.b / (2.0 * V0) * (kite.Clp * w[0] + kite.Clr * w[2]) +\
            kite.Clda * dA + kite.Cldr * dR)

        M = dyn_press * kite.S * kite.c * (kite.Cm0 + kite.Cma * alpha + \
            kite.c / (2.0 * V0) * kite.Cmq * w[1] + kite.Cmde * dE)

        N = dyn_press * kite.S * kite.b * (kite.Cn0 + kite.Cnb * beta + \
            kite.b / (2.0 * V0) * (kite.Cnp * w[0] + kite.Cnr * w[2]) +\
            kite.Cnda * dA + kite.Cndr * dR)

        return np.array([L, M, N])