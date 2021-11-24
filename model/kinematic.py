from utils.config_loader import ConfigParser
from utils.quaternions import Quaternion as quat
from model.subsystems.aero import Aero
from model.model import Model
from config.env import Physics
import numpy as np

class KinematicModel(Model):
    
    def __init__(self, config_file):
        """Generates a kinematic model object from a configuration file

        Args:
            config_file (string): address of yaml configuration file
        """
        
        Model.__init__(self,config_file)
        
        self.nx = 10;         # Number of states
        self.nu = 4;          # Number of inputs
        self.nd = 3;          # Number of dynamic params

        # self.sys             # Struct containing system properties
        self.teth_on = False # if true then we model the tether
        
        
        self.StateName = ['vx' 'vy' 'vz' \
                        'posN' 'posE' 'posD'\
                        'qw' 'qx' 'qy' 'qz']
        self.StateUnit = ['m/s' 'm/s' 'm/s' \
                        'm' 'm' 'm' \
                        '-' '-' '-' '-']
        
        self.InputName = ['wE' 'wR' 'wA', 'thrust']
        self.InputUnit = ['rad/s' 'rad/s' 'rad/s' 'N']
        self.ForceLabels = ['Fth','Fth_lon','Dt','Wt','Fa']
        
        self.phyUBU = np.array([np.deg2rad(80),  np.deg2rad(60), np.deg2rad(80), 6])
        self.phyLBU = np.array([-np.deg2rad(80), -np.deg2rad(60), -np.deg2rad(80), 0])
        
        self.defaultState = np.array([12, -0.01, -0.2, \
                            0.1, 0.1, -100, \
                            1, 0, 0, 0])
                        
        self.defaultControl = np.array([0, 0, 0, 0])
        
    def dyn_f(self,x,u, params) -> np.array:
        """Dynamics function (overides the base class dynamics function)

        Args:
            x (np.array): state
            u (np.array): control
            params (np.array): parameters

            [np.array]: state derivative x_dot
        """
        
        # Decompose control
        wE = u[0];
        wR = u[1];
        wA = u[2];
        w = np.array([wA,wE,wR]).T # Because the model is kinematic
        
        # Decompose state
        v = x[0:3];
        pos = x[3:6];
        q = x[6:10];
        q_bn = quat.inverse(q);
        n_vW = params[0:3];
        
        # Forces and moments
        
        # aerodynamic force and moment
        b_F_aero,Va,alpha,beta = Aero.force(v,w,np.zeros(4),q_bn,n_vW,self)

        # Gravitational acceleration
        b_g = quat.transform(q_bn, np.array([0, 0, Physics.g]))
        
        # tether force and moment
        b_F_tether = np.array([0,0,0])
        # if obj.teth_on
        #     [b_F_tether,~] = fixed_tether(pos,q_bn,n_vW,obj);
        # end 
        
        # engine thrust
        thrust = u[3]
        b_F_thrust = np.array([thrust,0,0])
        
        # Motion equations (body frame)

        # Linear motion equation
        spec_nongrav_force = (b_F_aero + b_F_thrust + b_F_tether) / self.mass;
        v_dot = spec_nongrav_force + b_g - np.cross(w, v)
        
        # Angular motion equation
        J = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))
        
        # Kinematic Equations (geodetic frame)
        # Translation: Aircraft position derivative
        r_dot = quat.transform(q, v)
        
        # Rotation: Aircraft attitude derivative
        # Quaternion representation
        LAMBDA_C = -5;
        q_dot = 0.5 * quat.mult(q, np.concatenate(([0], w))) +\
            0.5 * LAMBDA_C * q * (np.dot(q, q) - 1)        # Quaternion norm stabilization term,
        # as in Gros: 'Baumgarte Stabilisation over the SO(3) Rotation Group for Control',
        # improved: lambda negative and SX::dot(q, q) instead of lambda positive and 1/SX::dot(q, q).
        

        state_dot = np.concatenate((v_dot, r_dot, q_dot))
        
        return state_dot