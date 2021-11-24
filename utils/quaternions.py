import numpy as np
from utils.smooth import Smooth as sm

class Quaternion():
    
    @staticmethod
    def inverse(q) -> np.array:
        """Inverses a quaternion

        Args:
            q (np.array): The quaternion to inverse

        Returns:
            np.array: inverted quaternion
        """
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    @staticmethod
    def mult(q1, q2) -> np.array:
        """Multiplies two quaternions q1 q2

        Args:
            q1 (np.array): quat 1
            q2 (np.array): quat 2

        Returns:
            np.array: result quaternion
        """
        s1 = q1[0]
        v1 = q1[1:4]

        s2 = q2[0]
        v2 = q2[1:4]

        s = (s1 * s2) - np.dot(v1, v2)
        v = np.cross(v1, v2) + (s1 * v2) + (s2 * v1);

        return np.concatenate((np.array([s]),v))

    
    @staticmethod
    def to_euler(q) -> np.array:
        """Generates Euler angles from an orientation quaternion

        Args:
            q (np.array): orientation quaternion

        Returns:
            np.array: Euler angles
        """
        r11 = 2 * (q[1] * q[2] + q[0] * q[3] )
        r12 = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
        r21 = -2*(q[1]*q[3] - q[0]*q[2])
        r31 = 2.*(q[2]*q[3] + q[0]*q[1])
        r32 = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2


        r1 = np.atan2( r11, r12 );
        
        r21 = sm.sigmoid((r21+1)*10) - sm.sigmoid((r21-1)*10);
    
        r2 = np.asin( r21 )
        r3 = np.atan2( r31, r32 )
        
        return np.array([r1,r2,r3])

    def transform(q_ba,a_vect) -> np.array:
        """Transforms a vector according to a rotation quaternion

        Args:
            q_ba (np.array): rotation quaternion
            a_vect (np.array): input vector

        Returns:
            np.array: transformed vector
        """
        temp = Quaternion.mult(\
            q_ba, Quaternion.mult(np.concatenate((np.array([1.],dtype=float),a_vect)),\
            Quaternion.inverse(q_ba)))
        return temp[1:4]

    
    def t1(rot_angle) -> np.array:
        """rotation around the x axis

        Args:
            rot_angle (np.array): rotation angle

        Returns:
            np.array: rotation quaternion
        """
        return np.array([np.cos(-rot_angle / 2.0), np.sin(-rot_angle / 2.0), 0, 0]);
    
    def t2(rot_angle) -> np.array:
        """rotation around the y axis

        Args:
            rot_angle (np.array): rotation angle

        Returns:
            np.array: rotation quaternion
        """
        return np.array([np.cos(-rot_angle / 2.0), 0, np.sin(-rot_angle / 2.0), 0]);
    
    def t3(rot_angle) -> np.array:
        """rotation around the z axis

        Args:
            rot_angle (np.array): rotation angle

        Returns:
            np.array: rotation quaternion
        """
        return np.array([np.cos(-rot_angle / 2.0), 0, 0, np.sin(-rot_angle / 2.0)]);