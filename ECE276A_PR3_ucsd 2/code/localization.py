
"""
a) Implement IMU localization based on SE(3)\kinematics using the linear and angular velocity measurements.
"""
import numpy as np
from pr3_utils import *
from scipy.linalg import expm

data = load_data('/Users/sreebhargavibalija/Documents/ECE276A_PR3_ucsd 2/data/10.npz')
pose = np.eye(4)
pose_toplot = pose
ang_vel, lin_vel, _, timestamp, _, _, _ = parse_data(data)
prev_t = timestamp[0,0]

for index in range(1, lin_vel.shape[1]):
    omega = ang_vel[:, index]
    velocity = lin_vel[:, index]
    twist_matrix = np.array([[0, -omega[2], omega[1], velocity[0]],
                             [omega[2], 0, -omega[0], velocity[1]],
                             [-omega[1], omega[0], 0, velocity[2]],
                             [0, 0, 0, 0]])
    
    pose = pose @ expm(twist_matrix * (timestamp[0, index] - prev_t))
    prev_t = timestamp[0, index]
    pose_toplot = np.dstack((pose_toplot, pose))

visualize_trajectory_2d_imu(pose_toplot)

