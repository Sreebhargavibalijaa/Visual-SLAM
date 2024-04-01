import numpy as np
from pr3_utils import *


if __name__ == '__main__':

	# Load the measurements
	filename = "/Users/sreebhargavibalija/Downloads/ECE-276A-master/PR3/code/data/10.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)

