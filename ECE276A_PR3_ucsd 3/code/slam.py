
import numpy as np
from pr3_utils import *
from camera_utils import *
from scipy.linalg import expm
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler

def load_data(file_name):
    '''
    function to read visual features, IMU measurements and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npz"
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images, 
            with shape 4*n*t, where n is number of features
        linear_velocity: velocity measurements in IMU frame
            with shape 3*t
        angular_velocity: angular velocity measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        imu_T_cam: extrinsic matrix from (left)camera to imu, in SE(3).
            with shape 4*4
    '''
    with np.load(file_name) as data:
    
        t = data["time_stamps"] # time_stamps
        features = data["features"] # 4 x num_features : pixel coordinates of features
        linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
        angular_velocity = data["angular_velocity"] # angular velocity measured in the body frame
        K = data["K"] # intrindic calibration matrix
        b = data["b"] # baseline
        imu_T_cam = data["imu_T_cam"] # Transformation from left camera to imu frame 
    
    return t,features,linear_velocity,angular_velocity,K,b,imu_T_cam

def parse_data(data):
    '''
    Utility function to parse the npy file
    '''
    ang_vel = data[3]
    lin_vel = data[2]
    features = data[1]
    TS = data[0]
    imu_T_cam = data[-1]
    K = data[-3]
    b = data[-2]
    return ang_vel, lin_vel, features, TS, imu_T_cam, K, b

def load_data_from_dir(dir_name):
    '''
    function to read visual features, IMU measurements and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npz"
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images, 
            with shape 4*n*t, where n is number of features
        linear_velocity: velocity measurements in IMU frame
            with shape 3*t
        angular_velocity: angular velocity measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        imu_T_cam: extrinsic matrix from (left)camera to imu, in SE(3).
            with shape 4*4
    '''
    
    data = {}
    
    file_list = os.listdir(dir_name)
    for file in file_list:
        print(os.path.join(dir_name, file))
        data[file[:-4]] = np.load(os.path.join(dir_name, file))
    
    return data


def visualize_trajectory_2d(pose, mean = None, path_name="Path update after EKF",show_ori=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
    if mean is not None:
        ax.scatter(mean[0,:],mean[1,:],s = 2,label="landmark")
  
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.savefig('test.jpeg')
    plt.show(block=True)

    return fig, ax


import numpy as np
# from pr3_utils import Homogenize, Dehomogenize, hat



def translate_to_camera(x_w, w_T_imu, imu_T_cam, Ks):
    '''
    Translates the landmark coordinates in world frame to the
    left image pixel coordinates
    '''
# Validate the input shape for world coordinates
    assert x_w.shape[0] == 3, 'Expected an array of 3xN world points'
    # Transform from world coordinates to camera coordinates
    camera_coordinates = np.linalg.inv(w_T_imu @ imu_T_cam) @ Homogenize(x_w)

    # Project camera coordinates to image pixels
    pixel_coordinates = Ks @ (camera_coordinates / camera_coordinates[2, :])
    return pixel_coordinates
def compute_jacobian(camera_matrix, world_to_imu, imu_to_cam, world_points, idxs):
    '''
    Function to compute the Jacobian of the observation model wrt landmark points
    '''
    num_obs = len(idxs)
    num_points = world_points.shape[1]
    imu_to_world = np.linalg.inv(world_to_imu @ imu_to_cam)
    points_in_cam = imu_to_world @ Homogenize(world_points)
    jacobian_matrix = np.zeros((4*num_obs,3*num_points))
    cam_projection = np.hstack((np.eye(3), np.zeros((3,1))))
    for i in range(num_obs):
        landmark_idx = idxs[i]
        jacobian_matrix[4*i:4*i+4,3*landmark_idx:3*landmark_idx+3] = camera_matrix @ projection_function(points_in_cam[:,landmark_idx]) @ imu_to_world @ cam_projection.T
    return jacobian_matrix


def projection_function(q):
    '''
    Compute the derivative of the projection function
    at q
    '''
# Unpack the components of vector q
    q1, q2, q3, q4 = q

    # Compute the derivative matrix
    derivative_matrix = np.array([
        [1, 0, -q1 / q3, 0],
        [0, 1, -q2 / q3, 0],
        [0, 0, 0, 0],
        [0, 0, -q4 / q3, 1]
    ]) / q3

    return derivative_matrix

def translate_to_world(features, w_T_imu, imu_T_cam, K, b):
    '''
    Utility function translates features in image coordinate to 
    the world frame.
    '''
# Calculate depth from stereo disparity
    depth = (K[0,0] * b) / (features[0, :] - features[2, :])
    # Determine Y and X coordinates from depth and intrinsic parameters
    y_coord = depth * (features[1, :] - K[1,2]) / K[1,1]
    x_coord = depth * (features[0, :] - K[0,2]) / K[0,0]

    # Convert the coordinates into homogeneous form
    homogeneous_camera_coords = Homogenize(np.vstack((x_coord, y_coord, depth)))
    # Transform coordinates from camera to IMU frame
    imu_frame_coords = imu_T_cam @ homogeneous_camera_coords
    # Transform coordinates from IMU to world frame and dehomogenize
    world_frame_coords = Dehomogenize(w_T_imu @ imu_frame_coords)
    return world_frame_coords

def observation_model(s):
    '''
    Utility function to compute the dot operator for jacobian of observation model wrt map co-ord
    '''
    # Ensure the input is in 4D homogeneous format
    assert s.shape[0] == 4, 'Expected 4D homogeneous coordinates'
    # Normalize the vector
    normalized_s = s / s[-1]

    # Construct the matrix with skew-symmetric part for normalized vector
    transform_matrix = np.vstack((
        np.hstack((np.eye(3), -hat(normalized_s[:3]))),
        np.zeros((1, 6))
    ))
    return transform_matrix

def visualize_trajectory_2d_imu(pose, mean = None, path_name="IMU by EKF",show_ori=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
    if mean is not None:
        ax.scatter(mean[0,:],mean[1,:],s = 2,label="landmark")
  
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.savefig('test.jpeg')
    plt.show(block=True)

    return fig, ax

def get_features_current_frame(features, idx, K=10):
    '''
    Retrieves features from the current frame at the given time index.
    Returns a 3xM array where invalid points are marked with NaN.
    '''
    current_features = features[:, :, idx]
    
    # Assign NaN to invalid feature points
    current_features[:, current_features[0, :] < 0] = np.nan
    # Downsample the features by selecting every Kth point
    downsampled_features = current_features[:, : -1 : K]
    valid_features_indices = np.where(~np.isnan(downsampled_features[0, :]))
    
    return downsampled_features, valid_features_indices[0].tolist()

def get_idx(listA, list_global):
    '''
    Identifies indices in listA that are also in list_global,
    and indices that are unique to listA.
    '''
    existing_indices = []
    new_indices = []
    for element in listA:
        if element in list_global:
            existing_indices.append(element)
        else:
            new_indices.append(element)
    
    return existing_indices, new_indices

def hat(vector):
    '''
    Generates a skew-symmetric matrix from a 3-dimensional vector.
    '''
    assert vector.shape[0] == 3, 'Expected a 3D vector for skew-symmetric conversion.'
    
    skew_matrix = np.array([[0, -vector[2], vector[1]],
                            [vector[2], 0, -vector[0]],
                            [-vector[1], vector[0], 0]])
    
    return skew_matrix

def hat_alt(vector):
    '''
    Creates an extended skew-symmetric matrix from a 6-dimensional vector.
    '''
    assert vector.shape[0] == 6, 'Expected a 6D vector for extended skew-symmetric conversion.'
    
    primary_skew = hat(vector[:3])
    secondary_skew = hat(vector[3:])
    
    combined_matrix = np.vstack((
        np.hstack((secondary_skew, primary_skew)),
        np.hstack((np.zeros((3, 3)), secondary_skew))
    ))
    return combined_matrix
def jacob(camera_matrix, world_to_imu, imu_to_cam, world_points, idxs):
    '''
    Function to compute the Jacobian of the observation model wrt landmark points
    '''
    num_obs = len(idxs)
    num_points = world_points.shape[1]
    imu_to_world = np.linalg.inv(world_to_imu @ imu_to_cam)
    points_in_cam = imu_to_world @ Homogenize(world_points)
    jacobian_matrix = np.zeros((4*num_obs,3*num_points))
    cam_projection = np.hstack((np.eye(3), np.zeros((3,1))))
    for i in range(num_obs):
        landmark_idx = idxs[i]
        jacobian_matrix[4*i:4*i+4,3*landmark_idx:3*landmark_idx+3] = camera_matrix @ projection_function(points_in_cam[:,landmark_idx]) @ imu_to_world @ cam_projection.T
    return jacobian_matrix

def twist_matrix(vector):
    '''
    Constructs a twist matrix from a 6-dimensional vector.
    '''
    assert vector.shape[0] == 6, 'Expected a 6D vector for twist matrix creation.'
    twist_mat = np.vstack((
        np.column_stack((hat(vector[3:]), vector[:3])),
        np.zeros((1, 4))
    ))
    return twist_mat

def Homogenize(points):
    '''
    Converts points from inhomogeneous to homogeneous coordinates.
    '''
    return np.vstack((points, np.ones((1, points.shape[1]))))

def Dehomogenize(points):
    '''
    Converts points from homogeneous back to inhomogeneous coordinates.
    '''
    return points[:-1] / points[-1]

if __name__ == "__main__":
  print("proceeding")
    
data = load_data('/Users/sreebhargavibalija/Desktop/ECE-276A/PR3/data/03.npz')
feature_skip = 15

# Read Data
ang_vel, lin_vel, features, TS, imu_T_cam, K, b = parse_data(data)

# Stereo camera matrix
Ks = np.vstack((K[:2,:],K[:2,:]))
Ks = np.hstack((Ks, np.zeros((4,1))))
Ks[2,-1] = - Ks[0,0]*b

# init pose 
mean_pose = expm(twist_matrix(np.array([0,0,0,np.pi,0,0]))) #np.eye(4)

# Cache last time stamp
prev_TS = TS[0,0]

# Placeholder to store pose of the robot
pose_toplot = mean_pose

# Read features
curr_features,detected_LM_idx = get_features_current_frame(features, 0, feature_skip)

# Translate image features to world coordinates
x_w_landmark = translate_to_world(curr_features, mean_pose, imu_T_cam, K, b)

# Number of landmarks
M = curr_features.shape[1]

# Init mean and cov
mean_L = np.ones((3,M))*np.nan
cov = np.zeros((3*M+6,3*M+6))

mean_L[:,detected_LM_idx] = x_w_landmark[:,detected_LM_idx]
cov[-6:,-6:] = np.eye(6)
for i in range(len(detected_LM_idx)):
    idx = detected_LM_idx[i]
    cov[3*idx:3*idx+3,3*idx:3*idx+3] = np.eye(3)
    
# Initalize noises
# Measurement noise
V = np.eye(4)*10

# Motion noise
W = np.eye(6)
W[0:3,0:3] = W[0:3,0:3]*1#0.3
W[3:6,3:6] = W[3:6,3:6]*0.3#0.05

###### Loop over time #######
for fIdx in tqdm(range(1,lin_vel.shape[1])):
    
    # Manage time
    tau = TS[0,fIdx]-prev_TS
    prev_TS = TS[0,fIdx]
    
    w = ang_vel[:,fIdx]
    w[-1] = -w[-1]
    v = lin_vel[:,fIdx]
     
    # Get twist matix for motion model
    tm = twist_matrix(np.hstack((v,w)))
    
    # Mean pose predict
    mean_pose = mean_pose @ expm(tm*tau)    
    
    # Cov pose predict
    F = expm(-tau*hat_alt(np.hstack((v,w))))
    cov[-6:,-6:] = F @ cov[-6:,-6:] @ F.T + W
        
    # Featch image features
    curr_features,valid_landmarks = get_features_current_frame(features, fIdx, feature_skip)
    
    # Get feature idx to update and add
    idx_toUpdate, idx_toAdd = get_idx(valid_landmarks, detected_LM_idx)
    
    # Update
    if len(idx_toUpdate) > 0:
        # Compute Jacobian
        H = jacob(Ks, mean_pose, imu_T_cam, mean_L, idx_toUpdate)

        # Compute Kalman Gain
        KG = cov @ H.T @ np.linalg.inv(H @ cov @ H.T + np.kron(np.eye(len(idx_toUpdate)),V))
        
        # Mean update
        res = curr_features[:, idx_toUpdate] - translate_to_camera(mean_L[:,idx_toUpdate], mean_pose, imu_T_cam, Ks)
        delta_mean = KG @  res.ravel('F')
        
        # landmark pos update
        delta_mean_L = delta_mean[:-6].reshape(3,-1, order = 'F')
        mean_L[:,idx_toUpdate] = mean_L[:,idx_toUpdate] + delta_mean_L[:,idx_toUpdate]
        
        # mean pose update
        delta_mean_pose = delta_mean[-6:]
        mean_pose = mean_pose @ expm(twist_matrix(delta_mean_pose))

        t1 = (np.eye(3*M + 6) - KG @ H)
        cov = t1 @ cov @ t1.T + KG @ np.kron(np.eye(len(idx_toUpdate)),V) @ KG.T

    if len(idx_toAdd) > 0:
        x_w_landmark = translate_to_world(curr_features, mean_pose, imu_T_cam, K, b)
        mean_L[:,idx_toAdd] = x_w_landmark[:,idx_toAdd]
        for j in range(len(idx_toAdd)):
            idx = idx_toAdd[j]
            cov[3*idx:3*idx+3,3*idx:3*idx+3] = np.eye(3)
            detected_LM_idx.append(idx)
            
    # Save pose 
    pose_toplot = np.dstack((pose_toplot, mean_pose))
    
visualize_trajectory_2d_slam(pose_toplot, mean_L)

