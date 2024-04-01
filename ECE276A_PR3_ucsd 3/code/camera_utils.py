

import numpy as np
from pr3_utils import Homogenize, Dehomogenize, hat

def landmarks_to_camera_coordinates(landmarks_world, world_to_imu_transform, imu_to_camera_transform, camera_matrix):
    '''
    Converts landmark coordinates from world frame to pixel coordinates in the left camera image.
    '''
    assert landmarks_world.shape[0] == 3, 'Expected a 3xN array of world points'
    camera_coords = np.linalg.inv(world_to_imu_transform @ imu_to_camera_transform) @ Homogenize(landmarks_world)
    
    image_coords = camera_matrix @ camera_coords / camera_coords[2, :]
    return image_coords

def derivative_of_projection(q):
    '''
    Calculates the derivative of the projection function at a point q.
    '''
    q1, q2, q3, q4 = q
    
    derivative = np.array([[1, 0, -q1 / q3, 0],
                           [0, 1, -q2 / q3, 0],
                           [0, 0, 0, 0],
                           [0, 0, -q4 / q3, 1]]) / q3
    
    return derivative

def camera_to_world_coordinates(features, world_to_imu_transform, imu_to_camera_transform, camera_intrinsics, baseline):
    '''
    Transforms feature coordinates from image frame back to world coordinates.
    '''
    depth = camera_intrinsics[0, 0] * baseline / (features[0, :] - features[2, :])
    y_coordinates = depth * (features[1, :] - camera_intrinsics[1, 2]) / camera_intrinsics[1, 1]
    x_coordinates = depth * (features[0, :] - camera_intrinsics[0, 2]) / camera_intrinsics[0, 0]
    
    camera_coords_homogenous = Homogenize(np.vstack((x_coordinates, y_coordinates, depth)))
    imu_coords = imu_to_camera_transform @ camera_coords_homogenous
    world_coords = Dehomogenize(world_to_imu_transform @ imu_coords)
    return world_coords

def skew_symmetric_cross_operator(s):
    '''
    Computes the skew symmetric matrix (cross product operator) for the Jacobian of the observation model with respect to map coordinates.
    '''
    assert s.shape[0] == 4, 'Expected 3D homogeneous coordinates'
    s_normalized = s / s[-1]
    
    return np.vstack((np.hstack((np.eye(3), -hat(s[:3]))), np.zeros((1, 6))))
