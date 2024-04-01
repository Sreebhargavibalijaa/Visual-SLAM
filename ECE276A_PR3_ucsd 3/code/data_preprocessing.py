import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from pr3_utils import *
from tqdm.notebook import tqdm
import scipy
DATASET = "03"
DATASET = "03"
# Load the measurements
filename = f'/Users/sreebhargavibalija/Documents/ECE276A_PR3_ucsd 2/data/03.npz'
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
t = np.squeeze(t,0)

features = features[:,::2,:]
fsu, fsth, cu, fsv, cv = K[0,0], K[0,1], K[0,2], K[1,1], K[1,2]

Ks = np.array([[fsu,0,cu,0],
               [0,fsv,cv,0],
               [fsu,0,cu,-fsu*b],
               [0,fsv,cv,0]])

np.tile(t,(3,1)).shape

plt.plot(np.tile(t,(3,1)).T,linear_velocity.T)
plt.title(f'Linear Velocity (vt), Dataset {DATASET}')
plt.legend(['x','y','z'])
plt.show()
plt.plot(np.tile(t,(3,1)).T,angular_velocity.T)
plt.title(f'Angular Velocity (wt), Dataset {DATASET}')
plt.legend(['x','y','z'])
plt.show()
DATASET = "10"
# Load the measurements
filename = f'/Users/sreebhargavibalija/Documents/ECE276A_PR3_ucsd 2/data/10.npz'
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
t = np.squeeze(t,0)

features = features[:,::2,:]
fsu, fsth, cu, fsv, cv = K[0,0], K[0,1], K[0,2], K[1,1], K[1,2]

Ks = np.array([[fsu,0,cu,0],
               [0,fsv,cv,0],
               [fsu,0,cu,-fsu*b],
               [0,fsv,cv,0]])

np.tile(t,(3,1)).shape

plt.plot(np.tile(t,(3,1)).T,linear_velocity.T)
plt.title(f'Linear Velocity (vt), Dataset {DATASET}')
plt.legend(['x','y','z'])
plt.show()

plt.plot(np.tile(t,(3,1)).T,angular_velocity.T)
plt.title(f'Angular Velocity (wt), Dataset {DATASET}')
plt.legend(['x','y','z'])
plt.show()
