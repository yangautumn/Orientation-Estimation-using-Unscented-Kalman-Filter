#!/usr/bin/env python3.6

# Reference to https://github.com/yashv28/Orientation-Estimation-using-Unscented-Kalman-Filter
# I am trying to understand how mean and covariance of quaternions are
# done in this code, and I am also trying to fractor the code thereafter.

# Author: Yang Li
# Date: June 28, 2018
# Email: yangautumn43@gmail.com

import sys
import os

from scipy import io
import numpy as np
from math import pi
import pickle

import matplotlib.pyplot as plt
import cv2

from quat_helper import *
from ukf_helper import *

########################################################################
# Data Load and timestamp match
########################################################################


def run_ukf_on_dataset(dataset):

    Dataset = dataset

    imu = io.loadmat("imu/imuRaw"+str(Dataset)+".mat")
    imu_vals = np.transpose(imu['vals'])
    imu_ts = np.transpose(imu['ts']).flatten()

    # for testing, get part of imu data
    start, end = 0, -1
    imu_vals = imu_vals[start: end]
    imu_ts = imu_ts[start: end]

    # print('imu_vals shape:\n', np.shape(imu_vals))
    # print('part of imu_ts array:\n', imu_ts[0:5])
    # for time in imu_ts[0:10]:
    #     print(f'{time:f}')

    Vref = 3300

    # get accelerameter data and change directions of axis
    acc = imu_vals[:, 0:3] * np.array([-1, -1, 1])

    acc_sensitivity = 330.0
    acc_scale_factor = Vref/1023.0/acc_sensitivity
    acc_bias = acc[0] - (np.array([0, 0, 1])/acc_scale_factor)
    acc_val = acc * acc_scale_factor
    acc_val = acc_val - acc_bias * acc_scale_factor

    # print('part of acc_val array:\n', acc_val[0:5])

    # plt.plot(acc_val[:, 0], 'r', label='Accelerometer-x')
    # plt.plot(acc_val[:, 1], 'g', label='Accelerometer-y')
    # plt.plot(acc_val[:, 2], 'b', label='Accelerometer-z')
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #            ncol=3, mode="expand", borderaxespad=0.)
    # plt.show()

    # get gyro data and rotate the frame
    gyro = imu_vals[:, [4, 5, 3]]
    # gyro = imu_vals[:, [3, 4, 5]]     # test a wrong rotation

    print('gyro shape:\n', np.shape(gyro), type(gyro))

    gyro_bias = gyro[0]
    gyro_sensitivity = 3.33
    gyro_scale_factor = Vref/1023/gyro_sensitivity
    gyro_val = gyro * gyro_scale_factor
    gyro_val = (gyro_val - gyro_bias * gyro_scale_factor) * (pi/180)

    # print('part of gyro_val array;\n', gyro_val[0:5])

    # plt.plot(gyro_val[:, 0], 'r', label='Gyro-x')
    # plt.plot(gyro_val[:, 1], 'g', label='Gyro-x')
    # plt.plot(gyro_val[:, 2], 'b', label='Gyro-x')
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #            ncol=3, mode="expand", borderaxespad=0.)
    # plt.show()

    ########################################################################
    # Estimate the underlying 3D orientation by learning the appropriate
    # model parameters from ground truth data given by a Vicon motion capture
    # system, given IMU sensor readings from gyroscopes and accelerometers.

    # Challenge Description
    #   - The first part of the problem was to calculate bias and scale parameters for the accelerometer and gyroscope readings.
    #   - Convert IMU readings to quaternions.
    #   - Implement UKF.
    #   - Perform Image Stitching.
    ########################################################################

    '''
    Unscented Kalman filter

    Process Model - Predict
    - The UKF implementation was done using only orientation(gyroscope) in
        the state vector as the control input: q = [q0, q1, q2, q3]T .
    - Initialize P(Covariance matrix) as size of 3x3. Similarly, R and Q.
        R is measurement noise and Q is process noise.
    - After Kalman filter predict step, new P and state vector q are obtained,
        which are the used for update step.
    - Then Sigma Points are obtained by Cholesky decomposition of (P+Q).

    Motion Model - Update
    - This step deals with updating P and getting new mean state q.
        Which then leads to obtaining new Sigma Points. This new sigma points
        are used to calculate multiple covariances, like Pzz, Pxz, and Pvv.
    - The next step involves computing K(Kalman Gain) = Pxz Pvv-1 and
        I(Innovation term) = Accelerometer reading – Mean of Sigma Points
    - These are used to calculate the P and q for the next stage.'''

    P = np.identity(3)*1e-5          # Covariance
    Q = np.identity(3)*1e-5          # Process noise
    R = np.identity(3)*5e-3          # Measurement noise
    # The initial orientation as a quaternion
    q0 = np.array([1, 0, 0, 0])
    ut = gyro_val[0]

    # TODO what is g for ?
    # Used in measurement model
    g = np.array([0, 0, 0, 1])

    t = imu_ts.shape[0]
    R_calc = np.zeros((3, 3, np.shape(gyro_val)[0]))

    ukf = UKF(x_dim=4, z_dim=3, P_dim=3, x=q0, P=P, Q=Q, R=R)

    # UKF
    # if not os.path.exists('Parameters/param'+str(Dataset)+'.pickle'):
    # Read in gyro data one by one and estimate with unscented kalman filter
    for i in range(0, np.shape(gyro_val)[0]):

        print(f'Running the {i:d}th data', end='\r')

        if i == 0:
            ukf.predict(u=gyro_val[i], dt=imu_ts[0])
            predicted_q = q0
        else:
            ukf.predict(u=gyro_val[i], dt=imu_ts[i]-imu_ts[i-1])

        # print('predicted state\n', ukf.x_predicted)
        # print('predicted covariance\n', ukf.P_predicted)

        ukf.update(acc_val[i], g)

        # print('updated state\n', ukf.x)
        # print('updated covariance\n', ukf.P)

        predicted_q = np.vstack((predicted_q, ukf.x))
        R_calc[:, :, i] = geom.quaternion_to_matrix(ukf.x)

    # predicted_q = np.matrix(predicted_q)

    #
    roll = np.zeros(np.shape(predicted_q)[0])
    pitch = np.zeros(np.shape(predicted_q)[0])
    yaw = np.zeros(np.shape(predicted_q)[0])

    for i in range(np.shape(predicted_q)[0]):
        yaw[i], pitch[i], roll[i] = geom.quaternion_to_ypr(predicted_q[i])

    # Compare with ground-truth data
    if os.path.exists("vicon/viconRot"+str(Dataset)+".mat"):
        vicon = io.loadmat("vicon/viconRot"+str(Dataset)+".mat")
        vicon_vals = np.array(vicon['rots'])
        vicon_ts = np.array(vicon['ts']).flatten()

        # vicon_vals = vicon_vals[:, :, 20:-1]

        print()
        # print(vicon_ts.shape)
        # for time in vicon_ts[0:20]:
        #     print(f'{time:f}')

        vicon_vals = vicon_vals[:, :, start: end]
        print('vicon_vals shape:\n', np.shape(vicon_vals))

        num = np.shape(vicon_vals)[2]
        vicon_roll = np.zeros(num)
        vicon_pitch = np.zeros(num)
        vicon_yaw = np.zeros(num)
        for i in range(num):
            R = vicon_vals[:, :, i]
            vicon_yaw[i], vicon_pitch[i], vicon_roll[i] = geom.matrix_to_ypr(R)

        plt.figure()
        plt.subplot(311)
        plt.plot(vicon_roll, 'b', label='Ground truth')
        plt.plot(roll, 'r', label='UKF estimated')
        plt.ylabel('Roll')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        plt.subplot(312)
        plt.plot(vicon_pitch, 'b', pitch, 'r')
        plt.ylabel('Pitch')
        plt.subplot(313)
        plt.plot(vicon_yaw, 'b', yaw, 'r')
        plt.ylabel('Yaw')
        # plt.savefig('Results/RPY'+str(Dataset)+'.png')
        plt.show()
