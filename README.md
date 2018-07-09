Orientation Estimation using Unscented Kalman Filter
====================================================
In my implementation, I take the logistic from the original project. However, I focused on only orientation tracking to get a sense of how unscented kalman filter works, which helps me to implement a UKF for trakcing 3D position, velocity, and orientation by fusing IMU and GPS data.


Differences from orignal project
--------------------------------
* Tested on Mac with Python 3.6.5
* Used a library called `quat_helper.py` for quaternion operations
* Employed the framework of the `filterpy.kalman UKF`
* Used an easy way to get the mean of a set of quaternions, see `quaternion_mean` in `quat_helper.py`

**One thing to note** is that there are two conventions on quaternion product/multiplication/composition: Hamilton and JPL (see ['Why and How to Avoid the Flipped Quaternion Multiplication'](https://arxiv.org/abs/1801.07478)). In my implementation, I used Hamiton convention, which is commonly used (`qb = qmult(dq, qa)`).






====== Below is the original readme file ======
===============================================







Orientation Estimation using Unscented Kalman Filter
====================================================

In this project, I have implemented Unscented Kalman filter to track three dimensional orientation. This means to estimate the underlying 3D orientation by learning the appropriate model parameters from ground truth data given by a Vicon motion capture system, given IMU sensor readings from gyroscopes and accelerometers. Then be able to generate a real-time panoramic image from camera images using the 3D orientation filter.

* Tested on: Kubuntu 16.04.3, Intel i5-4200U (4) @ 2.600GHz 4GB
* Python 2.7, OpenCV 3.2

Challenge Description
---------------------
* The first part of the problem was to calculate bias and scale parameters for the accelerometer and gyroscope readings.
* Convert IMU readings to quaternions.
* Implement UKF.
* Perform Image Stitching.

# Unscented Kalman filter

### Process Model
* The UKF implementation was done using only orientation(gyroscope) in the state vector as the control input: q = [q<sub>0</sub>, q<sub>1</sub>, q<sub>2</sub>, q<sub>3</sub>]<sup>T</sup> . 
* Initialize P (Covariance matrix) as size of 3x3. Similarly, R and Q. R is measurement noise and Q is process noise.
* After Kalman filter predict step, new P and state vector q are obtained, which are the used for update step.
* Then Sigma Points are obtained by Cholesky decomposition of (P+Q).

### Motion Model
* This step deals with updating P and getting new mean state q. Which then leads to obtaining new Sigma Points. This new sigma points are used to calculate multiple covariances, like Pzz, Pxz, and Pvv. 
* The next step involves computing K (Kalman Gain) = Pxz Pvv-1 and I (Innovation term) = Accelerometer reading – Mean of Sigma Points
* These are used to calculate the P and q for the next stage.

# Results

| Roll-Pitch-Yaw | Vicon Stitch | Estimated Orientation Stitch |
| ----------- | ----------- | ----------- |
| <img src="Results/RPYn1.png" width="1000"> | <img src="Results/Pano1Vicon.jpg" width="1000"> | <img src="Results/Pano1IMU.jpg" width="1000"> |
| <img src="Results/RPYn8.png" width="1000"> | <img src="Results/Pano8Vicon.jpg" width="1000"> | <img src="Results/Pano8IMU.jpg" width="1000"> |

| Test Dataset Roll-Pitch-Yaw | Estimated Orientation Stitch |
| ----------- | ----------- |
| <img src="Results/RPYwoVicon11.png" width="1000">  | <img src="Results/Pano11.jpg" width="1000"> |

### Stitching in action
| Vicon Stitch | Estimated Orientation Stitch |
| ----------- | ----------- |
| [![](Results/Pano8Vicon.jpg)](https://youtu.be/p9OQAM88Epc) | [![](Results/Pano8IMU.jpg)](https://youtu.be/p30TAvp4XgA) |


Instructions
------------
Download:

[Training Dataset](https://upenn.app.box.com/v/ese650Proj2-train)

[Test Dataset](https://upenn.app.box.com/v/ese650Proj2-test)

Place the cam data in "cam" folder, vicon data in "vicon" folder and imu data in "imu" folder.

* Run ukf.py to calculate the predicted values for q state vectors.
	* Outputs of Roll, Pitch and Yaw for predicted vs vicon data will be in folder --> 'Results' with RPY prefix.
	* Outputs of stitching will be in folder --> 'Results' with 'Pano' prefix. 

	NOTE - All variables to toggle stitching, change dataset and change stitching medium(vicon/imu) are in 'ukf.py' at beginning.

* quat_helper.py contains neccessarry functions for quaternion state vector manipulation.

# References
* [A quaternion-based unscented Kalman filter for orientation tracking](http://ieeexplore.ieee.org/document/1257247/)
* [The Kalman Filter](http://www.stat.columbia.edu/~liam/teaching/neurostat-spr12/papers/hmm/KF-welling-notes.pdf)
