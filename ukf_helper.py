import numpy as np

import quat_helper

#######################################################################
#######################################################################

scale_factor = 1


class UKF():
    def __init__(self, x_dim, z_dim, P_dim, x=None, P=None, Q=None, R=None):
        self.x = x
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.P = P
        self.P_dim = P_dim
        self.Q = Q
        self.R = R

        self.num_sigma = 2*P_dim

    def predict(self, u, dt):
        qu = quat_helper.expq((u*dt)/2)

        L = np.linalg.cholesky(self.P+self.Q)
        new_vec = np.hstack(
            (L*np.sqrt(2*self.P_dim), -L*np.sqrt(2*self.P_dim)))
        new_vec = np.transpose(new_vec)

        # Sigma points obtained by disturbing the current state (quaternions)
        self.sigma_points = quat_helper.qmult(
            quat_helper.expq(new_vec/2), self.x)

        # Unscented transform
        # get transformed sigma points - motion_sig,
        # calulate the mean on the points - self.x_predicted
        # then evaluate the errors of transformed points from the mean - self.x_residuals

        # TODO The order of quaternion multiplication/composition matters
        self.sigma_f = quat_helper.qmult(self.sigma_points, qu)
        # self.sigma_f = quat_helper.qmult(qu, sigma_points)

        self.x_predicted = quat_helper.quaternion_mean(self.sigma_f)

        self.x_residuals = 2 * \
            quat_helper.logq(quat_helper.qmult(
                self.sigma_f, quat_helper.qinv(self.x_predicted)))

        self.P_predicted = np.zeros(np.shape(self.P))
        for r in self.x_residuals:
            self.P_predicted += np.outer(r, r)

        self.P_predicted /= self.num_sigma*scale_factor

    def update(self, z, g):

        # use accelerometer data to update/correct the prediction
        h = quat_helper.qmult(quat_helper.qmult(
            quat_helper.qinv(self.sigma_f), g), self.sigma_f)

        self.sigma_h = h[:, 1:]

        self.z_mean = np.average(self.sigma_h, axis=0)

        z_residuals = self.sigma_h - self.z_mean

        Pzz = np.zeros(np.shape(self.P))
        for r in z_residuals:
            Pzz += np.outer(r, r)
        Pzz /= self.num_sigma*scale_factor

        Pvv = Pzz + self.R

        Pxz = np.zeros([self.P_dim, self.z_dim])
        for i in range(self.num_sigma):
            Pxz += np.outer(self.x_residuals[i], z_residuals[i])
        Pxz /= self.num_sigma*scale_factor

        K = np.dot(Pxz, np.linalg.inv(Pvv))
        I = np.transpose(z - self.z_mean)
        KI = quat_helper.expq(np.dot(K, I)/2)

        self.x = quat_helper.qmult(KI, self.x_predicted)

        TT = np.dot(np.dot(K, Pvv), np.transpose(K))

        self.P = self.P_predicted - TT
