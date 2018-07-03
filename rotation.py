"""Collection of utilities for dealing with the mathematics of spatial
rotations."""

import numpy as np
from math import sin, cos, pi


def quaternion_mean(qs, weights=None):
    """Compute the weighted mean over unit quaternions as described in
    "Averaging Quaternions" by F.L. Markley et al., Journal of Guidance,
    Control, and Dynamics 30, no. 4 (2007): 1193-1197.

    Parameters
    ----------
    qs : ndarray - shape (N, 4)
        Array of unit quaternions
    weights : ndarray - shape (N,) or (1, N)
        Optional weight factors for each quaternion

    Returns
    -------
    Normalized orientation quaternion as ndarray with shape (4,).
    """
    if weights is not None:
        w = np.atleast_2d(weights).T  # w should be Nx1
        Q = (w * qs).T
    else:
        Q = qs.T

    vals, vecs = np.linalg.eig(Q @ Q.T)
    return np.real_if_close(vecs[:, 0])


def quaternion_distance(a, b):
    """Returns distances between arrays of normalized quaternions.

    Accounts for 2-fold degeneracy (q and -q perform the same rotation).

    Parameters
    ----------
    a, b : ndarray - shape (4,) or (N, 4)
        Quaternions in N rows

    Returns
    -------
    scalar or ndarray with shape (N,)
        Distance(s) between a and b.
    """
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    d2 = np.minimum(np.sum((a - b)**2, axis=1), np.sum((a + b)**2, axis=1))
    d = np.sqrt(d2)
    if d.shape == (1,):
        return np.asscalar(d)
    else:
        return d


def ypr_to_rotation_matrix(ypr):
    q = ypr_to_quaternion(ypr)
    return quaternion_to_matrix(q)


def matrix_to_ypr(R):
    q = matrix_to_quaternion(R)
    ypr = quaternion_to_ypr(q)
    return ypr


def matrix_to_quaternion(R):
    """Convert 3x3 orthogonal rotation matrix to quaternion. Implementation
    from https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Not yet vectorized to handle (N,3,3) arrays of rotation matrices (TODO)

    Parameters
    ----------
    R : ndarray - shape (3, 3)

    Returns
    -------
    q : ndarray - shape (4,)
    """
    if R.ndim != 2:
        raise NotImplementedError(
            'Vectorized R -> quaternion conversion not yet supported.')

    t = np.trace(R)

    if t >= 0:
        r = np.sqrt(1 + t)
        s = 0.5/r
        w = 0.5*r
        x = (R[2, 1] - R[1, 2])*s
        y = (R[0, 2] - R[2, 0])*s
        z = (R[1, 0] - R[0, 1])*s
    else:
        r = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        s = 0.5/r
        w = (R[2, 1] - R[1, 2])*s
        x = 0.5*r
        y = (R[0, 1] + R[1, 0])*s
        z = (R[2, 0] + R[0, 2])*s

    return normalize(np.array([w, x, y, z]))


def ypr_to_quaternion(ypr):
    """Convert array of (yaw, pitch, roll) values, either 1D for a single
    triplet or 2D for multiple triplets, to quaternion(s).

    Parameters
    ----------
    ypr : ndarray - shape (3,) or (N, 3)

    Returns
    -------
    ndarray - shape (4,) or (N, 4)
    """
    ypr = np.atleast_2d(ypr)
    assert ypr.shape[1] == 3
    y, p, r = ypr[:, 0], ypr[:, 1], ypr[:, 2]

    cy = np.cos(y/2)
    cp = np.cos(p/2)
    cr = np.cos(r/2)
    sy = np.sin(y/2)
    sp = np.sin(p/2)
    sr = np.sin(r/2)

    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy

    return np.array([qw, qx, qy, qz]).T.squeeze()


def quaternion_to_ypr(q):
    q = normalize(q)
    q = np.atleast_2d(q)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    ysqr = y*y

    t0 = +2.0*(w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + ysqr)
    rx = np.arctan2(t0, t1)

    t2 = +2.0*(w*y - z*x)
    t2[t2 > 1.] = 1.
    t2[t2 < -1.] = -1.
    ry = np.arcsin(t2)

    t3 = +2.0*(w*z + x*y)
    t4 = +1.0 - 2.0*(ysqr + z*z)
    rz = np.arctan2(t3, t4)

    return np.array([rz, ry, rx]).T.squeeze()


def normalize(q):
    if q.ndim == 1:
        n = qnorm(q)
        # if norm == 0:
        #     raise ZeroDivisionError('q norm is zero.')
        if n > 0:
            return q/n
        else:
            return qnull()
    elif q.ndim == 2:
        norms = qnorm(q)

        bad_rows = np.where(norms == 0)
        for i in bad_rows:
            q[i] = qnull()
        norms[norms == 0] = 1.0
        return q/norms[:, np.newaxis]
    else:
        raise ValueError('Invalid shape: {}'.format(q.shape))


def quaternion_to_matrix(q):
    """Return 3x3 rotation matrix from q."""
    q = normalize(q)

    Q = q_matrix(q)
    Qi = q_inv_matrix(q)

    if q.ndim == 1:
        M = np.matmul(Q, Qi.T)
        return M[1:4, 1:4]
    elif q.ndim == 2:
        M = np.matmul(Q, Qi.transpose(0, 2, 1))
        return M[:, 1:4, 1:4]
    else:
        raise ValueError('Invalid shape: {}'.format(q.shape))


def q_matrix(q):
    """Computes matrix or matrices for efficiently implementing quaternion
    multiplication. This function does NOT compute rotation matrices. For that,
    see quaternion_to_matrix.

    Parameters
    ----------
    q : ndarray - shape (4,) or shape (N, 4)

    Returns
    -------
    Q : ndarray - shape (4, 4) or shape (N, 4, 4)
    """
    q = np.atleast_2d(q)

    Q = np.array([
        [+q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]],
        [+q[:, 1], +q[:, 0], -q[:, 3], +q[:, 2]],
        [+q[:, 2], +q[:, 3], +q[:, 0], -q[:, 1]],
        [+q[:, 3], -q[:, 2], +q[:, 1], +q[:, 0]]]).transpose((2, 0, 1))

    return Q.squeeze()


def q_inv_matrix(q):
    """Computes inverse matrix for efficiently implementing quaternion
    multiplication. This function does NOT compute rotation matrices. For that,
    see quaternion_to_matrix.

    Parameters
    ----------
    q : ndarray - shape (4,) or shape (N, 4)

    Returns
    -------
    Q : ndarray - shape (4, 4) or shape (N, 4, 4)
    """
    q = np.atleast_2d(q)

    Q = np.array([
        [+q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]],
        [+q[:, 1], +q[:, 0], +q[:, 3], -q[:, 2]],
        [+q[:, 2], -q[:, 3], +q[:, 0], +q[:, 1]],
        [+q[:, 3], +q[:, 2], -q[:, 1], +q[:, 0]]]).transpose((2, 0, 1))

    return Q.squeeze()


def qinv(q):
    """Return conjugate/inverse of a quaternion or array of quaternions.

    Parameters
    ----------
    q : ndarray - shape (4,) or (N, 4)
    """
    validate(q)

    if q.ndim == 1 and q.shape[0] == 4:
        return np.array([q[0], *-q[1:]])
    elif q.ndim == 2 and q.shape[1] == 4:
        return np.hstack([q[:, :1], -q[:, 1:]])
    else:
        raise ValueError('Incorrect shape: {}'.format(q.shape))


def vector_quaternion(v):
    """Return quaternion(s) from vector(s) v by prepending a zero or a column
    of zeros. Vector quaternions have a scalar component of zero and are used
    in e.g. rotations.

    Parameters
    ----------
    v : ndarray - shape (3,) or (N, 3)

    Returns
    -------
    ndarray - shape (4,) or (N, 4)
    """
    v2 = np.atleast_2d(v)
    nrows, ncols = v2.shape
    assert ncols == 3, 'v should be a 3-vector or array of 3-vectors.'
    vq = np.hstack([np.zeros([nrows, 1]), v2])
    if v.ndim == 1:
        return vq.squeeze()
    else:
        return vq


def qrotate(x, q):
    """Rotate a 3-vector or array of 3-vectors by the unit quaternion or array
    of unit quaternions q."""

    p = vector_quaternion(x)  # prepend column of zeros
    x4 = np.atleast_2d(qmult(q, qmult(p, qinv(q))))
    return x4[:, 1:].squeeze()

    # This also works, but is slower:
    # p = expq(x/2)
    # x4 = qmult(q, qmult(p, qinv(q)))
    # return 2*logq(x4)


def qmult(a, b):
    """Multiplication op for unit quaternions, vectorized if a and b contain
    multiple quaternions in rows.

    Parameters
    ----------
    a, b : ndarray - either or both can have shape (N, 4) or shape (4,)
        Quaternions in N rows

    Returns
    -------
    q : ndarray - shape (N, 4) or shape (4,)
        Quaternion product(s)
    """
    assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray), \
        'arguments must be numpy arrays. Received {} and {}.'.format(
            type(a), type(b))

    if a.ndim == 1 and b.ndim == 1:
        # Single quaternions were provided
        av, bv = a[1:], b[1:]
        if len(av) != len(bv):
            raise ValueError(
                'Length mismatch: {} vs {}'.format(len(av), len(bv)))
        q0 = a[0]*b[0] - np.dot(av, bv)  # scalar component q_0
        qv = a[0]*bv + b[0]*av + np.cross(av, bv)  # vector component q_1 - q_3
        return np.array([q0, *qv])
    elif a.ndim == 1 and b.ndim == 2:
        A = q_matrix(a)  # (4,4)
        return np.matmul(A, b.T).T  # (4,4) x (N,4)' -> (4,N)'
    elif a.ndim == 2 and b.ndim == 1:
        A = q_matrix(a)  # (N,4,4)
        return np.matmul(A, b)  # (N,4,4) x (4,) -> (N,4)
    elif a.ndim == 2 and b.ndim == 2 \
            and a.shape[1] == 4 and b.shape[1] == 4 \
            and a.shape[0] == b.shape[0]:
        return _qmult_vectorized(a, b)
    else:
        raise ValueError(
            'Invalid array shapes passed to geom.qmult(): {}, {}'.format(
                a.shape, b.shape))


def _qmult_vectorized(a, b):
    """Vectorized multiplication op for 2d arrays of unit quaternions,
    which are expected to be contained in the rows.

    Parameters
    ----------
    a, b : ndarray - shape (N, 4)
        Quaternions in N rows
    """
    a0, b0 = a[:, :1], b[:, :1]  # N x 1
    av, bv = a[:, 1:], b[:, 1:]  # N x 3

    q0 = a0*b0 - np.sum(av*bv, axis=1)[:, np.newaxis]  # scalar components

    qv = a0*bv + b0*av + np.cross(av, bv)  # vector components

    return np.hstack([q0, qv])


def validate(x):
    assert isinstance(x, np.ndarray), 'Not of type ndarray: {}'.format(type(x))
    assert x.ndim == 1 or x.ndim == 2, 'Incorrect rank: {}'.format(x.ndim)


def _wrap(theta):
    """Wrap theta into [0, 2*pi)."""
    a = np.atleast_1d(theta)
    a = ((a + pi) % (2*pi)) - pi
    a[a == -pi] = pi
    if a.shape == (1,):
        return np.asscalar(a)
    else:
        return a


def q_vec_norm(q):
    validate(q)
    if q.ndim == 1:
        return np.linalg.norm(q[1:4])
    else:
        return np.linalg.norm(q[:, 1:4], axis=1)[:, np.newaxis]  # Nx1


def q_axis(q):
    validate(q)
    vec_norms = q_vec_norm(q)

    # Handle case(s) where vector norm == 0.
    # Set to 1 so that we will have zeros/0 -> zeros/1 below.
    if np.isscalar(vec_norms):
        if vec_norms == 0:
            vec_norms = 1.0
    else:
        vec_norms[vec_norms == 0] = 1.0

    if q.ndim == 1:
        return q[1:4]/vec_norms
    else:
        return q[:, 1:4]/vec_norms


def q_angle(q):
    validate(q)
    if q.ndim == 1:
        angle = 2.*np.arctan2(q_vec_norm(q), q[0])
        return _wrap(angle)
    else:
        return _wrap(2.*np.arctan2(q_vec_norm(q), q[:, :1]))


def expq(v):
    """Apply an exponential mapping from a rotation vector v to a unit
    quaternion q, using the correspondence between Lie algebras and Matrix Lie
    groups.

    Note: the user should call q = expq(w/2) if ||w|| is the rotation angle
    about w.

    Parameters
    ----------
    v : ndarray - shape (3,) or (N, 3)
        Rodrigues/axis-angle/rotation vector whose norm encodes the rotation
        about its direction.

    Returns
    -------
    q : ndarray - shape (4,)
        Unit quaternion
    """
    validate(v)
    if np.array(v).ndim == 1:
        if v.size != 3:
            raise ValueError('v not a 3-vector. Received {}'.format(v))
        theta = np.linalg.norm(v)
        if theta == 0:
            return qnull()
        qv = sin(theta)/theta * np.array(v)
        return np.array([cos(theta), *qv])
    elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 3:
        result = np.zeros([v.shape[0], 4])
        thetas = np.linalg.norm(v, axis=1)
        result[:, 0] = np.cos(thetas)
        nulls = thetas == 0  # mask for rows with zero norm. shape (N,)
        nz = thetas[~nulls]
        result[~nulls, 1:] = (np.sin(nz)/nz)[:, np.newaxis] * v[~nulls]
        return result
    else:
        raise ValueError('Invalid v: {}'.format(v))


def qnorm(q):
    """2-norm of quaternion or array of quaternions. Returns a scalar or 1d
    array. Should be 1.0 or very nearly so for unit quaternions."""
    norms_squared = np.sum(np.atleast_2d(q)**2, axis=1)
    if q.ndim == 1:
        return np.sqrt(norms_squared[0])
    elif q.ndim == 2:
        return np.sqrt(norms_squared)
    else:
        raise ValueError('q should be 1 or 2 dimensional. q.shape: '.format(
            q.shape))


def qnull(*args):
    """Return a quaternion or array of quaternions representing no rotation,
    i.e. one or more [1, 0, 0, 0]'s.

    Parameters
    ----------
    n : int (optional)
        If provided, return an array with shape (n, 4). If not provided,
        return array with shape (4,).
    """
    if len(args) == 0:
        n = 1
    elif len(args) == 1:
        n = args[0]
    else:
        raise ValueError('qrand accepts zero or one arguments.')
    q = np.zeros([n, 4])
    q[:, 0] = 1.0
    return q.squeeze()


def qrand(*args):
    """Generate array of n uniform random unit quaternions.

    Parameters
    ----------
    n : int (optional)
        If provided, return an array with shape (n, 4). If not provided,
        return array with shape (4,).
    """

    if len(args) == 0:
        n = 1
    elif len(args) == 1:
        n = args[0]
    else:
        raise ValueError('qrand accepts zero or one arguments.')

    rs = np.random.random_sample((n, 3))
    a, b, c = rs[:, 0], rs[:, 1], rs[:, 2]
    d, e = 2*pi*b, 2*pi*c
    q = [
        np.sqrt(1.0 - a)*(np.sin(d)),
        np.sqrt(1.0 - a)*(np.cos(d)),
        np.sqrt(a)*(np.sin(e)),
        np.sqrt(a)*(np.cos(e))
    ]
    return np.array(q).T.squeeze()


def expq_approx(v):
    """Like expq (see those docs), but assumes the rotation angle is small such
    that cos(x) ~ 1 and sin(x) ~ x."""
    return np.array(1, *v)


def logq(q):
    """Inverse of expq(). Maps quaternions to 3-vectors."""
    return 0.5*q_angle(q)*q_axis(q)


def quaternion_update(q, w):
    """Update a unit quaternion or array of unit quaternions q with a rotation
    vector or array of rotation vectors w: q <- q*dq, where dq is expq(w/2).

    Parameters
    ----------
    q : ndarray - shape (4,) or (N, 4)
        Original orientation(s)
    w : ndarray - shape (3,) or (N, 3)
        Rotation vector(s)

    Returns
    -------
    ndarray - shape (4,) or (N, 4)
        New orientation(s)
    """
    return qmult(q, expq(0.5*w))


def small_quaternion_update(self, qs, vs):
    """Vectorized orientation update for multiple quaternions. Assumes that the
    rotation vectors in `vs` are small changes about the quaternions in `qs`.
    Based on the the exponential map expq() and the definition of quaternion
    multiplication. Useful in applications such as particle filters and UKFs.

    Parameters
    ----------
    qs : ndarray (N, 4)
     Orientation quaternions
    vs : ndarray (N, 3)
     Angular velocity vectors
    """
    qs[:, 0] -= 0.5*np.sum(qs[:, 1:4] * vs, axis=1)
    qs[:, 1:4] += 0.5*qs[:, (0,)]*vs - 0.5*np.cross(qs[:, 1:4], vs)
    return qs
