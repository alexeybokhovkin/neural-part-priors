import numpy as np
import quaternion
import torch


def from_tqs_to_matrix(translation, quater, scale):
    """
    (T(3), Q(4), S(3)) -> 4x4 Matrix

    :param translation: 3 dim translation vector (np.array or list)
    :param quater: 4 dim rotation quaternion (np.array or list)
    :param scale: 3 dim scale vector (np.array or list)
    :return: 4x4 transformation matrix
    """
    q = np.quaternion(quater[0], quater[1], quater[2], quater[3])
    T = np.eye(4)
    T[0:3, 3] = translation
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(scale)

    M = T.dot(R).dot(S)
    return M


def decompose_mat4(M):
    R = M[0:3, 0:3]
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])

    s = np.array([sx, sy, sz])

    R[:, 0] /= sx
    R[:, 1] /= sy
    R[:, 2] /= sz
    q = quaternion.from_rotation_matrix(R[0:3, 0:3])

    t = M[0:3, 3]
    return t, q, s


def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M


def apply_transform(points, *args):
    """
    points = points х args[0] x args[1] x args[2] x ... args[-1]

    :param points: np.array N x (3|4)
    :param args: array of transformations. May be 4x4 np.arrays, or dict {
        'transformation': [t1, t2, t3],
        'rotation': [q1, q2, q3, q4],
        'scale': [s1, s2, s3],
    }
    :return: transformed points
    """
    # save origin dimensionality and add forth coordinate if needed
    initial_dim = points.shape[-1]
    if initial_dim == 3:
        points = add_forth_coord(points)

    # transform each transformation to 4x4 matrix
    transformations = []
    for transform in args:
        if type(transform) == dict:
            transformations.append(from_tqs_to_matrix(
                translation=transform['translation'],
                quater=transform['rotation'],
                scale=transform['scale']
            ))
        else:
            transformations.append(transform)

    # main loop
    for transform in transformations:
        points = points @ transform.T

    # back to origin dimensionality if needed
    if initial_dim == 3:
        points = points[:, :3]

    return points


def apply_inverse_transform(points, *args):
    """
    points = points х args[0] x args[1] x args[2] x ... args[-1]

    :param points: np.array N x (3|4)
    :param args: array of tranformations. May be 4x4 np.arrays, or dict {
        'transformation': [t1, t2, t3],
        'rotation': [q1, q2, q3, q4],
        'scale': [s1, s2, s3],
    }
    :return: transformed points
    """
    # save origin dimensionality and add forth coordinate if needed
    initial_dim = points.shape[-1]
    if initial_dim == 3:
        points = add_forth_coord(points)

    # transform each transformation to 4x4 matrix
    transformations = []
    for transform in args:
        if type(transform) == dict:
            t = from_tqs_to_matrix(
                translation=transform['translation'],
                quater=transform['rotation'],
                scale=transform['scale']
            )
            t = np.linalg.inv(t)
            transformations.append(t)
        else:
            t = np.linalg.inv(transform)
            transformations.append(t)

    # main loop
    for transform in transformations:
        points = points @ transform.T

    # back to origin dimensionality if needed
    if initial_dim == 3:
        points = points[:, :3]

    return points


def add_forth_coord(points):
    """forth coordinate is const = 1"""
    return np.hstack((points, np.ones((len(points), 1))))


def apply_transform_torch(points, *args):
    # save origin dimensionality and add forth coordinate if needed
    initial_dim = points.shape[-1]
    if initial_dim == 3:
        points = add_forth_coord_torch(points)

    # transform each transformation to 4x4 matrix
    transformations = []
    for transform in args:
        transformations.append(transform)

    # main loop
    for transform in transformations:
        points = points @ transform.T

    # back to origin dimensionality if needed
    if initial_dim == 3:
        points = points[:, :3]

    return points


def add_forth_coord_torch(points):
    """forth coordinate is const = 1"""
    return torch.hstack((points, torch.ones((len(points), 1)).to(points.device)))
