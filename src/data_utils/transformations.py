import numpy as np
import quaternion
import torch
import math


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


def perform_rot(pc, aug_rot):
    angle = 7 * aug_rot
    angle = np.pi * angle / 180.

    a, b = np.cos(angle), np.sin(angle)
    matrix = np.array([[a, 0, b],
                       [0, 1, 0],
                       [-b, 0, a]])
    matrix = torch.FloatTensor(matrix)
    pc[:, :3] = pc[:, :3] @ matrix.T

    return pc


def perform_translate_x(pc, aug_trans):
    shift = 0.1 * aug_trans
    pc[:, :3] = pc[:, :3] + torch.FloatTensor(np.array([[shift, 0, 0]]))

    return pc


def perform_translate_y(pc, aug_trans):
    shift = 0.1 * aug_trans
    pc[:, :3] = pc[:, :3] + torch.FloatTensor(np.array([[0, shift, 0]]))

    return pc

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])