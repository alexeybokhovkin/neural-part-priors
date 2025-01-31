import sys
import numpy as np
import quaternion
import torch
import math
import plyfile
import skimage

from ..utils.embedder import get_embedder_nerf


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


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)


def create_grid(
    decoder, latent_vec, N=256, max_batch=32 ** 3, offset=None, scale=None,
    input_samples=None, add_feat=None, mode=0, class_one_hot=None
):
    embedder, embedder_out_dim = get_embedder_nerf(10, input_dims=3, i=0)
    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    if input_samples is None:
        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long() / N) % N
        samples[:, 0] = ((overall_index.long() / N) / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    else:
        samples = input_samples
        samples = torch.cat([samples, torch.zeros((len(samples), 1)).cuda()], dim=1)

    num_samples = N ** 3
    samples.requires_grad = False
    head = 0

    if class_one_hot is not None:
        class_one_hot = class_one_hot.cuda()
        latent_vec = torch.hstack([latent_vec, class_one_hot])

    while head < num_samples:
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:3]
        sample_subset = embedder(sample_subset).cuda()

        samples[head: min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder, latent_vec, sample_subset, add_feat=add_feat, mode=mode)
                .squeeze(1)
                .detach()
                .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    sdf_values = sdf_values.detach().cpu().clone()
    samples = samples.detach().cpu().clone()

    return sdf_values, voxel_origin, voxel_size, offset, scale, samples


def decode_sdf(decoder, latent_vector, queries, add_feat=None, mode=0):
    num_samples = queries.shape[0]

    if mode == 2:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)
    elif mode == 1:
        add_feat = add_feat.expand(num_samples, -1)
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([add_feat, latent_repeat, queries], 1).cuda()
    elif mode == 3:
        add_feat = add_feat.expand(num_samples, -1)
        inputs = torch.cat([add_feat, queries], 1).cuda()

    sdf = decoder(inputs)

    return sdf
