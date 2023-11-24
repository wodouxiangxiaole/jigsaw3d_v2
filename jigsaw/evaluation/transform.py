from pytorch3d.transforms import quaternion_apply
from pytorch3d import transforms
import torch


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
        where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    # repeat to e.g. apply the same quat for all points in a point cloud
    # [4] --> [N, 4], [B, 4] --> [B, N, 4], [B, P, 4] --> [B, P, N, 4]
    if len(q.shape) == len(v.shape) - 1:
        q = q.unsqueeze(-2).repeat_interleave(v.shape[-2], dim=-2)
    assert q.shape[:-1] == v.shape[:-1]
    return quaternion_apply(q, v)


def qtransform(t, q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q,
        and then translate it by the translation described by t.
    Expects a tensor of shape (*, 3) for t, a tensor of shape (*, 4) for q and
        a tensor of shape (*, 3) for v, where * denotes any dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert t.shape[-1] == 3

    # repeat to e.g. apply the same trans for all points in a point cloud
    # [3] --> [N, 3], [B, 3] --> [B, N, 3], [B, P, 3] --> [B, P, N, 3]
    if len(t.shape) == len(v.shape) - 1:
        t = t.unsqueeze(-2).repeat_interleave(v.shape[-2], dim=-2)

    assert t.shape == v.shape

    qv = qrot(q, v)
    tqv = qv + t
    return tqv


def transform_pc(trans, rot, pc, rot_type=None):
    """Rotate and translate the 3D point cloud.

    Args:
        rot (torch.Tensor): quat
    """
    
    return qtransform(trans, rot, pc)


def quaternion_to_euler(quat, to_degree=True):
    """Convert quaternion to euler angle.

    Args:
        quat: [B, 4], quat
        to_degree: bool, whether to convert to degree

    Returns:
        [B, 3], euler angle
    """

    r_mat = transforms.quaternion_to_matrix(quat)
    euler = transforms.matrix_to_euler_angles(r_mat, convention="XYZ")
    if to_degree:
        euler = torch.rad2deg(euler)

    return euler
