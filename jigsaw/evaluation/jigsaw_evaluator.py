import torch
from jigsaw.evaluation.transform import (
    transform_pc,
    quaternion_to_euler
)
from jigsaw.evaluation.loss import _valid_mean
from typing import List, Optional, Tuple, Union



def trans_metrics(trans1, trans2, valids, metric):
    """Evaluation metrics for translation.

    Metrics used in the NSM paper.

    Args:
        trans1: [B, P, 3], pred translation
        trans2: [B, P, 3], gt translation
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ['mse', 'rmse', 'mae']
    if metric == 'mse':
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1)  # [B, P]
    elif metric == 'rmse':
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1)**0.5
    else:
        metric_per_data = (trans1 - trans2).abs().mean(dim=-1)
    metric_per_data = _valid_mean(metric_per_data, valids)
    return metric_per_data


@torch.no_grad()
def rot_metrics(rot1, rot2, valids, metric):
    """Evaluation metrics for rotation in euler angle (degree) space.

    Metrics used in the NSM paper.

    Args:
        rot1: [B, P, 4], pred quat
        rot2: [B, P, 4], gt quat
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ['mse', 'rmse', 'mae']
    deg1 = quaternion_to_euler(rot1, to_degree=True)  # [B, P, 3]
    deg2 = quaternion_to_euler(rot2, to_degree=True)

    diff1 = (deg1 - deg2).abs()
    diff2 = 360. - (deg1 - deg2).abs()
    # since euler angle has the discontinuity at 180
    diff = torch.minimum(diff1, diff2)
    if metric == 'mse':
        metric_per_data = diff.pow(2).mean(dim=-1)  # [B, P]
    elif metric == 'rmse':
        metric_per_data = diff.pow(2).mean(dim=-1)**0.5
    else:
        metric_per_data = diff.abs().mean(dim=-1)
    metric_per_data = _valid_mean(metric_per_data, valids)
    return metric_per_data


@torch.no_grad()
def calc_part_acc(pts, trans1, trans2, rot1, rot2, valids, chamfer_distance):
    """Compute the `Part Accuracy` in the paper.

    We compute the per-part chamfer distance, and the distance lower than a
        threshold will be considered as correct.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        trans1: [B, P, 3], pred_translation
        trans2: [B, P, 3], gt_translation
        rot1: [B, P, 4], Rotation3D, quat or rmat
        rot2: [B, P, 4], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], accuracy per data in the batch
    """
    B, P = pts.shape[:2]

    pts1 = transform_pc(trans1, rot1, pts)  # [B, P, N, 3]
    pts2 = transform_pc(trans2, rot2, pts)

    pts1 = pts1.flatten(0, 1)  # [B*P, N, 3]
    pts2 = pts2.flatten(0, 1)
    loss_per_data = chamfer_distance(pts1, pts2, bidirectional=True, 
                                    point_reduction="mean", batch_reduction=None,)  # [B*P, N]
    loss_per_data = loss_per_data.view(B, P).type_as(pts)

    # part with CD < `thre` is considered correct
    thre = 0.01
    acc = (loss_per_data < thre) & (valids == 1)
    # the official code is doing avg per-shape acc (not per-part)
    acc = acc.sum(-1) / (valids == 1).sum(-1)
    return acc


def randn_tensor(
        shape: Union[Tuple, List],
        generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
        layout: Optional["torch.layout"] = None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        transformations = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        transformations = torch.cat(transformations, dim=0).to(device)
    else:
        transformations = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return transformations

        