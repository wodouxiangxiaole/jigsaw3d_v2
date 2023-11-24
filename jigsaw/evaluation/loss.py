import torch


def _valid_mean(loss_per_part, valids):
    """Average loss values according to the valid parts.

    Args:
        loss_per_part: [B, P]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch, averaged over valid parts
    """
    nan_mask = torch.isnan(loss_per_part)
    loss_per_part[nan_mask] = 0.
    valids = valids.float().detach()
    loss_per_data = (loss_per_part * valids).sum(1) / valids.sum(1)
    return loss_per_data
