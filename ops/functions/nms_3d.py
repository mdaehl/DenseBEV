# ------------------------------------------------------------------------
# Copyright (c) OpenPCDeT
# ------------------------------------------------------------------------
import torch
import iou3d_nms


def suppress_nms_3d(
    boxes: torch.Tensor, scores: torch.Tensor, threshold: float
) -> torch.Tensor:
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms.nms_gpu(
        boxes, keep, threshold
    )  # also updates keep tensor inplace
    return order[keep[:num_out]]

