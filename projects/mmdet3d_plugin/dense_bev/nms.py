# ---------------------------------------------------
# Copyright (c) DenseBEV 2025. All rights reserved.
# ---------------------------------------------------

import torch
from ops.functions import suppress_nms_3d


def suppress_detections(
        proposals: torch.Tensor,  # x y z dy dy dz (vx vy)
        scores: torch.Tensor,  # score of best class
        nms_threshold: float
    ) -> torch.Tensor:
    keep_idxs = suppress_nms_3d(
        proposals[:, :7], scores, nms_threshold
    )

    return keep_idxs
