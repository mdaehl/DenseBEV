# ---------------------------------------------------
# Copyright (c) DenseBEV 2025. All rights reserved.
# ---------------------------------------------------

from mmdet3d.models.detectors.base import Base3DDetector
from mmdet.models import DETECTORS
import torch
from typing import Optional, Dict, List
from einops import rearrange
from mmcv.runner import auto_fp16
from projects.mmdet3d_plugin.misc.grid_mask import GridMask
from mmdet3d.models import builder
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes
import copy
from mmdet3d.core import bbox3d2result


@DETECTORS.register_module()
class DenseBEV3D(Base3DDetector):
    def __init__(
        self,
        img_backbone: dict,
        img_neck: dict,
        pts_bbox_head: dict,
        train_cfg: dict,
        test_cfg: dict,
        video_test_mode: bool,
        use_grid_mask: bool = False,
    ):
        super().__init__(init_cfg=None)

        self.video_test_mode = video_test_mode

        # build layers
        # backbone
        self.img_backbone = builder.build_backbone(img_backbone)
        # img neck
        self.img_neck = builder.build_neck(img_neck)
        # head alias transformer (pts is historical naming convention)
        pts_train_cfg = train_cfg.pts if train_cfg else None
        pts_bbox_head.update(train_cfg=pts_train_cfg)
        pts_test_cfg = test_cfg.pts if test_cfg else None
        pts_bbox_head.update(test_cfg=pts_test_cfg)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # required for grid mask augmentation
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )

        cache_dict = dict()
        for m in self.modules():
            m.cache_dict = cache_dict

        self.cache_dict["cls_branches"] = self.pts_bbox_head.cls_branches
        self.cache_dict["distinct_query_mask"] = []
        self.cache_dict["num_heads"] = (
            self.pts_bbox_head.transformer.encoder.layers[0].attentions[0].num_heads
        )

        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.test_flag = False  # for scene groups
        self.prev_scene_token = None  # for eval

    def forward(self, return_loss: bool = True, **kwargs) -> dict:
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[Dict[str, list]],
        gt_bboxes_3d: List[LiDARInstance3DBoxes],
        gt_labels_3d: List[torch.Tensor],
    ) -> dict:
        if self.test_flag and self.pts_bbox_head.use_memory: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False

        current_imgs = img[:, -1, ...]
        prev_imgs_queue = img[:, :-1, ...]

        len_queue = img.shape[1]

        # prev img metas are the same as from current timestamp
        prev_img_metas = copy.deepcopy(img_metas)

        # get previous bev feats if prev info exists
        prev_bev = self.obtain_history_bev(prev_imgs_queue, prev_img_metas)
        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]["prev_bev_exists"]:
            prev_bev = None

        # forward current timestamp img through backbone and grid mask
        img_feats = self.extract_feat(img=current_imgs)

        losses = self.forward_transformer_train(
            img_feats=img_feats,
            img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            prev_bev=prev_bev,
        )
        return losses

    def forward_test(
        self, img: List[torch.Tensor], img_metas: Dict[str, list], **kwargs
    ):
        self.test_flag = True
        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        # update idx
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0][0]["can_bus"][-1] = 0
            img_metas[0][0]["can_bus"][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img[0], img_metas[0], prev_bev=self.prev_frame_info["prev_bev"], **kwargs
        )
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev
        return bbox_results

    @auto_fp16(apply_to=("img"))
    def extract_feat(
        self, img: torch.Tensor, len_queue: Optional[int] = None
    ) -> torch.Tensor:
        img_feats = self.extract_img_feat(img, len_queue=len_queue)
        return img_feats

    def extract_img_feat(
        self, img: torch.Tensor, len_queue: Optional[int] = None
    ) -> torch.Tensor:
        if img is None:
            return None

        # store batch size
        B = img.shape[0]

        # flatten the multi camera dimension
        img = rearrange(img, "b n_cam c h w -> (b n_cam) c h w")

        # grid mask augmentation
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)
        # assure unified format, if multiple layers are output from backbone
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        img_feats = self.img_neck(img_feats)

        # reshape based on queue settings
        img_feats_reshaped = []
        for img_feat in img_feats:
            if len_queue is not None:
                B_queue = int(B / len_queue)
                img_feat_reshaped = rearrange(
                    img_feat,
                    "(b_q n_q n_cam) c h w -> b_q n_q n_cam c h w",
                    b_q=B_queue,
                    n_q=len_queue,
                )
            else:
                img_feat_reshaped = rearrange(
                    img_feat, "(b n_cam) c h w -> b n_cam c h w", b=B
                )

            img_feats_reshaped.append(img_feat_reshaped)

        return img_feats_reshaped

    def obtain_history_bev(
        self, imgs_queue: torch.Tensor, img_metas_list: Dict[str, list]
    ) -> torch.Tensor:
        # avoid backpropagation and set to eval
        self.eval()
        with torch.no_grad():
            prev_bev = None
            len_queue = imgs_queue.shape[1]
            imgs_queue = rearrange(
                imgs_queue, "b n_q n_cam c h w -> (b n_q) n_cam c h w"
            )

            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            # iterate over items of
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]

                if not img_metas[0]["prev_bev_exists"]:
                    prev_bev = None
                # select respective queue item
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True
                )

        self.train()
        return prev_bev

    def forward_transformer_train(
        self,
        img_feats: torch.Tensor,
        img_metas: Dict[str, list],
        gt_bboxes_3d: List[LiDARInstance3DBoxes],
        gt_labels_3d: List[torch.Tensor],
        prev_bev: Optional[torch.tensor] = None,
    ) -> dict:
        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test(
        self,
        img: torch.Tensor,
        img_metas: Dict[str, list],
        prev_bev: Optional[torch.Tensor],
        rescale: bool = False,  # not used
    ): 
        img_feats = self.extract_feat(img=img)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(img_feats, img_metas, prev_bev)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return new_prev_bev, bbox_list

    def simple_test_pts(
        self,
        img_feats: List[torch.Tensor],
        img_metas: Dict[str, list],
        prev_bev: Optional[torch.Tensor],
    ):
        # WORKS ONLY FOR BATCH SIZE 1
        if self.prev_frame_info["prev_bev"] is None:
            img_metas[0]['prev_bev_exists'] = img_feats[0].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            img_metas[0]['prev_bev_exists'] = img_feats[0].new_ones(1)
        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return outs["bev_embed"], bbox_results
    
    def aug_test(self, **kwargs):
        raise NotImplementedError
