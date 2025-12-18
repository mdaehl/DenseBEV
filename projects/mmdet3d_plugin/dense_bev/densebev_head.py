# ---------------------------------------------------
# Copyright (c) DenseBEV 2025. All rights reserved.
# ---------------------------------------------------

from mmdet.models import HEADS
from mmdet.core import multi_apply, reduce_mean
from mmdet.models.dense_heads import DETRHead
from mmcv.cnn import bias_init_with_prob, constant_init
from mmcv.utils import TORCH_VERSION, digit_version
from .utils import normalize_bbox
import torch.nn as nn
import torch
from typing import List, Optional, Dict, Tuple
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.runner import auto_fp16, force_fp32
from mmdet3d.core.bbox.coders import build_bbox_coder
import copy
from .utils import coordinate_to_encoding
from .misc_memory import memory_refresh, transform_reference_points, topk_gather, nerf_positional_encoding ,pos2posemb1d
from einops import rearrange, repeat
import numpy as np


@HEADS.register_module()
class DenseBEVHead(DETRHead):
    def __init__(
        self,
        transformer: dict,
        bbox_coder: dict,
        bev_h: int,
        bev_w: int,
        num_classes: int,
        use_memory: bool,
        code_weights: Optional[List[float]] = None,
        num_reg_fcs: int = 2,
        num_propagated: int = 300,
        memory_len: int = 1200,
        **kwargs,
    ):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.use_memory = use_memory

        if code_weights is not None:
            code_weights = code_weights  # (cx,cy,l,w,cz,h,sin(φ),cos(φ))
        else:
            code_weights = [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.2,
                0.2,
            ]  # (cx,cy,l,w,cz,h,sin(φ),cos(φ),v_x,v_y)

        self.code_size = len(code_weights)

        # add relevant modules
        transformer["num_reg_fcs"] = num_reg_fcs
        transformer["num_classes"] = num_classes
        transformer["code_size"] = len(code_weights)
        transformer["use_memory"] = use_memory

        if self.use_memory:
            self.num_propagated = num_propagated

            if self.code_size == 8:
                self.use_velo_memory = False
                # forward to decoder
                transformer["decoder"]["use_velo_memory"] = False
            elif self.code_size == 10:
                self.use_velo_memory = True
                # forward to decoder
                transformer["decoder"]["use_velo_memory"] = True
            else:
                raise ValueError("Not supported code size for memory.")
        else:
            self.num_propagated = 0
            assert self.num_propagated == 0

        super().__init__(
            transformer=transformer,
            num_reg_fcs=num_reg_fcs,
            num_classes=num_classes,
            **kwargs,
        )

        # transform code weights to torch params
        self.code_weights = nn.Parameter(
            torch.tensor(code_weights, requires_grad=False), requires_grad=False
        )

        # bbox decoder (unormalize and filtering)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        # forward pc range to transformer for decoding in nms procedure
        self.transformer.pc_range = self.pc_range
        self.transformer.decoder.pc_range = self.pc_range

        self.embed_dims = self.transformer.embed_dims

        if self.use_memory:
            self.pc_range = nn.Parameter(torch.tensor(
                self.bbox_coder.pc_range), requires_grad=False)
            self.memory_len = memory_len
            self.reset_memory()

            assert self.num_propagated <= self.transformer.num_queries, "Cannot propagate more objects than being predicted."
            assert self.memory_len >= self.transformer.num_queries, "Memory needs to be able to hold at least one prediction."
            assert self.memory_len % self.num_propagated == 0, "Number of propagated queries should be a multiple of the memory length."

    def _init_layers(self):
        # base cls
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        # base reg
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        num_pred_layer = self.transformer.decoder.num_layers

        # regular branches
        self.reg_branches = nn.ModuleList(
            [copy.deepcopy(reg_branch) for _ in range(num_pred_layer)]
        )
        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(cls_branch) for _ in range(num_pred_layer)]
        )

        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

        if self.use_memory:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)
            nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

        bias_init = bias_init_with_prob(0.01)

        branches = [self.cls_branches]
        # cls branches
        for branch in branches:
            for m in branch:
                nn.init.constant_(m[-1].bias, bias_init)

        branches = [self.reg_branches]
        # reg branches
        for branch in branches:
            for m in branch:
                constant_init(m[-1], 0, bias=0)
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def reset_memory(self) -> None:
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_velo = None
        self.memory_egopose = None

    def pre_update_memory(self, data: dict, bs: int, device: torch.device) -> None:
        prev_exist = torch.tensor([i["prev_bev_exists"] for i in data], device=device, dtype=torch.float32)
        timestamp = rearrange(torch.tensor([i["timestamp"] for i in data], device=device), "bs -> bs 1 1")
        ego_pose_inv = torch.tensor(np.array([i["ego_pose_inv"] for i in data]), device=device)

        if self.memory_embedding is None:
            self.memory_embedding = torch.zeros(bs, self.memory_len, self.embed_dims, device=device)
            self.memory_reference_point = torch.zeros(bs, self.memory_len, 3, device=device)
            self.memory_timestamp = torch.zeros(bs, self.memory_len, 1, device=device)
            self.memory_egopose = torch.zeros(bs, self.memory_len, 4, 4, device=device)
            if self.use_velo_memory:
                self.memory_velo = torch.zeros(bs, self.memory_len, 2, device=device)
        else:
            self.memory_timestamp += timestamp
            self.memory_egopose = ego_pose_inv[:, None] @ self.memory_egopose
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], prev_exist)
            self.memory_reference_point = transform_reference_points(self.memory_reference_point, ego_pose_inv, reverse=False)
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], prev_exist)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], prev_exist)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], prev_exist)
            if self.use_velo_memory:
                self.memory_velo = memory_refresh(self.memory_velo[:, :self.memory_len], prev_exist)

        # for the first frame, padding pseudo_reference_points (non-learnable)
        pseudo_reference_points = self.pseudo_reference_points.weight * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
        self.memory_reference_point[:, :self.num_propagated]  = self.memory_reference_point[:, :self.num_propagated] + (1 - prev_exist).view(bs, 1, 1) * pseudo_reference_points
        self.memory_egopose[:, :self.num_propagated]  = self.memory_egopose[:, :self.num_propagated] + (1 - prev_exist).view(bs, 1, 1, 1) * torch.eye(4, device=device)
    
    def post_update_memory(self, 
                           all_bbox_preds: torch.Tensor, 
                           all_cls_scores: torch.Tensor, 
                           hidden_states: torch.Tensor,
                           data: dict):
        device = all_bbox_preds.device
        bs = all_bbox_preds.shape[1]
        ego_pose = torch.tensor(np.array([i["ego_pose"] for i in data]), device=device, dtype=torch.float32)
        rec_ego_pose = repeat(torch.eye(4, device=device), "n m -> bs n_q n m", bs=bs, n_q=self.num_propagated)
        timestamp = rearrange(torch.tensor(np.array([i["timestamp"] for i in data]), device=device), "n -> n 1 1")
        # get data from prediction
        rec_reference_points = all_bbox_preds[..., [0, 1, 4]][-1]
        rec_velo = all_bbox_preds[..., -2:][-1]
        rec_score = all_cls_scores[-1].sigmoid().max(-1).values
        rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)[..., None]
        rec_memory = hidden_states[-1]

        # select topk
        _, topk_indexes = torch.topk(rec_score, self.num_propagated, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(rec_memory, topk_indexes).detach()
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()

        # concat to memory
        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)   
        self.memory_egopose= torch.cat([rec_ego_pose, self.memory_egopose], dim=1)     
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        if self.use_velo_memory:
            self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)

        # update
        self.memory_reference_point = transform_reference_points(self.memory_reference_point, ego_pose, reverse=False)
        self.memory_timestamp -= timestamp  # so the current timestamp is basically zero
        self.memory_egopose = ego_pose[:, None] @ self.memory_egopose


    @auto_fp16(apply_to=("mlvl_feats"))
    def forward(
        self,
        mlvl_feats: List[torch.Tensor],
        img_metas: List[Dict[str, list]],
        prev_bev: Optional[torch.Tensor] = None,
        only_bev: bool = False,
    ) -> Dict:
        # lists are w.r.t. multi scale output (if multi output from backbone)
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)
        device = bev_queries.device
        
        bev_mask = torch.zeros(
            (bs, self.bev_h, self.bev_w), device=device
        ).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:  # only use encoder to obtain BEV features
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        
        if self.use_memory:
            self.pre_update_memory(img_metas, bs, device)
            memory_output = self.temporal_alignment_prev()
        else:
            memory_output = None

        outputs = self.transformer(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches,
            img_metas=img_metas,
            prev_bev=prev_bev,
            memory_output=memory_output
        )

        # returned references in sigmoid space
        bev_embed, hs, inter_references, head_loss_inputs_dict = outputs
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            reference = inter_references[lvl]
            reference = inverse_sigmoid(reference)

            hidden_state = hs[lvl]

            outputs_class = self.cls_branches[lvl](hidden_state)
            tmp = self.reg_branches[lvl](hidden_state)

            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            )

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
        }
        outs.update(head_loss_inputs_dict)

        if self.use_memory:
            self.post_update_memory(outputs_coords, outputs_classes, hs, img_metas)

        return outs

    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts: dict):
        loss_dict = dict()

        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]

        # from two stage branch
        enc_outputs_class = preds_dicts.get("enc_outputs_class")
        enc_outputs_coord = preds_dicts.get("enc_outputs_coord")

        # convert to tensor
        device = gt_labels_list[0].device
        gt_bboxes_list = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(
                device
            )
            for gt_bboxes in gt_bboxes_list
        ]

        losses = self.loss_by_feat(
            gt_bboxes_list=gt_bboxes_list,
            gt_labels_list=gt_labels_list,
            all_cls_scores=all_cls_scores,
            all_bbox_preds=all_bbox_preds,
            enc_cls_scores=enc_outputs_class,
            enc_bbox_preds=enc_outputs_coord,
        )
        loss_dict.update(losses)

        return loss_dict

    def loss_by_feat(
        self,
        gt_bboxes_list,
        gt_labels_list,
        all_cls_scores: torch.Tensor,
        all_bbox_preds: torch.Tensor,
        enc_cls_scores: torch.Tensor,
        enc_bbox_preds: torch.Tensor,
    ):
        # duplicate gts
        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        # get distinct and dense predictions
        n = all_cls_scores.shape[2]
        mask = torch.ones(n, device=all_cls_scores.device, dtype=bool)
        mask[self.num_query:-self.num_propagated] = False
        distinct_cls_scores = all_cls_scores[:, :, mask]
        distinct_bbox_preds = all_bbox_preds[:, :, mask]

        # calculate loss for distinct queries (iterate over decoder layers)
        losses_cls, losses_bbox = multi_apply(
            self.loss_by_feat_distinct_single,
            distinct_cls_scores,
            distinct_bbox_preds,
            all_gt_bboxes_list,
            all_gt_labels_list,
            [i for i in range(len(all_gt_bboxes_list))],
        )

        loss_dict = {}
        # loss from final decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]
        # loss from intermediate decoder layer
        for lid, (loss_cls_i, loss_bbox_i) in enumerate(
            zip(losses_cls[:-1], losses_bbox[:-1])
        ):
            loss_dict[f"d{lid}.loss_cls"] = loss_cls_i
            loss_dict[f"d{lid}.loss_bbox"] = loss_bbox_i

        # loss of aux encoder head
        if enc_cls_scores is not None and enc_bbox_preds is not None:
            enc_loss_cls, enc_losses_bbox = self.loss_by_feat_single(
                enc_cls_scores, enc_bbox_preds, gt_bboxes_list, gt_labels_list
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox

        return loss_dict

    def loss_by_feat_distinct_single(
        self, cls_scores, bbox_preds, gt_bboxes_list, gt_labels_list, lid
    ):
        num_imgs = cls_scores.size(0)  # imgs == batch size

        if lid > 0:
            batch_mask = [
                self.cache_dict["distinct_query_mask"][lid - 1][
                    img_id * self.cache_dict["num_heads"]
                ][0]
                for img_id in range(num_imgs)
            ]
        else:
            batch_mask = None

        loss_cls, loss_bbox = self.loss_by_feat_single(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            gt_bboxes_list=gt_bboxes_list,
            gt_labels_list=gt_labels_list,
            batch_mask=batch_mask,
        )
        return loss_cls, loss_bbox

    def loss_by_feat_single(
        self,
        cls_scores,
        bbox_preds,
        gt_bboxes_list,
        gt_labels_list,
        batch_mask=None,
        gt_bboxes_ignore_list=None,
    ):
        num_imgs = cls_scores.size(0)

        if batch_mask:
            # only select the distinct queries in decoder for loss
            cls_scores_list = [cls_scores[i, batch_mask[i]] for i in range(num_imgs)]
            bbox_preds_list = [bbox_preds[i, batch_mask[i]] for i in range(num_imgs)]
        else:
            cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
            bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        cls_scores = torch.cat(cls_scores_list)
        bbox_preds = torch.cat(bbox_preds_list)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        # "normalize", which only applies to size and not position, which is log converted
        normalized_bbox_targets = normalize_bbox(bbox_targets)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos,
        )
        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(
        self, preds_dicts: Tuple[List[dict]], img_metas: Dict[str, list]
    ) -> List[dict]:
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]["box_type_3d"](bboxes, code_size)
            scores = preds["scores"]
            labels = preds["labels"]

            ret_list.append([bboxes, scores, labels])

        return ret_list

    def _get_target_single(
        self,
        cls_score: torch.Tensor,
        bbox_pred: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_bboxes_ignore: Optional[torch.Tensor] = None,
    ):
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore
        )

        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        # sampler does return empty 2d box instead of 3d box
        if len(sampling_result.pos_gt_bboxes) == 0:
            sampling_result.pos_gt_bboxes = torch.zeros_like(bbox_targets[pos_inds])
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_targets(
        self,
        cls_scores_list: List[torch.Tensor],
        bbox_preds_list: List[torch.Tensor],
        gt_bboxes_list: List[torch.Tensor],
        gt_labels_list: List[torch.Tensor],
        gt_bboxes_ignore_list: Optional[List[torch.tensor]] = None,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        int,
        int,
    ]:
        assert gt_bboxes_ignore_list is None, (
            "Only supports for gt_bboxes_ignore setting to None."
        )
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            gt_labels_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def temporal_alignment_prev(self):
        # hacky solution
        ego_pose_memory = self.transformer.decoder.ego_pose_memory
        ego_pose_pe = self.transformer.decoder.ego_pose_pe
        time_embedding = self.transformer.decoder.time_embedding

        # normalize previous reference points (between 0 and 1)
        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[:3])
        # get positional encoding for query and memory
        query_sine_embed = coordinate_to_encoding(temp_reference_point[..., :2], num_feats=self.embed_dims // 2)
        # hacky solution to use same embedding as in decoder
        temp_pos = self.transformer.decoder.ref_point_head(query_sine_embed)

        # process motion info for MLN
        flatten_memory_pose = rearrange(self.memory_egopose[..., :3, :], "bs n_q i j -> bs n_q (i j)")

        # check if velocity memory is used
        if self.use_velo_memory:
            memory_ego_motion = torch.cat([self.memory_velo, self.memory_timestamp, flatten_memory_pose], dim=-1).to(torch.float32)
        else:
            memory_ego_motion = torch.cat([self.memory_timestamp, flatten_memory_pose], dim=-1).to(torch.float32)
            
        memory_ego_motion = nerf_positional_encoding(memory_ego_motion)

        # update memory via MLN
        temp_memory = ego_pose_memory(self.memory_embedding, memory_ego_motion)
        # update pos via MLN and timestamp
        temp_pos = ego_pose_pe(temp_pos, memory_ego_motion)
        temp_pos += time_embedding(pos2posemb1d(self.memory_timestamp).to(torch.float32))

        # select propagate part
        tgt = temp_memory[:, :self.num_propagated]
        query_pos = temp_pos[:, :self.num_propagated]
        reference_points = temp_reference_point[:, :self.num_propagated]

        # memory queue without propagated
        temp_memory = temp_memory[:, self.num_propagated:]
        temp_memory_pos = temp_pos[:, self.num_propagated:]

        return {
            "tgt_prop": tgt,
            "query_pos_prop": query_pos,
            "reference_points_prop": reference_points,
            "temp_memory": temp_memory,
            "temp_pos": temp_memory_pos,
        }
