# ------------------------------------------------------------------------
# Copyright (c) BEVFormer. All rights reserved.
# ------------------------------------------------------------------------
# modified by DenseBEV
# ------------------------------------------------------------------------
 
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from einops import rearrange, repeat
import numpy as np
import torch
from typing import Tuple, List, Optional
from torchvision.transforms.functional import rotate
import torch.nn as nn
from ..bevformer.modules import (
    MSDeformableAttention3D,
    TemporalSelfAttention,
    CustomMSDeformableAttention,
)
import copy
from mmcv.cnn import constant_init, bias_init_with_prob
from .nms import suppress_detections
from mmcv.runner import auto_fp16
from .utils import align_tensor


@TRANSFORMER.register_module()
class BEVPerceptionTransformer(BaseModule):
    def __init__(
        self,
        num_queries: int,
        encoder: dict,
        decoder: dict,
        embed_dims: int,
        num_cams: int,
        num_classes: int,
        nms_cfg: dict,
        num_reg_fcs: int,
        code_size: int,
        bev_h: int,
        bev_w: int,
        use_memory: bool,
        dataset: str = None,
        num_feature_levels: int = 4,
        use_cams_embeds: bool = True,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.num_reg_fcs = num_reg_fcs
        self.cls_out_channels = num_classes
        self.code_size = code_size
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.use_memory = use_memory

        # pass forward nms cfg
        decoder["nms_cfg"] = nms_cfg
        decoder["use_memory"] = use_memory

        self.dataset = dataset
        self.nms_threshold = nms_cfg["threshold"]

        if self.use_memory:
            assert decoder["transformerlayers"]["type"] == "MemoryDetrTransformerDecoderLayer"

        self.use_cams_embeds = use_cams_embeds

        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.fp16_enabled = False
        self.init_layers()

    def init_layers(self):
        # same as in head
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
    
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # two stage
        self.two_stage_cls_branch = copy.deepcopy(cls_branch)
        self.two_stage_reg_branch = copy.deepcopy(reg_branch)

        self.query_map = nn.Linear(self.embed_dims, self.embed_dims)

        # bevformer ones
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )  # in theory can be connected to backbone output; currently redundant weights
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for name, p in self.named_parameters():
            if "query_embedding" or "query_pos_z" in name:  # keep original init of normal
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(
                m,
                (
                    MSDeformableAttention3D,
                    TemporalSelfAttention,
                    CustomMSDeformableAttention,
                ),
            ):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)

        nn.init.xavier_uniform_(self.query_map.weight)
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)

        cls_branches = [self.two_stage_cls_branch]
        reg_branches = [self.two_stage_reg_branch]

        bias_init = bias_init_with_prob(0.01)
        for m in cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)

        for m in reg_branches:
            constant_init(m[-1], 0, bias=0)
            nn.init.constant_(m[-1].bias.data[2:], 0.0)


    @auto_fp16(apply_to=("mlvl_feats", "bev_queries", "prev_bev", "bev_pos"))
    def get_bev_features(
        self,
        mlvl_feats: List[torch.Tensor],
        bev_queries: torch.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: Tuple[float, float],
        bev_pos: torch.Tensor,
        prev_bev: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """returns the encoder bev queries"""
        grid_length_y, grid_length_x = grid_length

        bs = mlvl_feats[0].size(0)
        bev_queries = repeat(bev_queries, "hw c -> hw b c", b=bs)
        bev_pos = rearrange(bev_pos, "b c h w -> (h w) b c")

        # calculate shift between current and previous BEV using can bus data
        # can bus:
        # translation delta [0:3] (x,y,z)
        # rotation [3:7] (rot_x, rot_y, rot_z),
        # accel: [7, 10] - not required -> set zero
        # rotation_rate: [10: 13] - not required -> set zero
        # velocity: [13: 16] - not required -> set zero
        # patch angle (partially delta): [16:18] (yaw[rad], yaw[°] delta)

        # iterate over all samples in batch
        delta_x = np.array([each["can_bus"][0] for each in kwargs["img_metas"]])
        delta_y = np.array([each["can_bus"][1] for each in kwargs["img_metas"]])
        ego_angle = np.array(
            [each["can_bus"][-2] / np.pi * 180 for each in kwargs["img_metas"]]
        )

        translation_length = np.sqrt(delta_x**2 + delta_y**2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle

        # shift in bev space
        # number of bev cell shift
        shift_y_bev_cells = (
            translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y
        )
        shift_x_bev_cells = (
            translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x
        )
        # relative number of cell shift w.r.t. total number of cells per direction
        shift_y = shift_y_bev_cells / bev_h
        shift_x = shift_x_bev_cells / bev_w

        # combine shifts
        shift = bev_queries.new_tensor(np.array([shift_x, shift_y]))  # xy, bs -> bs, xy
        shift = rearrange(shift, "xy bs -> bs xy")

        # "rectify" previous bev w.r.t. to current
        if prev_bev is not None:
            # ensure correct shape (sequence first)
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = rearrange(prev_bev, "b hw c -> hw b c")

            # rotate
            for i in range(bs):
                rotation_angle = kwargs["img_metas"][i]["can_bus"][-1]
                # reshape for rotation
                tmp_prev_bev = rearrange(
                    prev_bev[:, i], "(h w) c -> c h w", h=bev_h, w=bev_w
                )
                tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle)
                # back to orig shape
                tmp_prev_bev = rearrange(tmp_prev_bev, "c h w -> (h w) c")
                prev_bev[:, i] = tmp_prev_bev

        # flatten features and spatial shapes into sequence
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            h, w = feat.shape[-2:]
            spatial_shape = (h, w)
            feat = rearrange(feat, "b n_cam c h w -> n_cam b (h w) c")

            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)

            feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)  # concat along sequence (h w) dim
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),  # add zero at the beginning
                spatial_shapes.prod(1).cumsum(0)[
                    :-1
                ],  # cumulative sum of h*w per feature layer
            )
        )
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device
        )  # convert to torch
        feat_flatten = rearrange(feat_flatten, "n_cam b hw c -> n_cam hw b c")

        bev_embed = self.encoder(  # BEVFormerEncoder
            bev_query=bev_queries,
            key=feat_flatten,
            value=feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,  # learned positional encoding of bev grid
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,  # prev bev embed
            shift=shift,
            **kwargs,
        )
        return bev_embed

    def pre_decoder(
        self, memory: torch.Tensor, bev_h: int, bev_w: int
    ) -> Tuple[dict, dict]:
        bs = memory.shape[0]

        # proposals already in inverse sigmoid space
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, bev_h, bev_w
        )

        enc_outputs_class = self.two_stage_cls_branch(output_memory)
        enc_outputs_coord_unact = self.two_stage_reg_branch(output_memory)
        # cls output is x,y,w,h,z,...
        enc_outputs_coord_unact[..., [0, 1, 4]] += (
            output_proposals  # both are in the "inverse sigmoid space"
        )

        scores, class_ids = enc_outputs_class.max(-1)
        scores = scores.sigmoid()
        
        topk = self.num_queries

        proposals = self.decode_boxes(enc_outputs_coord_unact)
        topk_score = []
        topk_coords_unact = []
        query = []

        # iterate over imgs for nms filtering of distinct queries
        for img_id in range(bs):
            single_proposals = proposals[img_id]
            single_scores = scores[img_id]
            map_memory = self.query_map(
                memory[img_id].detach()
            )  # detach to avoid gradient backprop from query to encoder

            keep_idxs = suppress_detections(single_proposals, 
                                            single_scores, 
                                            self.nms_threshold)
            keep_idxs = keep_idxs[:topk]

            topk_score.append(enc_outputs_class[img_id, keep_idxs])
            topk_coords_unact.append(enc_outputs_coord_unact[img_id, keep_idxs])
            query.append(map_memory[keep_idxs])

        query = align_tensor(query, topk)
        topk_coords_unact = align_tensor(topk_coords_unact, topk)
        topk_score = align_tensor(topk_score, topk)

        topk_anchor = topk_coords_unact.clone()
        topk_anchor = self.denormalize_center(topk_anchor)

        # detach for usage in decoder, dont process reference points to th end
        topk_coords_unact = topk_coords_unact.detach()

        reference_points = topk_coords_unact[:, :, [0, 1, 4]]
        
        reference_points[:, :, :3] = reference_points[:, :, :3].sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            value=memory,
            reference_points=reference_points,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
        )

        head_inputs_dict = {}
        if self.training:
            head_inputs_dict.update({
                "enc_outputs_class": topk_score,
                "enc_outputs_coord": topk_anchor,
            })

        return decoder_inputs_dict, head_inputs_dict

    def gen_encoder_output_proposals(
        self, memory: torch.Tensor, bev_h: int, bev_w: int
    ):
        bs = memory.shape[0]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(
                0, bev_h - 1, bev_h, dtype=torch.float32, device=memory.device
            ),
            torch.linspace(
                0, bev_w - 1, bev_w, dtype=torch.float32, device=memory.device
            ),
        )

        scale = repeat(
            torch.tensor([bev_h, bev_w], device=memory.device), "c -> bs 1 1 c", bs=bs
        )
        grid = rearrange([grid_x, grid_y], "c h w -> h w c")
        grid = grid[None].repeat(bs, 1, 1, 1)
        grid = (grid + 0.5) / scale

        # add height dimension with zeros
        # set to 0.5, so inverse sigmoid is 0
        reference_point_z = torch.ones_like(grid[..., :1]) * 0.5

        grid = torch.cat([grid, reference_point_z], dim=-1)

        # dims and angle prior
        proposals = grid
        proposals = rearrange(proposals, "bs h w c -> bs (h w) c")

        # inverse_sigmoid
        output_proposals = torch.log(proposals / (1 - proposals))

        output_memory = self.memory_trans_fc(memory)
        output_memory = self.memory_trans_norm(output_memory)

        return output_memory, output_proposals

    @auto_fp16(
        apply_to=(
            "mlvl_feats",
            "bev_queries",
            "prev_bev",
            "bev_pos",
        )
    )
    def forward(
        self,
        mlvl_feats: List[torch.Tensor],
        bev_queries: torch.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: Tuple[float, float],
        bev_pos: torch.Tensor,
        reg_branches: nn.ModuleList,
        memory_output: Optional[dict] = None,
        prev_bev: Optional[torch.Tensor] = None,
        given_bev=None,
        **kwargs,
    ):
        # bev encoder
        if given_bev is None:
            memory = self.get_bev_features(
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                **kwargs,
            )
        else:
            memory = given_bev

        decoder_inputs_dict, head_loss_inputs_dict = self.pre_decoder(
            memory, bev_h, bev_w
        )
        
        if memory_output is not None:
            decoder_inputs_dict.update(memory_output)

        decoder_inputs_dict["reg_branches"] = reg_branches

        inter_states, inter_references = self.decoder(**decoder_inputs_dict)

        return memory, inter_states, inter_references, head_loss_inputs_dict
    
    def denormalize_center(self, tmp: torch.Tensor) -> torch.Tensor:
        cx = self.pc_range[0] + (
            tmp[..., :1].sigmoid() * (self.pc_range[3] - self.pc_range[0])
        )
        cy = self.pc_range[1] + (
            tmp[..., 1:2].sigmoid() * (self.pc_range[4] - self.pc_range[1])
        )
        cz = self.pc_range[2] + (
            tmp[..., 4:5].sigmoid() * (self.pc_range[5] - self.pc_range[2])
        )
        return torch.cat([cx, cy, tmp[..., 2:4], cz, tmp[..., 5:]], dim=-1)
    
    def decode_boxes(self, tmp: torch.Tensor) -> torch.Tensor:
        # decode to real world space and dims
        # pc range is set in the head
        # center
        cx = self.pc_range[0] + (
            tmp[..., :1].sigmoid() * (self.pc_range[3] - self.pc_range[0])
        )
        cy = self.pc_range[1] + (
            tmp[..., 1:2].sigmoid() * (self.pc_range[4] - self.pc_range[1])
        )
        cz = self.pc_range[2] + (
            tmp[..., 4:5].sigmoid() * (self.pc_range[5] - self.pc_range[2])
        )

        # size
        w = tmp[..., 2:3].exp()
        l = tmp[..., 3:4].exp()
        h = tmp[..., 5:6].exp()

        # rotation
        rot_sine = tmp[..., 6:7]
        rot_cosine = tmp[..., 7:8]
        rot = torch.atan2(rot_sine, rot_cosine)

        # combine
        if tmp.size(-1) > 8:  # nuscenes schema with velocity
            vx = tmp[..., 8:9]
            vy = tmp[..., 9:10]
            proposal = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
        else:
            proposal = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)

        return proposal
